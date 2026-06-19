# EvolBA — Brainstorming Session

## 0. Corrections to the baseline description

- The fitness function is **not binary**. It is the **distance from the original image** (L2 or similar), with a large penalizer added when the candidate is non-adversarial. This gives a continuous, rankable fitness landscape with a discontinuity at the decision boundary.
- The binary hard-label is used only to determine which side of the boundary a candidate is on. The ranking signal within each side is the distance metric.

---

## 1. CMA-ES variants (Evolutionary Strategies only)

### 1.1 Standard CMA-ES — why it struggles here

Two structural problems in this setting:

- **n² covariance matrix**: For CIFAR-10, n = 3072. The covariance matrix has ~9.4M entries. With population λ ≈ 28, the rank-μ update supports only a rank-14 estimate of a 3072-dimensional matrix. The rest is noise.
- **Fitness discontinuity at the boundary**: the transition from adversarial to non-adversarial creates a sharp discontinuity that disrupts the smooth ranking CMA-ES relies on.

### 1.2 (1+1)-CMA-ES

One parent, one offspring per generation. Accept if better, reject otherwise. Step size via 1/5 success rule. Rank-1 covariance update only.

**Why it fits:** 1 query per generation instead of λ=28. On a 1500-query budget, (1+1)-CMA-ES gets ~1400 adaptation steps vs ~50 for standard CMA-ES. The accept/reject mechanism maps naturally to the adversarial/non-adversarial split. No population diversity, but extremely query-efficient.

### 1.3 sep-CMA-ES (Separable / Diagonal)

Replaces the full covariance matrix with a diagonal: `C = diag(d₁, ..., dₙ)`. Each dimension is scaled independently; no cross-dimension correlations. O(n) storage, O(n) sampling, no eigendecomposition needed.

**Why it fits:** With λ ≈ 28 samples and n = 3072 parameters, the full matrix is statistically unestimable. The diagonal is estimable and captures per-pixel importance, which is often the dominant structure in adversarial perturbations.

### 1.4 VD-CMA (Vector-Diagonal)

Extends sep-CMA-ES with one rank-1 term: `C = D(I + vvᵀ)D`. Adds a single learned direction on top of the diagonal. O(n) storage and sampling.

**Why it fits:** Captures the one dominant correlation axis — likely the direction toward the decision boundary — while keeping the diagonal structure for all other dimensions. Useful especially for targeted attacks where there is one semantically meaningful perturbation direction.

### 1.5 LM-CMA (Limited Memory)

Approximates C using the last m evolution path vectors (m ≈ 4·log(n) ≈ 44 for n=3072). O(mn) storage instead of O(n²). Adapts to the problem's effective dimensionality over time.

**Why it fits:** The adversarial boundary has O(log n) effective dimensions that matter. LM-CMA learns those directions without filling a full n×n matrix with noise. Better than sep-CMA-ES when perturbation directions have real correlations (e.g. spatially adjacent pixels in texture regions).

### 1.6 Active CMA-ES (aCMA-ES)

Adds **negative weights** for the worst-ranked individuals in the rank-μ update, actively pushing the covariance away from bad directions. Framing for the boundary normal update (see Section 4).

### 1.7 IPOP / BIPOP restarts

Meta-strategies applied on top of any CMA-ES variant.

- **IPOP**: when stagnation is detected (σ collapse or no improvement in T generations), restart with population size doubled.
- **BIPOP**: alternates between large-population global exploration and small-population local exploitation.

**Why they fit:** Stagnation is a structural risk in this setting — the binary fitness signal can cause σ to collapse. Restarts cost nothing extra (pure bookkeeping) and are high-value insurance.

---

## 2. Lower-dimensional structured subspace

### Motivation

Working in the full pixel space (n = 3072 for CIFAR-10) is wasteful:
- The covariance matrix becomes unestimable.
- Salt-and-pepper pixel-by-pixel perturbation is not effective — neural network classifiers are not sensitive to spatially incoherent noise.
- The adversarially sensitive directions are structured (smooth, spatially coherent, semantically meaningful).

### 2.1 DCT basis (Option A)

Use the top-k DCT (Discrete Cosine Transform) frequency components as the search subspace. Each basis vector is a global sinusoidal pattern affecting all pixels simultaneously. The top-k components (sorted by spatial frequency) capture the perceptually important image structure.

- **k**: tunable, recommended k ∈ [100, 400] for CIFAR-10.
- **Pros**: fixed, free to compute, frequency-domain, globally mixed, perceptually motivated (high-frequency components are imperceptible anyway).
- **Cons**: model-agnostic, does not adapt to the specific image or model sensitivity.

### 2.2 SLIC superpixels (Option B)

SLIC (Simple Linear Iterative Clustering) segments the image into K compact, semantically coherent regions by k-means clustering in the 5D space [L, a, b, x, y] (CIELAB color + spatial coordinates). Each basis direction perturbs all pixels in one superpixel uniformly.

**Algorithm:**
1. Initialize K cluster centers on a regular grid (spacing S = √(N/K)).
2. Perturb each center to the lowest-gradient 3×3 neighbor (avoid placing centers on edges).
3. Assign each pixel to the nearest center within a 2S×2S search window using distance `D = √((ΔL²+Δa²+Δb²)/m² + (Δx²+Δy²)/S²)`. O(N) complexity.
4. Recompute centers as pixel means. Repeat ~10 iterations.
5. Post-process: merge small disconnected fragments.

**Why it fits:** Object classifiers are sensitive to semantic regions, not individual pixels. SLIC superpixels trace object boundaries and adapt to the specific image being attacked. For CIFAR-10 at 32×32, K=32–64 gives ~16 pixels per superpixel and a 32–64 dimensional search space.

### 2.3 Combined DCT + SLIC basis (Recommended)

Concatenate the top-k₁ DCT vectors and the K SLIC superpixel vectors, then orthogonalize via QR decomposition and keep the top-k combined vectors.

- DCT captures the frequency/imperceptibility prior.
- SLIC captures the semantic/model-sensitivity prior.
- Together they form a richer subspace than either alone, still at zero query cost.
- Total dimensionality k is tunable (e.g. k = 200).

### 2.4 Other options (lower priority)

- **Blurred random projections**: sample random Gaussian vectors, apply Gaussian blur (spatial scale σ_blur), orthogonalize. Free, controllable, simpler than DCT but less principled.
- **Graph Laplacian eigenvectors**: treat the image as a graph with pixel similarity weights. Eigenvectors of the graph Laplacian are the natural oscillation modes of the image — spatially smooth, globally mixed, and image-adaptive.

---

## 3. Directional bias — integrating points 1 and 2

### Geometric setup at the start of Phase 3

After Phase 2 (binary search), we sit at `x_b` on the decision boundary. Three known facts:
- Direction `(x_b - x)` → deeper adversarial, strictly increases distance, **never useful**.
- Direction `(x - x_b)` → crosses boundary immediately, non-adversarial, **wastes a query**.
- Directions **perpendicular to** `(x - x_b)` → tangent to the decision boundary, **this is where the useful search lives**.

### Initial covariance

Let `û = (x - x_b) / ||x - x_b||` (unit vector toward original image). Initialize:

```
C₀ = I - (1 - ε) · û·ûᵀ,   ε ≈ 0.01
```

This gives ~1% variance in the `û` direction and ~100% variance in all perpendicular (boundary-tangent) directions. Integrated with the DCT+SLIC subspace: project `û` into the k-dimensional subspace to get `û_sub`, then apply the same formula within the subspace.

### Half-space reflection

For each sampled candidate step `δ`, compute the dot product with the approaching direction:

```
d = dot(δ, û)
```

- If `d < 0` (step moves away from x): **reflect** across the boundary hyperplane:
  ```
  δ ← δ - 2·d·û
  ```
  Reflection is preferred over rejection because it preserves step magnitude and avoids biasing the covariance update.
- If `d ≥ 0`: accept as-is.

This constraint is permanent throughout Phase 3 — going away from x can never help.

### Initial mean shift

Start the CMA-ES mean at a small positive offset:

```
m₀ = x_b + δ_small · û,   δ_small ≈ 0.005 · ||x_b - x||
```

This biases samples toward a small positive dot product with `û` (slightly approaching), consistent with the "prefer d slightly positive" strategy.

### Temporal evolution

CMA-ES naturally increases variance in `û` over time as it accumulates successful approaching steps via the rank-μ update. No manual schedule needed — the algorithm learns when to approach more aggressively.

---

## 4. Exploiting non-adversarial query information

### The gap in standard CMA-ES

When a candidate is non-adversarial, standard CMA-ES applies a fitness penalty and treats the candidate like any other bad individual. But it contains additional geometric information: the direction `(x_i - x_b)` is approximately the **local boundary normal** (pointing from adversarial toward clean region).

### Boundary normal estimation

After evaluating the population, collect all non-adversarial candidates and estimate the boundary normal:

```
n̂ = mean over non-adversarial xᵢ of: (xᵢ - x_b) / ||xᵢ - x_b||
```

### Negative rank-1 covariance update (aCMA-ES framing)

Apply an additional covariance update that deflates variance in the estimated boundary normal direction:

```
C ← C - c_wall · n̂·n̂ᵀ
```

Then project back to positive definite (clip negative eigenvalues). This tells CMA-ES "don't sample there — that's where the wall is." The boundary normal estimate improves over generations as more non-adversarial candidates accumulate direction information.

**Why this is still an ES:** This is an extension of active CMA-ES (aCMA-ES), which uses the worst-ranked individuals for negative rank-μ updates. Here we use the non-adversarial candidates specifically for a boundary-informed negative update. No surrogate model is built; the update operates directly on the distribution.

---

## 5. Sequential island model with recombination-on-stagnation

### Setup

- Initialize K starting points cheaply (Phase 1+2 costs < 100 queries total per island, negligible).
- Maintain an archive of K CMA-ES states: `{(mₖ, σₖ, Cₖ, best_k)}`.
- Run only **one active island** at a time — others are frozen.

### Plateau detection

Trigger when either:
- Step size σ falls below threshold σ_min (distribution collapsed), or
- No improvement in the global best distance over the last T=10–15 generations.

Detect early to avoid wasting queries on a dead island.

### Recombination mechanism

When island A stagnates, switch to island B and inject information from A:

1. **Reset mean**: to the global best point found so far (from A or B).
2. **Inject cross-island direction**: add rank-1 update to C_B using the direction between the two islands' means:
   ```
   direction = (m_A - m_B) / ||m_A - m_B||
   C_new = C_B + α · direction · directionᵀ
   ```
   This tells the new island "explore along the axis connecting the two previously explored regions."
3. **Reset step size**: to a moderate value — do not carry over σ-collapse from the stagnated island.

**Key property:** the covariance learned by island B (its boundary geometry knowledge) is preserved. Only the mean is redirected and one new direction is injected.

### Activation order

Always continue from the island with the best current `best_k` (greedy). Switch only on stagnation.

### Query efficiency

- Only one island runs at a time → no parallel query cost.
- Recombination events cost zero queries (pure bookkeeping).
- Total query cost = sequential sum of active queries across islands, same as a single island but with smarter restarts.
- Advantage over IPOP: IPOP resets everything; this model preserves learned covariance structure and injects cross-island directional information.

---

## 6. Unified subspace across all phases

### The question

Phase 1 uses some noise to find an initial adversarial point. Phase 3 uses the DCT+SLIC structured subspace. Is this inconsistent?

### Resolution

The phases have **different objectives**:
- **Phase 1**: cross the decision boundary fast, magnitude is large → direction matters less → any noise works.
- **Phase 2**: 1D binary search along a fixed line → no noise model applies, inherently incomparable.
- **Phase 3**: minimize distance while staying adversarial, magnitude is small → direction is critical → structured subspace is essential.

### Recommendation: partial unification

Use the structured DCT+SLIC basis in Phase 1 as well, **with a fallback** to unstructured noise if the boundary is not found within a query limit. Benefit: if Phase 1 finds its initial adversarial point within the subspace, the binary search direction in Phase 2 inherits that structure, and Phase 3 starts from a point already "in-distribution" for the structured search. This avoids Phase 3 starting from a boundary point that has poor projection onto the subspace.

---

## 7. Hybrid ES + binary search (Memetic ES)

### The inefficiency in standard CMA-ES

The population serves two purposes simultaneously: (1) direction finding and (2) step size calibration. Both are estimated noisily by random sampling. But binary search is a more precise tool for step size calibration along a known direction.

### Hybrid per-generation loop

```
each generation:
  1. sample λ candidates from N(m, σ²C)          → λ queries
  2. apply half-space reflection (d < 0 reflected)
  3. query model for each candidate               → λ queries (counted above)
  4. split into adversarial set A and non-adversarial set N
  5. identify x_best from A (closest to x)
  6. binary search between x_b and x_best        → B ≈ 7 queries
  7. update CMA-ES mean using the refined point
  8. standard rank-μ update using all A
  9. boundary normal negative update using N (Section 4)
```

### Direction bisection

Given one adversarial direction `d₁` and one non-adversarial direction `d₂`, binary search their interpolation:

```
d(α) = normalize(α·d₁ + (1-α)·d₂),   binary search on α ∈ [0,1]
```

Finds the sharpest direction that stays adversarial. Combined with step size binary search gives the most aggressive feasible move along the boundary.

### Classification

This is a **Memetic ES** (also called Lamarckian ES) — a well-established ES variant that adds local refinement inside the evolutionary loop. Every refinement step queries the real model; no predictions substitute for queries. Firmly within evolutionary computation scope.

---

## 8. Non-convexity handling

When the local decision boundary is non-convex, the standard mechanisms are:

- **Larger population** (more λ): samples more directions per generation, less likely to be stuck in a local tangent direction.
- **Island switching**: switch to a different island that explores a different region of the boundary (Section 5).
- **Step size increase**: reset σ upward (via IPOP restart or manual schedule) to jump out of the local neighborhood.
- **Direction bisection between islands**: the cross-island direction (Section 5) may point along the non-convex arc.

---

## 9. Full Phase 3 loop (integrated)

```
Initialize:
  û = (x - x_b) / ||x - x_b||
  C₀ = I - (1-ε)·û·ûᵀ   projected into DCT+SLIC subspace
  m₀ = x_b + 0.005·||x_b - x||·û
  σ₀ = moderate initial value
  K islands initialized cheaply from perturbed boundary points

Each generation (active island only):
  1. Sample λ candidates in DCT+SLIC subspace from N(m, σ²C)
  2. Reflect all candidates with dot(δ, û) < 0
  3. Map candidates back to pixel space via inverse DCT + SLIC reconstruction
  4. Query model for each candidate
  5. Split: adversarial set A, non-adversarial set N
  6. Binary search between x_b and best candidate in A   [B extra queries]
  7. Update mean using binary-search refined point
  8. Standard rank-μ covariance update from A
  9. Boundary normal negative update from N
  10. CSA step size update
  11. Check plateau → if stagnated: recombine with next island (Section 5)
```
