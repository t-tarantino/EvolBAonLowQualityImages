# Stage 2 — Query Efficiency: Same Quality, Tight Budget

## Goal

Stage 1 proved that meaningful optimization is possible: the subspace sep-CMA-ES
consistently reduces L2 beyond the Phase 2 starting point. But it also revealed that
almost all improvement is front-loaded — the running-best curve is essentially flat
after the first 10–20 generations (~100–200 queries), while the remaining ~800–900
queries of the P3 budget add nothing.

Stage 2's single objective:

> **Achieve the same (or better) adversarial quality as Stage 1 using ≤ 500 queries
> in Phase 3, by exploiting the front-loading structure.**

Success criterion: median L2 after Phase 3 ≤ Stage 1 median at the same query count,
measured on the same 200-image set (standard + robust WRN-28-10).

---

## Inherited configuration (Stage 1 best)

```
Basis:       DC-free DCT (K_DCT=20) + grid superpixel (K_SP=20) → k ≈ 40
Fitness:     L2 (= ‖θ‖₂, exact quadratic bowl in subspace)
Constraint:  L∞(x_cand, x_orig) ≤ 0.10 (hard rejection)
Early stop:  SSIM ≥ 0.95
ES variant:  sep-CMA-ES, λ=10, diagonal D_sub ∈ ℝ^k
P3 budget:   1000 queries (Stage 1 baseline)
```

---

## Stage 2 experiments

### Experiment 1 — Stagnation detection and budget reallocation

**Motivation:** if 90% of improvement happens in gens 1–20, the remaining budget
should be redeployed rather than wasted.

**Implementation:**
- Detect plateau: no improvement in global best L2 over T consecutive generations
  (try T = 10, 15, 20).
- On plateau trigger: restart Phase 3 with a different initialization drawn from the
  corruption family (best unused family member by SSIM, then hybrid fallback).
- Track total query count across all restarts — budget is shared, not per-restart.

**What to measure:**
- Improvement per restart attempt (does the second start find a better solution?)
- Optimal T for plateau detection across the 200-image set
- Fraction of images where restart produces measurable gain

---

### Experiment 2 — (1+1)-CMA-ES in subspace

**Motivation (from Stage 1 open questions):** with k = 40 and 1 query per generation,
a 500-query budget gives ~500 individual adaptive steps. Sep-CMA-ES with λ=10 gives
only ~50 generations for the same budget. The 1/5 success rule is well-calibrated
at k = 40.

**Implementation:**
- Replace the λ=10 population loop with a (1+1) loop:
  - sample one offspring from N(m, σ²·D_sub)
  - query model → adversarial / clean
  - if adversarial AND L2 improves: accept, update mean, rank-1 covariance update
  - update step size via 1/5 success rule: σ *= exp((success - 0.2) / 0.4)
- Keep L∞ constraint and SSIM early stop.

**Variants to compare:**
- (1+1) with diagonal D_sub (analogous to sep)
- (1+1) with full k×k C (feasible at k=40 with rank-1 updates only)

**What to measure:**
- L2 curve vs query count vs sep-CMA-ES λ=10 at equal budgets (200, 500, 1000)
- Success rate and improvement distribution across 200 images

---

### Experiment 3 — Full k×k covariance CMA-ES (not sep)

**Motivation:** k ≈ 40 is small enough that the full k×k covariance matrix is
computationally tractable (~1600 entries). Cross-direction correlations within the
subspace may exist (e.g. correlated DCT frequencies, correlated superpixel regions
sharing an object boundary) and learning them could accelerate convergence.

**Implementation:**
- Replace diagonal D_sub with full C_sub ∈ ℝ^(k×k)
- Standard rank-μ update with μ = λ/2 selected individuals
- Eigendecomposition at each generation (cheap at k=40)

**Ablation:**
- sep (diagonal) vs full covariance at budgets 200, 500, 1000
- Measure whether full covariance provides benefit early (gens 1–20) or only later

---

### Experiment 4 — IPOP restarts

**Motivation:** when the (1+1) or sep variant stagnates (σ < σ_min), restart with
doubled population size. This is the IPOP meta-strategy applied within the subspace.

**Implementation:**
- Start with λ=4 (smallest viable population)
- On stagnation: λ ← 2λ, reset σ to σ₀, reset mean to best found so far
- Cap at λ_max = 40 (= k, one sample per subspace dimension)
- Shared query budget across restarts

**What to measure:**
- Does IPOP find better solutions than a single run at the same total budget?
- How often does stagnation trigger, and at what generation?

---

### Experiment 5 — Image-adaptive SLIC superpixels

**Motivation (from Stage 1 open question):** the current grid-superpixel approximation
is image-agnostic. True SLIC (CIELAB + spatial k-means, K=20 superpixels) would trace
object boundaries and adapt to the specific image. Hypothesis: a better basis projection
quality → fewer subspace dimensions needed → faster covariance adaptation.

**Implementation:**
- Implement SLIC: initialize centers on regular grid, 10 iterations of assignment +
  center update in 5D [L, a, b, x, y] with distance parameter m=10, S=√(N/K).
- Post-process: merge fragments < 6 pixels.
- Build superpixel basis vectors (K=20 vectors ∈ ℝ^3072, one per segment).
- QR-orthogonalize with the DCT vectors as before.

**Compare:**
- Grid superpixel (Stage 1) vs SLIC superpixel on L2 improvement at 200, 500 queries
- Subspace projection quality: ||x_adv - proj(x_adv)||₂ as a diagnostic

---

### Experiment 6 — Query budget profiling (per phase)

**Motivation:** we have never precisely measured Phase 1 and Phase 2 query costs on
the 200-image set. This is required to design a total query budget with the correct
phase allocation.

**Measurements:**
- Phase 1: mean and std of queries to first adversarial point, per corruption type,
  per model; fraction requiring hybrid fallback
- Phase 2: mean and std of binary search steps × 2 (one query each), per model
- Phase 3: queries to SSIM early stop vs full budget, per model
- Total: histogram of total queries across all 200 images for the full pipeline

**Output:** a stacked bar chart showing P1 / P2 / P3 query share, separately for
standard and robust model.

---

## Evaluation protocol

All experiments use the same 200-image set as Stage 1 (jointly correctly classified
by both WRN-28-10 models). Metrics reported at query counts 200, 500, 1000:

| Metric | Description |
|---|---|
| Median L2 improvement | L2(P2 start) − L2(P3 best), median over 200 images |
| Attack success rate | Fraction of images with any Phase 3 improvement |
| SSIM early stop rate | Fraction of images where SSIM ≥ 0.95 is triggered |
| Mean query count | For the same quality as Stage 1 at 1000 queries |

---

## Notebooks

| Notebook | Purpose | Key question |
|---|---|---|
| `utils_stage2.py` | Shared utilities: Oracle, Phase 1/2, subspace, sep-CMA-ES | — |
| `exp1_stagnation_ipop.ipynb` | Stagnation detection + family restart + IPOP | Does restarting with saved queries beat running to exhaustion? |
| `exp2_population_subspace.ipynb` | Joint (λ, k) grid study | What is the optimal λ/k ratio at a 500-query budget? |
| `exp3_one_plus_one.ipynb` | (1+1)-ES, (1+1)-sep, (1+1)-full-CMA-ES | Does 1 query/gen beat λ=10 at equal total budget? |
| `exp4_full_covariance.ipynb` | Full k×k CMA-ES vs sep-CMA-ES ablation | Do cross-subspace correlations exist and help? |
| `exp5_slic.ipynb` | DCT+SLIC vs DCT+grid superpixels | Does image-adaptive segmentation improve basis quality? |
| `exp6_query_profiling.ipynb` | Per-phase query profiling on 200 images | Which phase is the bottleneck, and how front-loaded is Phase 3? |
| `stage2_summary.ipynb` | Unified ranking of all Stage 2 variants | What is the best method at 200 / 500 / 1000 queries? |

---

## Priority order

1. **exp3** — (1+1)-CMA-ES: highest expected impact; 1 query/gen × 1000 = 1000 adaptation
   steps vs ~100 generations for sep-CMA-ES λ=10 at the same budget.
2. **exp1** — Stagnation + IPOP: almost no extra code; reallocates wasted tail budget.
3. **exp6** — Query profiling: confirms budget assumptions before scaling to 200 images.
4. **exp2** — (λ, k) grid: establishes theoretically grounded operating point.
5. **exp4** — Full covariance: diagnostic; upper bound for sep ablation.
6. **exp5** — SLIC: highest cost; likely incremental on 32×32 CIFAR-10.

---

## Open questions deferred to Stage 3

- Targeted attacks (query overhead vs untargeted)
- Cross-model transferability of adversarial examples
- L∞ perceptual threshold calibration (human study or BAPPS)
- Warm-starting CMA-ES from prior successful attacks (meta-learning)
- Per-class initializer selection
