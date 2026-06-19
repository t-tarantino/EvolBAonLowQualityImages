# Phase 0 — Research Resume

> **Purpose.** This file collects every finding, number, and decision from the Phase 0
> studies so that future work can continue without re-reading four notebooks.

---

## Experimental Setup

| Item | Value |
|---|---|
| Dataset | CIFAR-10 test set (10 classes, 32×32 px, images in [0, 1]) |
| Standard model | WRN-28-10 (`Standard`, RobustBench) — normalises internally |
| Robust model | WRN-28-10 (`Wang2023Better_WRN-28-10`, TRADES ℓ∞, RobustBench) |
| Query budget | 3 000 per full EvolBA run (Studies 1–4); 41 per Phase-1 trial (Study 5) |
| Study 5 sample | 200 CIFAR-10 images, 20 per class, class-stratified, standard-model-correct only |
| Codebase entry | `evolba_baseline.py` — reference implementation, unmodified from paper |

---

## Study 1 & 2 — Query Budget and Fitness Curves

**Notebook:** `study1_query_cost_and_fitness.ipynb`  
**Sample:** 8 images × 2 models = 16 attacks, up to 3 000 queries each.

### Where the budget goes (Study 1)

| Phase | Standard q/gen | Robust q/gen | Notes |
|---|---|---|---|
| Offspring evaluation | ~28 | ~28 | λ = 28, always adversarial (frac_adv = 1.0 exactly) |
| ξ-shrink loop | **63.6** | **143.2** | Dominant cost; exceeds a full offspring evaluation |
| Backtracking (interleaved BS) | 0.57/gen | **1.93/gen** | Robust hits TAU=3 cap most of the time |
| **Total per generation** | ~92 | **~173** | Robust costs ≈1.9× per generation |
| **Mean generations** | ~23 | ~12 | Same budget → fewer generations on robust |

Key finding: ξ-shrink + backtracking account for ~75% of every query spent on both models.

### How fitness improves (Study 2)

| Model | Init L2 | Final L2 | Improvement | Shape |
|---|---|---|---|---|
| Standard | ~2.0 | ~0.7 | ~65% | Sharp breakthrough around gen 12–13, then plateau |
| Robust | ~7.0 | ~6.7 | ~4% | Near-flat throughout |

Cost-effectiveness (ΔL2/query) falls as the run progresses — the back third of any run
buys measurably less than the first two-thirds. Population tracks incumbent in lock-step
(move/backtrack logic is not the bottleneck).

### Root cause hypothesis

Every offspring is adversarial (deep inside the adversarial region), yet the directed mean-shift
`m + ξv` routinely overshoots back across the boundary.
The reason: both moves share the same scalar ξ, but operate at wildly different scales.

- Offspring perturbation norm: `ξ · √n ≈ 55 · ξ` (n = 3072)
- Directed step magnitude: `ξ · ‖v‖ = ξ`

**The factor of ~55 mismatch means one ξ cannot simultaneously be a good exploration radius
and a good directed-step size.**  The Eq. 4 schedule `ξ = dist/√t` and `cmu_scale` were tuned
on VGG19 at 224×224 (n ≈ 150 528, √n ≈ 388) — never re-derived for CIFAR-10 (√n ≈ 55).

**Recommended fix:** decouple exploration scale (offspring sampling) from directed-step scale
(mean shift), or add an explicit `√n` correction to Eq. 4.

---

## Study 3 — Frequency-Band Sensitivity and Resolution

**Notebook:** `study0_frequency_resolution.ipynb`

### Part A — Band sensitivity (CIFAR-10, 32×32)

Frequency bands defined by normalized radial frequency ρ in the 2D DCT domain.

| Model | Band sensitivity trend | Best band for Phase 1 |
|---|---|---|
| Standard | Monotonically cheaper as frequency **increases** (~4× gain from low→high) | High-frequency (ρ ≥ 0.8) |
| Robust | Roughly flat; mildly **prefers low/mid**, disfavors high | No band gives a meaningful edge |

This is a near **mirror-image** relationship between models.
Standard CNNs' well-documented high-frequency sensitivity drives the standard model result.
Adversarial training (TRADES) partially corrects this: robust model's frequency profile is
flattened and inverted — high-freq injection is if anything *harder* against it.

**Implication:** a band-limited high-frequency direction is a strictly better Phase-1
initialiser than general corruption for the standard model. For the robust model,
Phase-1 initialisation quality matters less — Phase 3 is the bottleneck.

### Part B — Resolution dependence

Holding per-pixel RMS constant and upscaling 32→224 px: all three bands (low, mid, high)
degrade SSIM in near lock-step (spread ≤ 0.05 SSIM across bands vs a ~0.2–0.43 total drop).
**The paper's claim that HF components occupy a disproportionate share of a 32×32 image is
not supported as a purely perceptual (SSIM-measurable) phenomenon.**

---

## Study 4 — Failure Analysis

**Notebook:** `study4_failure_analysis.ipynb`  
**Sample:** 100 images × 2 models × 2 init methods = 400 attacks.  
Failure buckets: **hard** (Phase 1 failed) · **flat** (IR < 5%) · **partial** (5–50%) · **good** (IR ≥ 50%).

### Failure rates

| Condition | hard | flat | partial | good | mean IR |
|---|---|---|---|---|---|
| standard / uniform | 0% | 11% | 59% | **30%** | **0.386** |
| standard / fractal | 0% | 22% | 64% | 14% | 0.274 |
| robust / uniform | 10% | 47% | 42% | 1% | 0.088 |
| robust / fractal | **3%** | 49% | 47% | 1% | 0.090 |

### Key findings

1. **Phase 3 is the main bottleneck for the robust model.**
   IR ≈ 0.08 regardless of init method; 96% of robust attacks are flat+partial.
   Fixing Phase 1 alone will not rescue robust attacks — Phase 3 needs fundamental changes.

2. **Fractal init hurts standard Phase 3.** good: 30%→14%, mean IR: 0.39→0.27.
   Fractal init should not be used with the standard model.
   For robust, it reduces hard failures (10%→3%) but doesn't help Phase 3.

3. **Attractor class mechanism (uniform init).**
   Random noise in [0,1]³ collapses to two dominant attractor classes:
   - Robust: `bird` (IR = 0.131) — bad attractor; all 10 robust hard failures are image=bird, init=bird
   - Standard: `cat` (IR = 0.373) — moderate attractor
   Attractor identity is model-specific. Diverse-attractor sampling would reduce hard failures.

4. **init_l2 is NOT a good predictor of Phase 3 success.**
   Pearson r(init_l2, IR): standard/uniform = +0.156, standard/fractal = +0.450, robust/uniform = −0.021.
   Correlations are positive or near-zero — starting closer does not help Phase 3.
   (Study 5's effort to minimise init_l2 is motivated by naturalness, not Phase 3 quality.)

5. **70–85% of failures are shared across both models.**
   Success is almost exclusively standard-model-only. The shared failure set is geometric, not model-specific.

6. **ξ halvings cannot serve as an early-stopping signal.**
   Median halvings in gen 1–3: flat runs = 4 (standard), 0 (robust); good runs = 0 (standard), 0 (robust).
   Implement a `best_l2 plateau` detector (< 0.1% improvement over 5 generations) instead.

### Priority fixes (from Study 4)

1. **Phase 3 direction quality** — larger λ, adaptive cμ, or directed exploration bonus for offspring that cross the boundary.
2. **Attractor diversity** — diverse-attractor Phase 1 sampler that seeks multiple adversarial classes before choosing the closest start.
3. **Stagnation detection** — best_l2 plateau detector to trigger restart rather than burning remaining budget.

---

## Study 5 — Corruption-Based Phase 1 Initialisation

**Notebook:** `study5_corruption_init.ipynb`  
**Sample:** 200 images (20/class), 10 corruptions, 2 models = 4 000 trials.  
**Budget per trial:** 1 (fast-fail) + 15 (severity binary search) + 25 (image binary search) = **41 queries**.  
**Saved outputs:** `STAGE_0/outputs/study5_corruptions/`  
- `results.parquet` — 4 000 rows × 11 columns (row_id, image_idx, model, corruption, true_class, success, severity_star, init_l2, init_ssim)
- `boundaries.npy` — (4 000, 3, 32, 32) float32, decision-boundary images
- `lpips_scores.npy` — (4 000,) float32, LPIPS at boundary (post-hoc, AlexNet)

### Protocol

1. `corrupt(x, s=1.0)` — fast-fail: if model still correct at maximum severity, skip (immune).
2. Binary search on severity s ∈ [0,1] (15 steps, precision ≈ 3×10⁻⁵) → minimum adversarial s*.
3. Binary search in image space from x_orig to corrupt(x, s*) → decision-boundary point x_bnd.
4. Measure init_l2 = ‖x_bnd − x_orig‖₂, init_ssim (SSIM), init_lpips (LPIPS).

Fixed seed per (image_idx, corruption) ensures stochastic corruptions use the same noise
field at all severity levels, guaranteeing monotone binary search.

### Success rates

| Corruption | Standard | Robust |
|---|---|---|
| gaussian_noise | 90% | 90% |
| salt_pepper | 91% | 90% |
| brightness | 90% | 90% |
| contrast_low | 90% | 90% |
| fog | 90% | 90% |
| jpeg | 82% | 46% |
| gaussian_blur | 87% | 80% |
| fractal_blend | 89% | 79% |
| posterize | 70% | 25% |
| **hue_shift** | **12%** | **13%** |

hue_shift is 88% immune — 180° hue rotation rarely crosses the decision boundary.

### Metric comparison (standard model, successful trials)

| Corruption | Median L2 | L2 rank | Median SSIM | Median LPIPS | LPIPS rank |
|---|---|---|---|---|---|
| jpeg | 2.47 | 1 | 0.910 | 0.013 | 3 |
| **gaussian_blur** | **2.54** | **2** | **0.906** | **0.073** | **6** ← misleading |
| fractal_blend | 3.43 | 3 | 0.847 | **0.004** | **1** |
| gaussian_noise | 3.55 | 4 | 0.815 | 0.010 | 2 |
| salt_pepper | 3.87 | 5 | 0.829 | 0.033 | 4 |
| hue_shift | 4.70 | 6 | 0.901 | 0.114 | 7 |
| posterize | 7.21 | 7 | 0.755 | 0.038 | 5 |
| contrast_low | 11.1 | 8 | 0.340 | 0.253 | 8 |
| fog | 11.1 | 9 | 0.340 | 0.253 | 9 |
| brightness | 25.5 | 10 | 0.384 | 0.274 | 10 |

**Key L2 vs LPIPS discrepancy:** gaussian_blur ranks 2nd by L2 but 6th by LPIPS.
Blurring moves each pixel a small amount (low L2) but destroys texture structure (high LPIPS).
The visual gallery confirms this — gaussian_blur boundary images look heavily blurry despite
short pixel-distance. **Use LPIPS as the primary perceptual quality indicator.**

### Metric comparison (robust model, successful trials)

| Corruption | Median L2 | Median SSIM | Median LPIPS |
|---|---|---|---|
| hue_shift | 4.52 | 0.908 | 0.092 |
| jpeg | 4.54 | 0.743 | 0.074 |
| gaussian_blur | 4.81 | 0.670 | **0.246** ← worst LPIPS |
| fractal_blend | 7.01 | 0.533 | **0.061** |
| gaussian_noise | 8.86 | 0.532 | **0.061** |
| salt_pepper | 6.90 | 0.641 | 0.094 |
| contrast_low/fog | 8.00 | 0.677 | 0.073 |
| posterize | 8.96 | 0.703 | 0.055 |
| brightness | 21.9 | 0.609 | 0.127 |

### Ensemble quality (mean-min-L2 metric, PENALTY=30 for immune)

**Standard model** — best single: fractal_blend (score 6.27):

| k | Score | Marginal gain | Best set |
|---|---|---|---|
| 1 | 6.27 | — | {fractal_blend} |
| **2** | **2.84** | **3.43** | {gaussian_blur, fractal_blend} |
| 3 | 2.31 | 0.53 | {salt_pepper, gaussian_blur, fractal_blend} |
| 4 | 2.12 | 0.19 | {salt_pepper, jpeg, gaussian_blur, fractal_blend} |
| 5 | 2.09 | 0.03 | {gaussian_noise, salt_pepper, jpeg, gaussian_blur, fractal_blend} |
| ≥6 | ≈2.08 | ≈0 | — |

**Robust model** — best single: salt_pepper (score 9.23):

| k | Score | Marginal gain | Best set |
|---|---|---|---|
| 1 | 9.23 | — | {salt_pepper} |
| **2** | **4.77** | **4.46** | {salt_pepper, gaussian_blur} |
| 3 | 4.67 | 0.10 | {salt_pepper, gaussian_blur, fractal_blend} |
| 4 | 4.62 | 0.06 | {salt_pepper, gaussian_blur, hue_shift, fractal_blend} |
| ≥5 | ≈4.58 | ≈0 | — |

**Elbow at k=2**: the first pair captures >85% of total ensemble gain on both models.
Running all 10 corruptions adds almost nothing beyond the best 5.

### Human-perception philosophy (Section H)

The right criterion for choosing corruptions: **the model boundary s* should fall in the
"still perceptually natural" zone** — i.e. LPIPS(x_orig, corrupt(x, s*)) ≪ LPIPS at s=1.

| Corruption | Median s* (std) | LPIPS at s* | LPIPS at s=1.0 | Verdict |
|---|---|---|---|---|
| gaussian_noise | 0.047 | **0.018** | 0.397 | Model fails before any visible noise |
| fractal_blend | 0.344 | **0.008** | 0.174 | Model fails while image is near-pristine |
| jpeg | 0.747 | **0.018** | 0.120 | Barely degrades structure even at s=0.75 |
| posterize | 0.786 | **0.031** | 0.118 | Gradual; label preserved until extreme quantisation |
| gaussian_blur | 0.169 | 0.075 | 0.434 | Already blurry at boundary — borderline |
| salt_pepper | 0.035 | 0.105 | 0.630 | 3.5% pixel flips already visible; good s* |
| hue_shift | 0.474 | 0.128 | 0.125 | 88% immune; color-only but model resilient |
| brightness | 0.652 | 0.303 | 0.569 | **Bad**: overexposed at boundary |
| contrast_low/fog | 0.857 | 0.265 | 0.579 | **Worst**: near-flat-grey when model fails |

Corruptions that intuitively seem "label-preserving" (brightness, contrast, fog) are actually
the worst: the model boundary falls deep in the degraded region.
Corruptions where the model is disproportionately sensitive compared to humans (gaussian_noise,
fractal_blend) are ideal — they fool the model before humans perceive any change.

### Known bugs

- **contrast_low ≡ fog**: both implement `x*(1-s) + 0.5*s` — mathematically identical.
  Spearman r = 1.000, identical L2/SSIM/LPIPS. Fix: contrast_low should be
  `mean(x) + (x - mean(x)) * (1 - s)` (true contrast reduction preserving mean luminance).
  **Must be fixed before any experiment that includes contrast_low.**

---

## Cross-Study Synthesis

### What Phase 0 established about EvolBA's bottlenecks

| Bottleneck | Evidence | Location |
|---|---|---|
| ξ scale mismatch (exploration vs directed step) | Study 1/2: ξ-shrink accounts for 75% of all queries | Phase 3 internals |
| Phase 3 stagnation on robust model | Study 4: mean IR = 0.08 regardless of init | Phase 3 |
| Attractor-class collapse in uniform init | Study 4: "bird" attractor causes 100% of hard failures | Phase 1 |
| init_l2 does not predict Phase 3 quality | Study 4: Pearson r = −0.021 to +0.45 | Phase 1/3 link |
| L2 is a misleading perceptual metric | Study 5: gaussian_blur L2-rank 2 but LPIPS-rank 6 | Evaluation |

### The Phase 1 picture

For the **standard model**: Phase 1 succeeds trivially (0% hard failures, any init works).
Optimising Phase 1 for lower init_l2 or better LPIPS does not improve Phase 3 outcome.
The payoff is naturalness of the final adversarial example, not convergence speed.

For the **robust model**: Phase 1 failure is a real problem (10% hard with uniform init).
Fractal init reduces this to 3% but doesn't fix Phase 3. The dominant issue is that
Phase 3 is essentially non-functional against the robust model (IR ≈ 0.08, flat curve).

---

## Recommended Next Steps (Study 6+)

### Priority 1 — Fix the ξ scale mismatch (Phase 3)
Decouple the exploration radius (offspring sampling: `ξ · √n`) from the directed-step size
(`ξ`). Options: separate `σ_explore` and `σ_step` parameters, or add an explicit
`1/√n` correction to the directed step. Expected: large reduction in ξ-shrink cost on robust.

### Priority 2 — End-to-end corruption study (Study 6)
Plug the best Phase-1 corruption sets (fractal_blend + gaussian_blur for standard;
salt_pepper + gaussian_blur for robust) into full EvolBA and measure:
- Does a closer/more natural Phase-1 boundary actually reduce Phase-3 query count?
- Does Study 4's finding (init_l2 not correlated with IR) still hold with corruption-based init?
Fix contrast_low formula before running.

### Priority 3 — Attractor diversity in Phase 1
Replace uniform random search with a diverse-attractor sampler: run multiple short random
searches and keep the one landing in the adversarial class farthest from the true class
(or closest in image space). Would reduce both hard failures and bird-attractor lock-in.

### Priority 4 — Phase 3 fundamentals for robust model
The robust model's near-zero IR (≈0.08) suggests the mean-shift direction is incorrect.
Candidates: larger λ for better gradient estimate, adaptive cμ, or exploration bonus for
offspring that cross the boundary. Consider sep-CMA-ES or VD-CMA for better per-pixel
adaptation in the constrained search space.

### Priority 5 — Stagnation detection and restart
Implement best_l2 plateau detector (< 0.1% improvement over 5 generations) to trigger
Phase-1 restart. ξ halvings are not a reliable signal (Study 4, Analysis G).

---

## File Map

```
STAGE_0/
├── study0_frequency_resolution.ipynb     # Study 3: band sensitivity + resolution
├── study1_query_cost_and_fitness.ipynb   # Studies 1&2: query budget + fitness curves
├── study4_failure_analysis.ipynb         # Study 4: failure taxonomy + attractor analysis
├── study5_corruption_init.ipynb          # Study 5: corruption-based Phase 1
├── PHASE_0_RESUME.md                     # ← this file
└── outputs/
    ├── study1_query_fitness/             # Study 1&2 plots + telemetry
    ├── study3_frequency/                 # Study 3 plots
    ├── study4_failure/                   # Study 4 plots + parquet
    └── study5_corruptions/
        ├── results.parquet               # 4000 trial results
        ├── boundaries.npy               # (4000, 3, 32, 32) boundary images
        ├── lpips_scores.npy             # (4000,) LPIPS scores (post-hoc)
        ├── degradation_curves.npy       # (10, 200, 20) LPIPS vs severity
        ├── G_metric_comparison.png      # L2 / SSIM / LPIPS side-by-side
        ├── E_pair_matrix_*.png          # Ensemble pair quality matrices
        ├── E_marginal_gain_*.png        # Marginal gain curves
        └── F_gallery_*.png              # Visual boundary galleries

make_study5.py                           # Source script for study5 notebook (always edit this,
                                         # then regenerate with: python make_study5.py)
```
