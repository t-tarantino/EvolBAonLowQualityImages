# FINAL_RUN — the best configuration found in STAGE_3, run at scale

This folder reproduces the single best-performing configuration identified across all of
STAGE_0–STAGE_3 (see report `report/report_v3.tex`, Part 4 / `STAGE_3/exp_fixed_subspace_ipop2.py`,
arm `E_lam56`), and runs it at a larger scale (500 images instead of 200) with everything needed
to replot/re-analyse later saved to disk — no rerun required for anything covered below.

## Why this configuration

`STAGE_3/exp_fixed_subspace_ipop2.py` compared three arms: fixed λ=56, IPOP starting at λ=28,
and IPOP starting at λ=56. Checking the actual results (`STAGE_3/outputs/exp_fixed_subspace_ipop2_q1500_n200/results.parquet`)
against the report's claims:

| model | arm | mean best_l2 |
|---|---|---|
| standard | **E_lam56** | **1.647** (best) |
| standard | E_IPOP_28 | 1.671 |
| standard | E_IPOP_56 | 1.705 |
| robust | E_IPOP_56 | 4.197 (best) |
| robust | **E_lam56** | **4.203** |
| robust | E_IPOP_28 | 4.262 |

Plain fixed λ=56 wins clearly on the standard model and is statistically indistinguishable
from IPOP_56 on the robust model (0.14% apart, well within noise at N=200) — and it is what
the report's own appendix labels as the "best validated config." FINAL_RUN therefore uses
**plain fixed λ=56, no IPOP** — the simpler of the two, and the one actually best overall.

## The algorithm, in full

### Setup
- Two CIFAR-10 WideResNet-28-10 targets: `Standard` (no adversarial training) and
  `Wang2023Better_WRN-28-10` (TRADES-robust), loaded via RobustBench.
- Images are sampled from the CIFAR-10 test set, balanced per class, restricted to images
  **jointly correctly classified by both models** (so the comparison between models isn't
  contaminated by images one model already gets wrong).
- All randomness is seeded per-image as `seed_base = image_idx * 1000` (Phase 1), with
  `seed_base + 1` for Phase 3 — deterministic and reproducible.

### Phase 1 — fixed corruption initialisation (≈141 queries)
For each image, three corruption types are binary-searched to the decision boundary, in order:
**jpeg compression**, **Gaussian blur**, **fractal high-frequency blend** (random-seeded). Each
binary search costs ~20 queries (severity search) + ~26 (boundary refinement). The corruption
with the **lowest L2 boundary distance** is kept as the Phase 1 starting point `x_b`.

A further **8 corruption directions are added at zero oracle cost** (`DIRECTION_ZOO` —
brightness, contrast, inversion, hue-shift, posterize, sharpen, saturation, gamma), each
evaluated once at a fixed hardcoded severity (no search): the direction vector is simply
`d = corrupt(x_orig) − x_orig`. These don't contribute to picking `x_b`, only to spanning the
search subspace below. Three DCT frequency-band vectors (low/mid/high) are appended the same way.

This gives up to **14 direction vectors total**, frozen once per image before Phase 3 starts.

### Subspace construction (k ≤ 14)
The (up to) 14 direction vectors are stacked and QR-orthogonalised into an orthonormal basis
`B ∈ ℝ^(k×n)` (linearly dependent directions are dropped, hence `k ≤ 14`). This guarantees the
search-space norm `‖θ‖₂` equals the true pixel-space perturbation norm `‖δ‖₂` — necessary for
the L2 objective, the ξ step schedule, and binary search to behave correctly in θ-space exactly
as they do in pixel space.

### Phase 3 — sep-CMA-ES in the k-dim subspace, λ=56 fixed (remaining budget)
Standard EvolBA generation loop (Tajima & Ono, 2024, Eq. 1–10 / Algorithm 1), run in θ-space
instead of pixel space:

1. **Sample** λ=56 offspring: `θ_cand = θ_m + ξ·D·z`, `z ~ N(0, I_k)`.
2. **Evaluate** each candidate: project to pixel space (`x = clip(x_b + B^T θ, 0, 1)`), query the
   oracle, compute L2 to `x_orig`. Objective: `f = L2` if adversarial, `L2 + 1000` otherwise
   (Eq. 1–3) — guarantees every adversarial candidate ranks above every non-adversarial one.
3. **Mean-shift direction `v`** (Eq. 9–10): rank all λ offspring by fitness; weight by the usual
   log-decreasing CMA-ES weights, but **negate the direction of non-adversarial offspring** —
   "go this way, and also, don't go that other way." `v` is renormalised to a unit vector.
4. **Diagonal covariance update** (rank-μ term only — EvolBA drops the rank-one/evolution-path
   term since it replaces CSA with a deterministic step schedule, see below).
5. **Step size** `ξ = ‖θ_m − θ_orig‖ / √t` (Eq. 4) — deterministic, distance-to-original-driven,
   not adapted from path correlation (CSA's stationarity assumption doesn't hold here, since the
   boundary itself moves every generation as the perturbation shrinks).
6. **Move and pull back onto the boundary**: `θ_m + ξv` is halved (`ξ /= 2`) until adversarial,
   then a 26-step binary search pulls it back toward `θ_orig` along the line connecting them.
7. **Backtracking** (≤3 retries): if the pulled-back point ended up *farther* from `θ_orig` than
   before, halve `ξ` again and retry.

Loop continues until the query budget is exhausted.

### Query budget
Total Q = **2,000** per (image, model) — phase 1 spends ≈141, Phase 3 gets the remaining
≈1,859. (STAGE_3 used Q=1,500; this run uses 2,000, matching what was requested for the
macro-run.)

### Image sample
**500 images**, both models, 50 images per class, jointly correctly classified by both —
2.5× the N=200 used in STAGE_3.

## What gets recorded (so nothing needs rerunning)

Per (image, model) row in `results.parquet`:
- `phase1_l2`, `phase1_queries`, `winning_corruption`, `k_total` (basis size actually used)
- `phase1_ssim` / `phase1_mse` / `phase1_linf` (perceptual metrics at the Phase 1 boundary)
- `best_l2` (running minimum over the trajectory), `final_l2` (L2 at the last generation)
- `final_ssim` / `final_mse` / `final_linf` / `final_label` (metrics + predicted class at the
  final point)
- `IR_phase3 = (phase1_l2 − best_l2) / phase1_l2`
- `n_gens`, `queries_phase3`
- query-cost breakdown: `q_offspring_eval`, `q_xi_shrink`, `q_theta_bs` (binary-search pull +
  backtracking share this counter, since they call the same routine)
- `runtime_sec` (wall-clock time for that image/model)
- `l2_history`, `queries_history` — the **full per-generation trajectory**, so L2-vs-query and
  L2-vs-generation plots can be regenerated/restyled without rerunning anything.

For **10 evenly-spaced images** (selected up front, both models), 7 checkpoints are captured —
Original | Phase 1 boundary | Phase 3 @0%/25%/50%/75%/100% of Phase-3 budget — each saved as a
raw image array plus its L2, SSIM, and predicted label, in `snapshots.npz` +
`snapshots_meta.csv`.

All plotting is done by `make_final_plots.py`, which reads only the saved parquet/npz files —
changing a plot's style, labels, or binning never requires rerunning the attack.
