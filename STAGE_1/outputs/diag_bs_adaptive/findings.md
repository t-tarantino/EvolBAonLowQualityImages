# Diagnostic: does `binary_search_adaptive` (bs_cap=26) actually save queries?

Script: `STAGE_1/diag_bs_adaptive.py`. Collected real `(x_adv, x_orig, y_true)` pairs
from `evolba_tuned` trajectories at **Q=2000** (the FULL Study-5 budget), shared
HPARAMS = `xi_step_scale=0.5, tau=1, lam_override=14, cmu_scale=1.0`.

## 1. Mechanism is mechanically correct

Direct step-by-step trace of two pairs shows `||hi-lo||` halving *exactly* every
step (e.g. 5.94e-1 -> 2.97e-1 -> 1.49e-1 -> ... -> 2.14e-6 -> 2.14e-6), at which
point `mid == hi` (or `mid == lo`) becomes True in float32 — exactly the ULP
floor expected for pixel magnitudes ~0.2-1.0 (~1.8e-6 to 2.1e-6). `array_equal`
fires correctly and `binary_search_adaptive` breaks at that step. Float32 is
confirmed in use throughout the call chain (images, x0, m_shifted, cand, mid).

## 2. At Q=300 (mock) the mechanism barely fires — trajectories are too short

The Study-5 mock (Q=300, mostly early generations) showed ~0 median savings.
This was NOT a bug — `L = ||x_adv - x_orig||` simply hadn't shrunk enough yet.

## 3. At Q=2000, real (non-zero) savings appear

40 pairs sampled across generations 1-51, extended (40-step) bisection used to
find the TRUE saturation step `n_true` (independent of any cap):

| n_true | count | saved vs cap=26 |
|---|---|---|
| 21 | 11 | 5 |
| 24 | 7  | 2 |
| 25 | 3  | 1 |
| 26 | 4  | 0 |
| >=27 (didn't saturate by 40) | 15 | 0 |

- 21/40 (52.5%) of calls save 1-5 queries with `bs_cap=26`.
- median saved = 1, mean saved ~= 1.8 (~7% of the 26-step budget per call).

## 4. Savings are model-dependent (geometry-driven)

`n_true ~= log2(L) + ~19-21` (the slowest single pixel of 3072 dominates, since
`array_equal` requires ALL components to match).

- **standard**: L shrinks to ~0.6-2.7 by gen 30-50 -> n_true 21-25 -> real savings.
- **robust**: L stays ~5.3-11.5 even at gen 42 (matches Study 3's final L2~7-8)
  -> n_true mostly 24-40 -> little/no savings.

## Takeaway

Binary search dominates per-generation query cost (26 vs ~14 offspring with
`lam_override=14`), so even a ~5-7% BS reduction (concentrated in the standard
model's later generations) buys a few extra CMA-ES generations per run within
the same total budget — i.e. the benefit shows up as slightly better final L2,
not as the attack finishing early. Q=2000 (Study 5's FULL config) is the
right scope to quantify this; Q=300 (mock) is too short to be representative.
