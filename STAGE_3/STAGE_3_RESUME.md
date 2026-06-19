# Stage 3 — Mean-Shift-Direction Bias Diagnostic

> **Purpose.** Record the result of the `mean_shift_direction` (Eq. 9-10) bias diagnostic
> so future work doesn't have to re-derive or re-run it.

---

## Background / Hypothesis

`evolba_baseline.mean_shift_direction()` (Eq. 9-10) negates the sampled direction `z` of
every **non-adversarial** offspring before summing it into `v`. The hypothesis going into
this diagnostic:

> Negating non-adversarial offspring's `z` systematically pushes `v` (the CMA-ES mean-shift
> direction) **away from `x_orig`**, and this is the structural reason the ξ-shrink
> while-loop (Alg. 1 lines 12-14) fires so often (Stage 0, Study 1: ~75% of every query on
> the standard model, ~75% on robust too).

Three candidate update rules were compared, run **end-to-end / on-policy** (each variant
drives its own trajectory, not retroactively evaluated on another variant's states), with
**paired seeds** (`seed = image_idx * 100`, identical across variants for a given image so
initial boundary point + per-generation noise draws line up at each generation index):

- **`current`** — `evolba_baseline.mean_shift_direction`, unchanged (validated baseline).
- **`eq10`** — literal paper Eq. 10: for the non-adversarial group, weights are paired with
  z-ranks in **reversed** order vs. `current` (`w_{l+1}` ↔ worst non-adv, `w_mu` ↔ best non-adv).
- **`drop`** — "don't go there": non-adversarial offspring contribute **nothing** to `v`
  (only adversarial offspring, renormalised weights; `v=0` if zero offspring are adversarial
  that generation).

**Primary metric:** `v_dot_u = v · (x_orig - m)/‖x_orig - m‖` — cosine-similarity-like
projection of `v` (a unit vector) onto the direction toward `x_orig`.
- `v_dot_u < 0` → `v` points **away** from `x_orig` (the hypothesised bias).
- `v_dot_u ≈ 0` → `v` is tangential (the paper's stated ideal — "move along the boundary").
- `v_dot_u > 0` → `v` points **toward** `x_orig`.

**Secondary metrics:** `n_shrink` (ξ-halvings in the shrink loop), `n_backtrack`,
`frac_adv`, `xi`, `dist_to_orig`.

**Setup:** CIFAR-10, RobustBench `Standard` (non-robust) model, N=20 images
(2 per class, balanced, correctly classified), `λ=28`, `μ=λ`, no fractal init / no jump
operator (Alg. 1's plain loop). Code: [`diag_mean_shift_bias.py`](diag_mean_shift_bias.py).

---

## Results

### Summary tables

**Q=500** (`outputs/diag_mean_shift_full/`, 379 generation-records):

| variant | n_gens | mean v·u | median v·u | P(v·u<0) | mean n_shrink | mean n_backtrack |
|---|---|---|---|---|---|---|
| current | 126 | 0.0157 | 0.0149 | 0.230 | 5.10 | 0.69 |
| eq10    | 126 | 0.0174 | 0.0161 | 0.183 | 5.39 | 0.70 |
| drop    | 127 | 0.0163 | 0.0150 | 0.228 | 5.22 | 0.67 |

**Q=2000** (`outputs/diag_mean_shift_full_q2000/`, 1013 generation-records):

| variant | n_gens | mean v·u | median v·u | P(v·u<0) | mean n_shrink | mean n_backtrack |
|---|---|---|---|---|---|---|
| current | 353 | 0.0120 | 0.0109 | 0.272 | 44.56 | 0.538 |
| eq10    | 325 | 0.0142 | 0.0125 | 0.234 | 54.01 | 0.545 |
| drop    | 335 | 0.0127 | 0.0118 | 0.269 | 50.44 | 0.540 |

Histograms (`v_dot_u_hist.png` in each output dir): all three variants' `v·u`
distributions are tightly concentrated around 0, slightly positive, and visually
indistinguishable from each other.

### n_shrink vs. generation index `t` (Q=2000, pooled across images/variants)

`n_shrink` does **not** grow smoothly with `t`. It's flat and small for the first few
generations, then hits sporadic huge spikes, then settles into a noisy moderate range:

| gen (t) | mean n_shrink | mean ξ | mean v·u |
|---|---|---|---|
| 1 | 0.85 | 4.20 | 0.023 |
| 2 | 1.73 | 1.77 | 0.017 |
| 3 | 2.50 | 1.26 | 0.017 |
| 4 | 2.58 | 0.94 | 0.013 |
| 5 | 162.9 | 0.93 | 0.020 |
| **6** | **305.4** | 0.39 | 0.012 |
| 7 | 94.4 | 0.49 | 0.010 |
| **8** | **277.5** | 0.20 | 0.015 |
| 9-25 | noisy, 2-40 | decreasing, noisy | small, mixed sign |

A single generation with `n_shrink≈300` consumes ~15% of the entire Q=2000 budget on the
shrink loop alone.

### Per-image generation counts (Q=2000) — where do variants diverge?

| image_idx | current | eq10 | drop |
|---|---|---|---|
| 0 | 29 | 13 | 16 |
| 1 | 27 | 27 | 27 |
| 2 | 8 | 8 | 8 |
| 3 | 30 | 30 | 30 |
| 4 | 5 | 5 | 5 |
| 5 | 6 | 6 | 6 |
| 6 | 27 | 6 | 25 |
| 7 | 24 | 6 | 21 |
| 8 | 8 | 23 | 8 |
| 9 | 5 | 5 | 5 |
| 10 | 28 | 28 | 28 |
| 11 | 7 | 7 | 7 |
| 12 | 27 | 27 | 27 |
| 13 | 6 | 6 | 6 |
| 14 | 8 | 8 | 8 |
| 15 | 6 | 6 | 6 |
| 16 | 31 | 31 | 31 |
| 17 | 27 | 28 | 25 |
| 18 | 29 | 29 | 29 |
| 19 | 15 | 29 | 14 |

**15/20 images give byte-identical generation counts across all three variants.** This is
consistent with the "`l ≈ μ`" degeneracy: for most images, nearly all `λ` offspring stay
adversarial throughout the run, so `current`/`eq10`/`drop` reduce to the **same formula**
every generation (when `l=μ`, all three are the plain weighted sum of all-offspring `z`'s
with `+` signs).

For the other 5/20 images, the variants diverge sharply (e.g. img 0: 29/13/16 gens,
img 8: 8/23/8, img 19: 15/29/14) — but this is **chaotic path-dependence**, not a
systematic directional difference: `v·u` stays small (~0.01-0.02) for all three either way.
A tiny difference in `v` shifts whether a trajectory happens to land in one of the
catastrophic `n_shrink≈300` generations, which then consumes a large fraction of that
trajectory's remaining budget.

---

## Conclusion — was the hypothesis right?

**No.** `v_dot_u` is consistently **small and positive** (mean ≈ +0.012 to +0.017) across
both query budgets and all three update-rule variants, with only ~18-27% of generations
showing `v_dot_u < 0`. `v` is, on average, close to **tangential to the boundary** — the
paper's stated ideal — not systematically biased away from `x_orig`. Changing the
non-adversarial-offspring handling (`current` → `eq10`'s reversed pairing → `drop`'s
"contribute nothing") does **not** meaningfully change `v_dot_u`, nor does it
systematically reduce `n_shrink` (44.6 / 54.0 / 50.4 — all in the same range).

**The "go opposite vs. don't go there" question for Eq. 9-10 is effectively settled**: for
the standard CIFAR-10 model, it doesn't matter much, because for the large majority of
images `l≈μ` throughout the search (the non-adversarial group is empty or tiny), making
the three formulas algebraically identical.

## What this *does* point to

The real cost driver — confirmed at larger scale here — is the same one Stage 0 (Study 1/2)
already identified: **ξ (Eq. 4: `dist_to_orig/√t`) is a global, slowly-decreasing schedule
that periodically becomes wildly oversized relative to the local boundary scale near `m`**,
forcing dozens-to-hundreds of halvings in a single generation. `mean_n_shrink` grew ~9×
(5.2 → 49.7 avg) for a 4× increase in query budget — i.e. the problem gets *worse*, not
better, the longer the search runs, and it's essentially independent of which
mean-shift-direction formula drives `v`.

---

## File Map

```
STAGE_3/
├── diag_mean_shift_bias.py            # diagnostic script (python diag_mean_shift_bias.py [--mock])
├── STAGE_3_RESUME.md                  # ← this file
└── outputs/
    ├── diag_mean_shift_mock/          # N=3, Q=200 smoke test
    ├── diag_mean_shift_full/          # N=20, Q=500
    │   ├── results.parquet            # 379 generation-records
    │   ├── summary.csv
    │   └── v_dot_u_hist.png
    └── diag_mean_shift_full_q2000/    # N=20, Q=2000
        ├── results.parquet            # 1013 generation-records
        ├── summary.csv
        └── v_dot_u_hist.png
```
