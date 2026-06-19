# Stage 3 — Subspace Experiment Results

## What was tested

Two experiments:
1. **Small sweep** (`exp_subspace_random_vs_dct.py`): DCT vs random basis, k∈{30,60,120,240,480}, N=20, Q=2000 — no L2 history saved.
2. **Full run** (`exp_subspace_perf_visual.py`): same k-sweep, N=150, Q=2000 — with L2 history, convergence curves, and visual progression snapshots for 5 images at k=480.

Both use the same subspace port of Algorithm 1: uniform_random_init + binary_search in pixel space (unchanged), then sep-CMA-ES in k-dim theta-space with theta_orig as binary-search target.

---

## Main results (N=150, Q=2000)

| k | basis | mean IR | median IR | mean best L2 | mean residual norm | mean n_gens |
|---|---|---|---|---|---|---|
| 30  | dct    | 0.157 | 0.149 | 4.054 | 4.039 | 23.5 |
| 30  | random | 0.002 | 0.003 | 4.809 | 4.797 | 21.7 |
| 60  | dct    | 0.180 | 0.173 | 3.897 | 3.907 | 21.9 |
| 60  | random | 0.006 | 0.008 | 4.792 | 4.771 | 20.1 |
| 120 | dct    | 0.192 | 0.192 | 3.883 | 3.781 | 19.9 |
| 120 | random | 0.014 | 0.017 | 4.755 | 4.729 | 20.2 |
| 240 | dct    | 0.212 | 0.208 | 3.726 | 3.629 | 18.6 |
| 240 | random | 0.032 | 0.036 | 4.665 | 4.631 | 18.8 |
| 480 | dct    | 0.234 | 0.229 | 3.600 | 3.417 | 16.9 |
| 480 | random | 0.058 | 0.072 | 4.527 | 4.431 | 15.2 |

**DCT dominates random at every k.** Both improve monotonically with k. Random barely moves at any k — it provides almost no useful search direction.

---

## Comparison with pixel-space sep-CMA-ES

| Method | N | mean IR | mean best L2 |
|---|---|---|---|
| **Pixel-space sep-CMA-ES** (Stage 3 diagnostic) | 20 | **0.370** | **3.252** |
| DCT subspace k=480 | 150 | 0.234 | 3.600 |
| DCT subspace k=240 | 150 | 0.212 | 3.726 |
| Random subspace k=480 | 150 | 0.058 | 4.527 |

**Pixel-space beats every subspace configuration at every tested query budget** (100, 200, 400, 800, 1600). The gap widens after ~200 queries.

### Why subspace loses — the residual floor

Mean DCT k=480 residual norm = **3.417**.
Pixel-space mean best L2 = **3.252**.

The pixel-space algorithm achieves distances below the DCT k=480 theoretical floor — the adversarial directions it exploits partially live **outside** the DCT basis. The subspace constraint is not just an approximation — it actively blocks the relevant search directions.

Even with k=480 directions (15.6% of n=3072), DCT does not span the adversarial boundary well enough. The residual component is adversarially important.

---

## Convergence behaviour

- **Front-loaded**: almost all improvement happens in the first 300–600 queries for both DCT and pixel-space. The curve flattens sharply after that.
- **Visual progression** (k=480, 5 images): image 30 (frog) shows clear visual convergence by q≈550. Images 0 and 60 show degenerate behaviour — algorithm burns nearly the entire budget in 1–2 very expensive generations stuck in the xi-shrink loop, with no visible progression between snapshots. Confirmed by n_shrink counts (often >100/gen for DCT k=480).
- **Stagnation cause**: same xi-scale mismatch identified in Stage 0 (offspring spread ≈ xi·√k; directed step ≈ xi) persists in subspace, just scaled by √k instead of √n. Stage 1 addressed this for pixel-space with xi_step_scale, but not for the subspace experiments.

---

## Key conclusions

1. **Structured basis (DCT) >> random at every k.** Random subspace is a poor baseline for any practical use.
2. **Subspace < pixel-space.** Working in k-dim DCT space hurts performance despite reducing the mismatch from √3072≈55 to √480≈22. The residual floor is the binding constraint.
3. **The bottleneck for pixel-space is not the basis — it is the initialization.** The boundary point from uniform_random_init starts at L2≈4.8; corruption-based Phase 1 can reach L2≈1.9–2.5. This is the next experiment.
4. **Both bases plateau early.** Q=2000 is largely wasted after q≈500. Any real improvement must come from either a better starting point (Phase 1) or more efficient Phase 3 movement (Stage 1's xi_step_scale, tau=0 already help).

---

## Files

```
STAGE_3/
├── exp_subspace_random_vs_dct.py           # N=20 sweep (no L2 history)
├── exp_subspace_perf_visual.py             # N=150 full run (L2 history + snapshots)
└── outputs/
    ├── exp_subspace_full_q2000/            # N=20 results
    │   ├── results.parquet
    │   └── summary.csv
    ├── exp_subspace_perf_q2000/            # N=150 results
    │   ├── results.parquet                 # 1500 rows, includes l2_history / queries_history
    │   ├── summary.csv
    │   ├── convergence_by_k.png
    │   ├── ir_vs_k.png
    │   └── visual_progression_img{0,30,60,90,120}.png
    └── comparison_pixel_vs_subspace.png    # cross-budget comparison (q=100→1600)
```
