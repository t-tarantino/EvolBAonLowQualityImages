# Code Structure

This file describes what's actually used by the experiments — verified by tracing every
`import` across STAGE_0 through STAGE_3 and FINAL_RUN, not by design intent. An earlier version of this
document described a modular OO framework (`attacks/core`, `attacks/phases`,
`attacks/algorithms`) as the live pipeline; that framework was never adopted by any real
experiment and has been removed. See "A note on `attacks/`" below.

## Top-level layout

```
EvolBA/
├── config.py            single source of truth for hyperparameters
│                         (SEED, BS_STEPS, PENALTY, MAX_Q_FMNIST, MAX_Q_CIFAR10, ...)
├── evolba_baseline.py    faithful implementation of the EvolBA paper (Tajima & Ono, 2024,
│                         arXiv:2407.02248) — init / binary search / mean-shift / jump /
│                         CMA-ES loop, each as its own function. This is the algorithm
│                         every stage actually builds on.
│
├── attacks/              small shared math-utility module (not a framework — see below)
│   └── utils/
│       ├── metrics.py     ssim, mse, linf, l2, all_metrics  (CHW arrays)
│       ├── boundary.py    approaching_direction, half_space_reflect,
│       │                  boundary_normal_estimate, init_covariance_biased
│       └── subspace.py    dct_basis, grid_superpixel_basis, combined_basis,
│                          random_basis, corruption_basis — used by STAGE_3 and FINAL_RUN
│
├── models/                robustbench checkpoints: cifar10/Linf/{Standard,
│                          Wang2023Better_WRN-28-10}.pt
├── data/                  CIFAR-10 / FashionMNIST raw datasets
│
├── STAGE_0/ … STAGE_3/    the actual experiments — see per-stage map below
├── FINAL_RUN/             N=500 validation run of Part 2's best subspace config, plus the
│                         5-image visual snapshot grids used in the report's Part 2
│                         (`models/` symlinked like every STAGE folder)
├── report/                the report itself (`report.tex` / `report.pdf`), `figures/`,
│                         and `make_pixel_space_grid.py` (generates the Part 1 pixel-space
│                         snapshot grids the same way FINAL_RUN generates Part 2's)
└── STAGE -1/              earliest throwaway POC notebooks, predates STAGE_0, not maintained
```

---

## Report Part ↔ folder mapping

The report (`report/report.tex`) tells the project's story in **two** Parts. Each Part now
spans several folders (this used to be a clean one-Part-per-`STAGE_X` mapping in an earlier
draft; the report was later restructured to group diagnosis + both pixel-space fix attempts
into one Part, see below):

| Report Part | What it covers | Folder(s) |
|---|---|---|
| Part 1 — Diagnosis and the Limits of Pixel-Space Fixes | instrument the baseline, find what breaks (§4.1–4.2); hyperparameter tuning (§4.3); richer covariance structure, VkD-CMA / (1+1)-CMA-ES vs. plain Sep-CMA-ES (§4.4) | `STAGE_0/` (diagnosis) + `STAGE_1/` (tuning) + `STAGE_2/` (covariance) |
| Part 2 — Inductive Bias in the Search Space | corruption-direction subspace construction, Q1–Q3 experiments, validation at $N{=}500$ | `STAGE_3/` (subspace construction + Q1–Q3) + `FINAL_RUN/` (the $N{=}500$ validation and its snapshot figures) |

The pixel-space snapshot figures added to Part 1 §4.3 (`figures/pixel_space_snapshot_grid_*.png`)
are generated directly by `report/make_pixel_space_grid.py`, not by any `STAGE_X` script — no
STAGE_0/STAGE_1 script ever saved actual image checkpoints (only scalar L2 trajectories), so
this one lives next to the report instead.

---

## How imports actually work (read this before moving anything)

Every `STAGE_X/*.py` script is run with its **working directory set to its own `STAGE_X/`
folder**, not the repo root — relative paths like `outputs/study1_full/` and
`models/cifar10/...` resolve against that. To still reach the root-level `config.py` /
`evolba_baseline.py`, each script manually prepends the repo root to `sys.path` near the top:

```python
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
```

Several STAGE_2/STAGE_3 scripts add a *second* path for STAGE_1, because they import
STAGE_1's `phase1_zoo.py` directly across stage boundaries:

```python
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'STAGE_1'))
```

`STAGE_X/models/` is a symlink to the top-level `models/` (previously four duplicate
~1.3GB copies, one per stage) — this still works because robustbench's
`load_model(model_dir='./models')` default resolves the symlink transparently.

---

## Per-stage dependency map

### `STAGE_0/` — first formal studies (report Part 1, §4.1–4.2)
| Script/notebook | Depends on | Report section |
|---|---|---|
| `study1_query_cost_and_fitness.ipynb` | `config.py` | §4.1 Query Cost Breakdown |
| `study4_failure_analysis.ipynb` | `config.py` | §4.2 Failure Modes and the Phase 3 Plateau |
| `study0_frequency_resolution.ipynb` (internally titled "Study 3") | `config.py` | background for §3.1's resolution argument; not directly tabulated |
| `make_study5.py` → generates `study5_corruption_init.ipynb` | `evolba_baseline.py` (`binary_search`, `blend_frequencies`, `generate_fractal_image`, `BS_STEPS`) | closest existing source for §5.1's corruption-comparison methodology (L2/SSIM/LPIPS across 10 corruption types); names and count don't line up exactly with the report's Table~5, so treat it as closely related rather than the literal one-to-one source |

### `STAGE_1/` — hyperparameter/ablation studies (report Part 1, §4.3, plus one Part 2 input)
The `run_studyN.py` (+ `--mock` flag) → `make_studyN_results.py` → `studyN_results.ipynb`
pattern (see workflow memory). Two stage-local modules wrap the root baseline:

| Module | Depends on | Role |
|---|---|---|
| `evolba_tuned.py` | `config.py`, `evolba_baseline.py` | the "tuned" variant most studies actually run |
| `phase1_zoo.py` | `evolba_baseline.py` | zoo of alternative Phase-1 init strategies |

| Script | Imports | Report section |
|---|---|---|
| `run_study1.py`, `run_study3.py` | `evolba_tuned.py` | §4.3, $\xi_\text{step\_scale}$ sweep |
| `run_study6.py` | `evolba_tuned.py` | §4.3, backtracking budget $\tau$ |
| `run_study7.py`, `run_study8.py` | `evolba_tuned.py` | §4.3, compounding/best-config check |
| `run_study2.py` | `evolba_tuned.py` + `evolba_baseline.py` + `phase1_zoo.py` | §5.1's "Validation against uniform-random initialisation (Study 2)" — exact source: 8 corruption strategies via `multi_init` vs. a uniform-random baseline |
| `run_study4.py` | `evolba_tuned.py` + `evolba_baseline.py` + `phase1_zoo.py` | exploratory, not cited directly in the current report |
| `run_study5.py` | `evolba_tuned.py` + `evolba_baseline.py` | adaptive binary search, folded into §4.3's prose |
| `diag_bs_adaptive.py` | standalone diagnostic | same |

### `STAGE_2/` — evolutionary-computation variants (report Part 1, §4.4)
| Module | Depends on | Role |
|---|---|---|
| `evolba_ec.py` | `config.py`, `evolba_baseline.py` | `evolba_vkd`, `evolba_one_plus_one` |
| `utils_stage2.py` | self-contained | no root dependency |

| Script | Imports | Report section |
|---|---|---|
| `run_ec1.py` → `make_ec1_results.py` → `ec1_results.ipynb` | `evolba_ec.py` | §4.4's Study EC1 (the only STAGE_2 result actually in the report: sep vs. VkD($k$=1,2,3) vs. (1+1)-CMA-ES) |
| `validate_ec.py` | `evolba_ec.py` + `evolba_tuned.py` | sanity-check script, not cited |
| `exp1_stagnation_ipop.ipynb` … `exp6_query_profiling.ipynb`, `stage2_summary.ipynb` | various | earlier exploratory notebooks superseded by `run_ec1.py`; kept for history, not what §4.4 is drawn from |

### `STAGE_3/` — subspace construction and Q1–Q3 (report Part 2, §5.2–5.6)
All of: `exp_corruption_phase1.py`, `exp_corruption_subspace.py`, `exp_fixed_subspace_ipop.py`,
`exp_fixed_subspace_ipop2.py`, `exp_full_cmaes_subspace.py`, `exp_subspace_perf_visual.py`,
`exp_subspace_random_vs_dct.py`, `diag_mean_shift_bias.py` import:
- `evolba_baseline.py` directly
- `STAGE_1/phase1_zoo.py` (cross-stage import, see sys.path note above)
- `attacks.utils.subspace` (`corruption_basis` / `dct_basis` / `random_basis`)

| Script | Report section |
|---|---|
| `exp_corruption_subspace.py` | §5.3 Q1, §5.4 Q2 (the "buggy full Phase 1" experiments the report flags explicitly) |
| `exp_fixed_subspace_ipop2.py` | §5.5 Q3 (fixed Phase 1, $k{=}14$, population study; despite the filename this is the population experiment, not the IPOP one — IPOP was tried and dropped from the report) |
| `exp_fixed_subspace_ipop.py`, `exp_full_cmaes_subspace.py`, `exp_subspace_perf_visual.py`, `exp_subspace_random_vs_dct.py`, `diag_mean_shift_bias.py` | earlier/exploratory, not cited directly |

### `FINAL_RUN/` — validation at scale (report Part 2, §5.7)
| Script | Imports | Role |
|---|---|---|
| `run_final.py` | `evolba_baseline.py` helpers + `attacks.utils.subspace` | reruns Part 2's best config (fixed $\lambda{=}56$, $k{\leq}14$ subspace, fixed Phase 1) on $N{=}500$ images, saves `snapshots.npz` (actual image arrays at each checkpoint, not just scalars) |
| `make_final_plots.py` | reads `outputs/final_*/{results.parquet,snapshots.npz}` | produces the report's `final_run_snapshot_grid_{standard,robust}.png` and `final_run_l2_vs_query.png` (Figures in §5.7) |

### `report/` — the report and its figure-generation scripts
| File | Role |
|---|---|
| `report.tex` / `report.pdf` | the report itself |
| `figures/` | all `\includegraphics` targets, copied or generated from the STAGE/FINAL_RUN outputs above |
| `make_pixel_space_grid.py` | standalone (does **not** read from any `STAGE_X/outputs/`): reruns Part 1's best-validated pixel-space config on the same 5 images as `FINAL_RUN`'s grids, since no STAGE_0/STAGE_1 script ever saved actual image checkpoints. Imports `evolba_baseline.py` directly; needs its own `models/` symlink (`report/models -> ../models`), same convention as every STAGE folder |

---

## A note on `attacks/`

An earlier iteration of this project built a full object-oriented attack framework under
`attacks/core` (`Oracle`, `Recorder`, `AttackState`, `Phase`, `Experiment`),
`attacks/phases` (`GaussianInit`, `CorruptionFamilyInit`, `BinarySearchBoundary`, `PyCMAES`,
`SepCMAES`, `OnePlusOneCMAES`), and `attacks/algorithms` (an `EvolBA` class wiring it all
together). **It was never adopted by any experiment** — every real run in STAGE_0 through
STAGE_3 goes through the procedural `evolba_baseline.py` pipeline above instead. The only
part of `attacks/` that real experiments touch is `attacks/utils/subspace.py` (plus
`metrics.py`/`boundary.py`, which it doesn't even depend on but which were kept as related
math helpers). The unused core/phases/algorithms code has been deleted; what remains in
`attacks/` is exactly what's listed in the top-level layout above.

---

## Output convention

`run_studyN.py` / `exp_*.py` scripts write to `outputs/<name>_{mock,full}/` (relative to the
stage's own directory): a checkpointed `results.parquet` (one row per trial/image,
columns vary per study), `trajectories.pkl`, a timestamped `run.log`, and PNG plots. There
is no shared `Experiment` class producing a fixed schema — each script defines its own
result columns for what that study measures.
