#!/usr/bin/env python3
"""
make_study3_results.py
Generate STAGE_1/study3_results.ipynb — loads the cached results from
outputs/study3_full/ (produced by run_study3.py) and presents tables,
plots, and analysis.
"""
import nbformat, os

NB_PATH = 'STAGE_1/study3_results.ipynb'

def code(src): return nbformat.v4.new_code_cell(src)
def md(src):   return nbformat.v4.new_markdown_cell(src)

C = []

C.append(md("""\
# Study 3 — Results: Directed-Step Scale (`xi_step_scale`) Sweep

Loads cached results from `../outputs/study3_full/` (1 400 runs: 7 values of
`xi_step_scale` × 100 images × 2 models, 2 000 queries each).

Fixed hyperparameters (Study 1's individually-best, combined for the first time):
`bs_steps=15, tau=1, lam_override=14, cmu_scale=1.0`.

`xi_step = xi * xi_step_scale`. Grid: `0.0180 (=1/sqrt(3072), old "broken"
xi_correction=True)`, `0.0625`, `0.125`, `0.25`, `0.5`, `1.0 (old
xi_correction=False)`, `2.0 (untested, beyond old baseline)`.
"""))

C.append(code("""\
import pandas as pd, numpy as np, pickle
from IPython.display import Image, display

OUT = '../outputs/study3_full'
df = pd.read_parquet(f'{OUT}/results.parquet')
with open(f'{OUT}/trajectories.pkl', 'rb') as f:
    all_traj = pickle.load(f)

ok = df[df.success]
print(f'{len(df)} runs total, {len(ok)} successful ({df.success.mean()*100:.1f}%)')
ok.head()
"""))

C.append(md("## Summary table — median final L2, IR, and per-generation overhead"))

C.append(code("""\
COND_ORDER = ['0.0180', '0.0625', '0.125', '0.25', '0.5', '1.0', '2.0']

summary = ok.groupby(['condition','model']).agg(
    n            = ('best_l2', 'count'),
    median_init  = ('init_l2', 'median'),
    median_best  = ('best_l2', 'median'),
    median_IR    = ('improvement_ratio', 'median'),
    median_gen   = ('n_generations', 'median'),
    shrink_per_gen = ('shrink_per_gen', 'median'),
    bt_per_gen     = ('backtrack_per_gen', 'median'),
).round(4)
summary = summary.reindex(
    pd.MultiIndex.from_product([COND_ORDER, ['standard','robust']], names=['condition','model'])
)
summary
"""))

C.append(md("""\
## Headline result

| xi_step_scale | standard IR | robust IR |
|---|---|---|
| 0.0180 (old "broken") | 0.031 | 0.005 |
| 0.0625 | 0.114 | 0.018 |
| 0.125 | 0.174 | 0.039 |
| 0.25 | 0.282 | 0.043 |
| **0.5** | 0.366 | **0.066** |
| 1.0 (old "best") | **0.406** | 0.026 |
| 2.0 | 0.395 | **-0.058** (worse than init!) |

**Standard model**: smooth, monotonic curve from 0.018 to 1.0, slight dip at
2.0. `1.0` is confirmed near-optimal — Study 1's conclusion ("drop
xi_correction") was directionally right, and there's no hidden sweet spot in
between for this model.

**Robust model**: non-monotonic, and the surprise of this study —
**`xi_step_scale=0.5` beats `1.0` by 2.5×** (IR 0.066 vs 0.026), and even
beats Study 1's best robust config (`lam_small`, IR=0.056). `2.0` is actively
*harmful*: median IR is negative, meaning the attack typically ends up
**farther** from the original image than where Phase 1 left it.
"""))

C.append(md("## Why does `scale=2.0` hurt the robust model?"))

C.append(code("""\
for scale in ['0.5', '2.0']:
    sub = ok[(ok.condition == scale) & (ok.model == 'robust')]
    frac_worse = (sub.improvement_ratio <= 0).mean()
    print(f'scale={scale}: IR<=0 (ended worse than Phase-1 start) in '
          f'{frac_worse*100:.1f}% of robust images  '
          f'(mean IR={sub.improvement_ratio.mean():.3f}, '
          f'median IR={sub.improvement_ratio.median():.3f})')
"""))

C.append(md("""\
At `scale=2.0` the directed step starts *larger* than the exploration radius
itself. For the robust model — whose Phase-3 boundary is much flatter/harder
to follow (Study 1: baseline robust IR was already only 0.057) — an
oversized first guess regularly overshoots into a region from which the
shrink/backtrack loop (capped at `tau=1` backtrack) can't recover within
budget, leaving `m` farther from `x_orig` than its Phase-1 starting point in
>56% of images.
"""))

C.append(md("## Plot A — Final L2 vs xi_step_scale (violin)"))
C.append(code("display(Image(filename=f'{OUT}/A_final_l2.png'))"))

C.append(md("## Plot B — Convergence curves"))
C.append(code("display(Image(filename=f'{OUT}/B_convergence.png'))"))

C.append(md("## Plot C — Improvement ratio vs xi_step_scale"))
C.append(code("display(Image(filename=f'{OUT}/C_improvement_ratio.png'))"))

C.append(md("## Plot D — Generations completed in budget"))
C.append(code("display(Image(filename=f'{OUT}/D_generations.png'))"))

C.append(md("## Plot E — Shrink/backtrack overhead per generation"))
C.append(code("display(Image(filename=f'{OUT}/E_overhead.png'))"))

C.append(md("""\
## Conclusions

1. **Standard model: keep `xi_step_scale=1.0`.** The relationship is smooth
   and monotonic across the whole grid — `1.0` sits at (or very near) the
   peak (IR=0.406). No hidden intermediate sweet spot.
2. **Robust model: switch to `xi_step_scale=0.5`.** This more than doubles
   the improvement ratio vs the previous best (0.066 vs 0.026 at `1.0`,
   vs 0.056 for Study 1's best robust config `lam_small`). `1.0` is
   apparently *too aggressive* a starting step for the robust model's
   flatter boundary.
3. **`xi_step_scale=2.0` should be avoided entirely**, especially for the
   robust model, where it makes the median run end up *worse* than its
   Phase-1 starting point.
4. **Trade-off if a single config is needed for both models**:
   `xi_step_scale=0.5` costs the standard model ~10% relative IR (0.366 vs
   0.406) but gives the robust model a 2.5× gain (0.066 vs 0.026) — likely
   the better overall choice if Stage 1's goal includes the robust model.
5. Recommended config going forward: **`xi_step_scale=0.5, bs_steps=15,
   tau=1, lam_override=14, cmu_scale=1.0`** — or a model-conditional choice
   (`1.0` for standard, `0.5` for robust) if per-model tuning is acceptable.
"""))

os.makedirs(os.path.dirname(NB_PATH), exist_ok=True)
nb = nbformat.v4.new_notebook()
nb.cells = C
with open(NB_PATH, 'w') as f:
    nbformat.write(nb, f)
print(f'Wrote {NB_PATH}  ({len(C)} cells)')
