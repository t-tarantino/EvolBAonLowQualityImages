#!/usr/bin/env python3
"""
make_study7_results.py
Generate STAGE_1/study7_results.ipynb — loads the cached results from
outputs/study7_full/ (produced by run_study7.py) and presents tables,
plots, and analysis.
"""
import nbformat

NB_PATH = 'study7_results.ipynb'

def code(src): return nbformat.v4.new_code_cell(src)
def md(src):   return nbformat.v4.new_markdown_cell(src)

C = []

C.append(md("""\
# Study 7 — Results: Does the tuning compound?

Loads cached results from `outputs/study7_full/` (800 runs: 2 conditions x
200 images x 2 models, 2000 queries each).

Studies 1-6 settled on, in isolation:
  - `xi_step_scale=0.5`  (Study 3, validated at `lam_override=14`)
  - `bs_adaptive=True`   (Study 5, ~5-7% saved on binary search queries)
  - `tau=0`              (Study 6, + best_l2 running-min fix, validated at `lam_override=14`)
  - `cmu_scale=1.0`       (Study 3: already == baseline's default)
  - `lam_override=None`  (deliberately NOT changed -- left at baseline's
    default `4+3*ln(n)=28` for CIFAR-10, per explicit instruction; lam is
    deferred to a future study)

Both arms run through `evolba_tuned()` (so both get the `best_l2`
running-minimum fix uniformly -- not a confound):

```
BASELINE: xi_step_scale=1.0, tau=3, bs_steps=26, bs_adaptive=False, cmu_scale=1.0, lam_override=None
TUNED:    xi_step_scale=0.5, tau=0, bs_adaptive=True, bs_cap=26,    cmu_scale=1.0, lam_override=None
```

**Question.** Do the individually-validated improvements stack into a clear
net win, or do they interact/cancel once `lam` is held at baseline's value?
"""))

C.append(code("""\
import pandas as pd, numpy as np, pickle
from IPython.display import Image, display

OUT = 'outputs/study7_full'
df = pd.read_parquet(f'{OUT}/results.parquet')
with open(f'{OUT}/trajectories.pkl', 'rb') as f:
    all_traj = pickle.load(f)

ok = df[df.success]
print(f'{len(df)} runs total, {len(ok)} successful ({df.success.mean()*100:.1f}%)')
ok.head()
"""))

C.append(md("## Summary table"))

C.append(code("""\
COND_ORDER  = ['baseline', 'tuned']
MODEL_NAMES = ['standard', 'robust']

summary = ok.groupby(['condition','model']).agg(
    n               = ('best_l2', 'count'),
    median_init     = ('init_l2', 'median'),
    median_best     = ('best_l2', 'median'),
    median_final    = ('final_l2', 'median'),
    median_IR       = ('improvement_ratio', 'median'),
    median_gen      = ('n_generations', 'median'),
    median_queries  = ('queries', 'median'),
    median_bsq      = ('bs_queries_actual', 'median'),
).round(4)
summary = summary.reindex(
    pd.MultiIndex.from_product([COND_ORDER, MODEL_NAMES], names=['condition','model'])
)
summary
"""))

C.append(md("""\
## Headline result: the gains do NOT compound

| | xi_step_scale | tau | bs_steps | median_gen | standard IR | robust IR |
|---|---|---|---|---|---|---|
| **baseline** | 1.0 | 3 | 26 (fixed) | 16-20.5 | 0.386 | 0.060 |
| **tuned** | 0.5 | 0 | adaptive (cap 26) | 33-34 | 0.380 | 0.047 |

`tuned` delivers on its *mechanical* promise -- roughly **double** the
generations in both models (smaller steps + no backtracking + adaptive
binary search all save queries). But this does not translate into a better
attack:

- **standard**: a wash. IR 0.380 vs 0.386 (~-1.5% relative).
- **robust**: `tuned` is actually *worse*. IR 0.047 vs 0.060 (~-22% relative).
"""))

C.append(md("## A: Final L2 distributions"))
C.append(code("display(Image(f'{OUT}/A_final_l2.png'))"))

C.append(md("## B: Convergence curves"))
C.append(code("display(Image(f'{OUT}/B_convergence.png'))"))

C.append(md("""\
For **standard**, `baseline` leads throughout Q=250-1500 and `tuned` only
catches up to parity by Q=2000.

For **robust**, `baseline` leads across the *entire* range, and `tuned`
never catches up (Q2000: baseline=7.00 vs tuned=7.26).
"""))

C.append(md("## C: Paired per-image comparison"))
C.append(code("display(Image(f'{OUT}/C_paired.png'))"))

C.append(code("""\
print('=== PAIRED COMPARISON (per image, tuned vs baseline) ===')
for mname in MODEL_NAMES:
    base = ok[(ok.condition=='baseline')&(ok.model==mname)].set_index('image_idx')['best_l2']
    tun  = ok[(ok.condition=='tuned')   &(ok.model==mname)].set_index('image_idx')['best_l2']
    common = base.index.intersection(tun.index)
    b, t_ = base.loc[common], tun.loc[common]
    win_rate = float((t_ < b).mean())
    rel_impr = float(np.median((b - t_) / b))
    print(f'{mname}: n={len(common)}  tuned-wins={win_rate*100:.1f}%  '
          f'median relative improvement={rel_impr*100:.1f}%')
"""))

C.append(md("""\
**standard**: win rate 50.0%, median relative improvement +0.1% -- points
scatter evenly around the diagonal, no systematic edge either way.

**robust**: win rate only 26.7%, median relative improvement -0.9% -- points
are tightly clustered along the diagonal with a consistent slight bias
toward `tuned` being *worse*. Not a few outliers: a small, systematic
disadvantage across nearly the whole test set.
"""))

C.append(md("## D: Generations, query allocation, improvement ratio"))
C.append(code("display(Image(f'{OUT}/D_summary.png'))"))

C.append(md("""\
## Why doesn't doubling the generations help?

Compare to Study 6 (same `xi_step_scale=0.5, tau=0`, but at
`lam_override=14`): there, `tau=0` got **57-59** generations and
IR **0.441 / 0.071**. Here at `lam=28`, the *same* xi_scale=0.5/tau=0 combo
gets roughly half the generations (**33-34**, as expected -- `lam` sets
offspring-per-generation) -- but the IR *also* drops (0.441->0.380,
0.071->0.047), not just proportionally.

Meanwhile `baseline`'s bigger steps (`xi_step_scale=1.0`) extract enough
progress per generation that even with **half** the generations of `tuned`,
it keeps pace (standard) or wins outright (robust).

**Conclusion**: `xi_step_scale=0.5` was the right choice *for the lam=14
regime*, where generations are abundant (57-59) and many small steps
compound well. At `lam=28`, generations are scarcer (16-34), and bigger
steps per generation seem to matter more. `lam` is not an independent knob
-- it determines which step-size regime is optimal, so the Study 3/5/6
recommendations (found under `lam_override=14`) do not transfer cleanly to
`lam=28` as-is.

## Next step

`lam x xi_step_scale` is the natural joint sweep: does `xi_step_scale=0.5`
recover its Study-3/6 advantage once paired with a smaller `lam`, or does
`lam=28` + `xi_step_scale=1.0` (closer to baseline) remain the better
operating point regardless? Deferred to a future study.
"""))

nb = nbformat.v4.new_notebook()
nb['cells'] = C
nb['metadata'] = {
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
}
with open(NB_PATH, 'w') as f:
    nbformat.write(nb, f)
print(f'Wrote {NB_PATH}')
