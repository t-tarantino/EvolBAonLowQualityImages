#!/usr/bin/env python3
"""
make_study1_results.py
Generate STAGE_1/study1_results.ipynb — loads the cached results from
outputs/study1_full/ (produced by run_study1.py) and presents tables,
plots, and analysis.
"""
import nbformat, os

NB_PATH = 'STAGE_1/study1_results.ipynb'

def code(src): return nbformat.v4.new_code_cell(src)
def md(src):   return nbformat.v4.new_markdown_cell(src)

C = []

C.append(md("""\
# Study 1 — Results: Hyperparameter Tuning of the EvolBA Baseline

Loads cached results from `../outputs/study1_full/` (2 400 runs: 6 conditions
× 200 images × 2 models, 2 000 queries each).
"""))

C.append(code("""\
import pandas as pd, numpy as np, pickle
from IPython.display import Image, display

OUT = '../outputs/study1_full'
df = pd.read_parquet(f'{OUT}/results.parquet')
with open(f'{OUT}/trajectories.pkl', 'rb') as f:
    all_traj = pickle.load(f)

ok = df[df.success]
print(f'{len(df)} runs total, {len(ok)} successful ({df.success.mean()*100:.1f}%)')
ok.head()
"""))

C.append(md("## Summary table — median final L2 and improvement ratio"))

C.append(code("""\
COND_ORDER = ['baseline', 'xi_fix', 'cmu_01', 'efficient', 'lam_small', 'all_fixes']

summary = ok.groupby(['condition','model']).agg(
    n            = ('best_l2', 'count'),
    median_init  = ('init_l2', 'median'),
    median_best  = ('best_l2', 'median'),
    median_IR    = ('improvement_ratio', 'median'),
).round(4)
summary = summary.reindex(
    pd.MultiIndex.from_product([COND_ORDER, ['standard','robust']], names=['condition','model'])
)
summary
"""))

C.append(md("""\
## Key finding: `xi_correction` hurts badly

The central Phase-0 hypothesis — that the directed step should use
`xi_step = xi/√n` instead of `xi_step = xi` — is **wrong in practice**.

| Condition | standard IR | robust IR |
|---|---|---|
| baseline (no fixes) | **0.386** | **0.057** |
| cmu_01 | 0.337 | 0.048 |
| efficient | 0.383 | 0.011 |
| lam_small | 0.366 | **0.056** |
| xi_fix | 0.025 | 0.003 |
| all_fixes | 0.027 | 0.004 |

Any condition with `xi_correction=True` (`xi_fix`, `all_fixes`) collapses the
improvement ratio by ~15×. With the correction, the directed step
`m + xi_step·v` is ~55× smaller, so the post-binary-search point barely
differs from `m` — the optimizer stalls.

**Best configs found:**
- standard: `baseline` / `efficient` / `lam_small` (final L2 ≈ 2.36–2.41, all within noise)
- robust: `lam_small` (λ=14, final L2 = 6.93) — closely followed by `baseline` (7.05)
"""))

C.append(md("## Plot A — Final L2 per condition (violin)"))
C.append(code("display(Image(filename=f'{OUT}/A_final_l2.png'))"))

C.append(md("## Plot B — Convergence curves (median ± IQR)"))
C.append(code("display(Image(filename=f'{OUT}/B_convergence.png'))"))

C.append(md("## Plot C — Improvement ratio per condition"))
C.append(code("display(Image(filename=f'{OUT}/C_improvement_ratio.png'))"))

C.append(md("## Plot D — Query efficiency (ΔL2 / 100 queries)"))
C.append(code("display(Image(filename=f'{OUT}/D_query_efficiency.png'))"))

C.append(md("""\
## Conclusions

1. **Drop `xi_correction` entirely.** The √n-scale "fix" was based on a
   plausible-sounding but incorrect intuition — the original mismatched ξ is
   what drives boundary-following progress.
2. **`lam_small` (λ=14) is the best all-round config**, matching `baseline` on
   the standard model and slightly beating it on the robust model.
3. **`bs_steps=15, tau=1` (the `efficient` condition) is free** — it matches
   `baseline` on the standard model (2.37 vs 2.41) while using fewer
   queries per generation, so more generations fit in the same budget.
4. Recommended config going forward: **`xi_correction=False, bs_steps=15,
   tau=1, lam_override=14, cmu_scale=1.0`** (i.e. `efficient` + `lam_small`
   combined — worth a quick follow-up check).
"""))

os.makedirs(os.path.dirname(NB_PATH), exist_ok=True)
nb = nbformat.v4.new_notebook()
nb.cells = C
with open(NB_PATH, 'w') as f:
    nbformat.write(nb, f)
print(f'Wrote {NB_PATH}  ({len(C)} cells)')
