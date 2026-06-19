#!/usr/bin/env python3
"""
make_study2_results.py
Generate STAGE_1/study2_results.ipynb — loads the cached results from
outputs/study2_full/ (produced by run_study2.py) and presents tables,
plots, and analysis.
"""
import nbformat, os

NB_PATH = 'STAGE_1/study2_results.ipynb'

def code(src): return nbformat.v4.new_code_cell(src)
def md(src):   return nbformat.v4.new_markdown_cell(src)

C = []

C.append(md("""\
# Study 2 — Results: Multi-Init Phase 1

Loads cached results from `../outputs/study2_full/` (800 runs: 2 conditions
× 200 images × 2 models, 2 000 total queries each — Phase-1 probe queries are
charged against the same budget for `multi_init`).

Both conditions use the `all_fixes` hyperparameters from Study 1
(`xi_correction=True, bs_steps=15, tau=1, cmu_scale=0.1`) — **note Study 1
later found `xi_correction=True` cripples Phase 3**, so the gains below come
almost entirely from Phase 1, with Phase 3 doing very little in either
condition. A follow-up (Study 3) re-runs `multi_init` with the corrected
hyperparameters.
"""))

C.append(code("""\
import pandas as pd, numpy as np, pickle
from IPython.display import Image, display

OUT = '../outputs/study2_full'
df = pd.read_parquet(f'{OUT}/results.parquet')
with open(f'{OUT}/trajectories.pkl', 'rb') as f:
    all_traj = pickle.load(f)
with open(f'{OUT}/init_breakdown.pkl', 'rb') as f:
    init_breakdown = pickle.load(f)

ok = df[df.success]
print(f'{len(df)} runs total, {len(ok)} successful ({df.success.mean()*100:.1f}%)')
ok.head()
"""))

C.append(md("## Summary table — init L2 vs final L2"))

C.append(code("""\
summary = ok.groupby(['condition','model']).agg(
    n            = ('best_l2', 'count'),
    median_init  = ('init_l2', 'median'),
    median_best  = ('best_l2', 'median'),
    median_IR    = ('improvement_ratio', 'median'),
).round(4)
summary
"""))

C.append(md("""\
## Headline result

| Model | baseline init→final | multi_init init→final | reduction |
|---|---|---|---|
| standard | 3.80 → 3.65 | **2.21 → 2.19** | **−40%** |
| robust | 7.51 → 7.50 | **4.97 → 4.95** | **−34%** |

Multi-init Phase 1 cuts final L2 by 34–40% — almost entirely from a better
*starting point*, since Phase 3 (`all_fixes`, IR≈0.01) makes negligible
further progress in either condition.
"""))

C.append(md("## Which Phase-1 strategy wins?"))

C.append(code("""\
mi = ok[ok.condition == 'multi_init']
for mname in ['standard', 'robust']:
    sub = mi[mi.model == mname]
    counts = sub['init_winner'].value_counts()
    print(f'{mname}:')
    for name, cnt in counts.items():
        print(f'  {name:<15s}: {cnt:3d} / {len(sub)}  ({cnt/len(sub)*100:.1f}%)')
    print()
"""))

C.append(md("## Median Phase-1 boundary L2 per strategy (lower = better init)"))

C.append(code("""\
init_names = ['blur','brightness','contrast','inversion','hue_shift',
               'posterize','fractal_random','low_freq_rand']
init_cols  = [f'init_{k}_l2' for k in init_names]

for mname in ['standard', 'robust']:
    sub  = mi[mi.model == mname]
    meds = sub[init_cols].median().sort_values()
    print(f'{mname}:')
    for col, val in meds.items():
        print(f'  {col.replace("init_","").replace("_l2",""):<15s}: {val:.4f}')
    print()
"""))

C.append(md("## Plot A — Phase-1 init L2 (does multi-init find a closer boundary point?)"))
C.append(code("display(Image(filename=f'{OUT}/A_init_l2.png'))"))

C.append(md("## Plot B — Final L2 (does the better start survive into Phase 3?)"))
C.append(code("display(Image(filename=f'{OUT}/B_final_l2.png'))"))

C.append(md("## Plot C — Convergence curves"))
C.append(code("display(Image(filename=f'{OUT}/C_convergence.png'))"))

C.append(md("## Plot D — Which strategy wins per image"))
C.append(code("display(Image(filename=f'{OUT}/D_init_winner.png'))"))

C.append(md("## Plot E — Median init L2 per strategy"))
C.append(code("display(Image(filename=f'{OUT}/E_init_l2_per_strategy.png'))"))

C.append(md("""\
## Conclusions

1. **Multi-init Phase 1 is a clear win**: −40% L2 (standard), −34% L2
   (robust), even while *charging* the ~150-200 probe queries against the
   same total budget as the baseline.
2. **`blur` is the single best individual strategy** (wins 53% of standard
   images, 70.5% of robust images) — Gaussian blur destroys the
   high-frequency texture both models rely on, landing very close to the
   boundary cheaply.
3. **`fractal_random` is the strong #2 for the standard model** (42% of
   wins) — consistent with the paper's frequency-blend motivation.
4. **`brightness`, `contrast`, `inversion` are consistently the worst**
   (median init L2 8–24) — large global shifts are far from the boundary.
5. Because both conditions used the *broken* `all_fixes` hyperparameters
   (Study 1 finding: `xi_correction=True` collapses Phase 3's IR to ~0.01),
   **the full benefit of multi-init is not yet realized**. Re-running
   `multi_init` with Study 1's actual best config (`xi_correction=False,
   bs_steps=15, tau=1, lam_override=14`) is the natural next step — Phase 3
   could plausibly take the multi-init starting point (L2≈2.21 standard) and
   apply ~39% further reduction (→ L2≈1.3), instead of the ~1% it currently
   manages.
"""))

os.makedirs(os.path.dirname(NB_PATH), exist_ok=True)
nb = nbformat.v4.new_notebook()
nb.cells = C
with open(NB_PATH, 'w') as f:
    nbformat.write(nb, f)
print(f'Wrote {NB_PATH}  ({len(C)} cells)')
