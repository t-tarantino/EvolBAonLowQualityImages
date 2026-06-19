#!/usr/bin/env python3
"""
make_ec1_results.py
Generate STAGE_2/ec1_results.ipynb -- loads the cached results from
outputs/ec1_full/ (produced by run_ec1.py) and presents tables, plots,
and analysis.
"""
import nbformat

NB_PATH = 'ec1_results.ipynb'

def code(src): return nbformat.v4.new_code_cell(src)
def md(src):   return nbformat.v4.new_markdown_cell(src)

C = []

C.append(md("""\
# Study EC1 -- Results: CMA-ES variant comparison

Loads cached results from `outputs/ec1_full/` (1000 runs: 5 conditions x
100 images x 2 models, 2000 queries each).

All conditions share the Study 7 "tuned except tau=3" carrier config
(`xi_step_scale=0.5, bs_adaptive=True, cmu_scale=1.0, lam_override=None,
tau=3`), differing only in the covariance representation:

```
sep : evolba_vkd(vk_rank=0)  -- sep-CMA-ES (== evolba_tuned, validated bit-for-bit)
vd1 : evolba_vkd(vk_rank=1)  -- VD-CMA   (C = D(I + VV^T)D, V is n x 1)
vd2 : evolba_vkd(vk_rank=2)  -- VkD-CMA, k=2
vd3 : evolba_vkd(vk_rank=3)  -- VkD-CMA, k=3
o11 : evolba_one_plus_one()  -- (1+1)-CMA-ES, structurally different
      generation loop (1 query/gen, elitist, p_succ-based step adaptation)
```

Full CMA-ES and LM-CMA were considered and discarded -- see `CHANGES.md`.

**Question.** Do a handful of extra low-rank covariance directions (VkD,
k=1..3) capture exploitable structure that sep-CMA-ES's diagonal misses?
And how does the radically different (1+1)-CMA-ES compare?
"""))

C.append(code("""\
import pandas as pd, numpy as np, pickle
from IPython.display import Image, display

OUT = 'outputs/ec1_full'
df = pd.read_parquet(f'{OUT}/results.parquet')
with open(f'{OUT}/trajectories.pkl', 'rb') as f:
    all_traj = pickle.load(f)
with open(f'{OUT}/v_norms.pkl', 'rb') as f:
    v_norms_store = pickle.load(f)
with open(f'{OUT}/o11_diag.pkl', 'rb') as f:
    o11_diag = pickle.load(f)

ok = df[df.success]
print(f'{len(df)} runs total, {len(ok)} successful ({df.success.mean()*100:.1f}%)')
ok.head()
"""))

C.append(md("## Summary table"))

C.append(code("""\
COND_ORDER  = ['sep', 'vd1', 'vd2', 'vd3', 'o11']
MODEL_NAMES = ['standard', 'robust']
VK_RANK     = {'sep': 0, 'vd1': 1, 'vd2': 2, 'vd3': 3}

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
## Headline result: sep-CMA-ES (k=0) wins -- extra V directions don't pay off

| condition | standard median_best | robust median_best |
|---|---|---|
| **sep (k=0)** | **2.7016** | 6.9084 |
| vd1 (k=1) | 2.7985 | 6.9171 |
| vd2 (k=2) | 2.7916 | 6.9122 |
| vd3 (k=3) | 2.7454 | 6.9018 |
| o11 | 2.9008 | 6.9411 |

On the **standard** model, plain sep-CMA-ES is the best of all five
conditions. vd1/vd2/vd3 are all slightly worse, though vd3 (the largest
rank tried) partially recovers towards sep -- a monotonic trend
sep < vd3 < vd2 < vd1 in how close each gets to sep's performance.

On the **robust** model, all four CMA-ES-family conditions (sep/vd1/vd2/vd3)
are within noise of each other (6.90-6.92); the improvement ratios are tiny
(~0.04-0.05) so there's little signal for the extra V directions to exploit
either way.

This is the *opposite* of the trend seen in the N=10 mock (Q=300), where
vd1->vd2->vd3 ticked very slightly *below* sep on the standard model. At
the full query budget the ordering flips -- the mock's short horizon
(4-5 generations) wasn't representative.
"""))

C.append(md("## A: Final L2 distributions"))
C.append(code("display(Image(f'{OUT}/A_final_l2.png'))"))

C.append(md("## B: Convergence curves"))
C.append(code("display(Image(f'{OUT}/B_convergence.png'))"))

C.append(md("## Paired per-image comparison (vs sep)"))
C.append(code("""\
print('=== PAIRED COMPARISON (per image, best_l2 vs sep) ===')
for cname in ['vd1', 'vd2', 'vd3', 'o11']:
    for mname in MODEL_NAMES:
        base = ok[(ok.condition=='sep')  & (ok.model==mname)].set_index('image_idx')['best_l2']
        oth  = ok[(ok.condition==cname) & (ok.model==mname)].set_index('image_idx')['best_l2']
        common = base.index.intersection(oth.index)
        b, o_ = base.loc[common], oth.loc[common]
        win_rate = float((o_ < b).mean())
        rel_impr = float(np.median((b - o_) / b))
        print(f'{cname:4s} vs sep | {mname:8s}: n={len(common)}  '
              f'{cname}-wins={win_rate*100:5.1f}%  median rel impr={rel_impr*100:+.2f}%')
"""))

C.append(md("""\
- **vd1/vd2/vd3 vs sep**: win rates 40-44% on both models, median relative
  *dis*improvement of 0.0-0.3% -- a small, consistent edge to sep, not a
  few outliers.
- **o11 vs sep, standard**: only 36% win rate, median relative
  disimprovement -6.6% -- a substantial and consistent gap.
- **o11 vs sep, robust**: 58.9% win rate with median relative *improvement*
  +1.4% -- o11 wins on *more* individual images here, yet its median
  best_l2 (6.9411) is slightly *worse* than sep's (6.9084). The two
  rankings disagree because the distribution is skewed: o11 nudges ahead on
  many easy images but loses by a larger margin on a smaller number of hard
  ones, which dominate the median-of-best_l2 comparison less but the
  win-rate comparison more.
"""))

C.append(md("## D: Generations, query allocation, improvement ratio"))
C.append(code("display(Image(f'{OUT}/D_summary.png'))"))

C.append(md("## F: Effect of VkD rank k"))
C.append(code("display(Image(f'{OUT}/F_k_sweep.png'))"))

C.append(md("## G: VkD dominant-direction norms over generations"))
C.append(code("display(Image(f'{OUT}/G_v_norms.png'))"))

C.append(code("""\
print('=== Median ||V[:,i]|| at final generation, per condition/model ===')
for cname in ['vd1', 'vd2', 'vd3']:
    for mname in MODEL_NAMES:
        finals = [v_norms_store[(cname, mname, i)][-1]
                  for i in range(100)
                  if (cname, mname, i) in v_norms_store and v_norms_store[(cname, mname, i)]]
        if finals:
            arr = np.array(finals)
            meds = np.median(arr, axis=0)
            print(f'{cname} {mname}: n={len(arr)}  median final norms = '
                  + ', '.join(f'{m:.3f}' for m in meds))
"""))

C.append(md("""\
The V column norms are remarkably **consistent across k and across models**:
column 0 settles near **~1.05**, column 1 near **~0.80**, column 2 near
**~0.64**, regardless of whether k=1, 2, or 3, and regardless of standard
vs robust model. Adding more columns doesn't change the earlier columns'
magnitudes, and the ordering (col0 > col1 > col2) is the same every time.

This looks like a **stable property of the self-normalised update itself**
rather than evidence of attack-relevant structure in any particular image's
loss landscape -- i.e. V grows to a "typical noise" magnitude set by
`mu`/`mueff` and the eigenvalue-ratio statistics of `Y = zs_ranked * w_eff`
(which are themselves roughly image-independent), not because it has locked
onto a genuinely exploitable direction. This is consistent with the paired
comparison: V grows to a non-trivial size in every run, but doesn't
translate into a systematic win over sep's pure diagonal.
"""))

C.append(md("## H: (1+1)-CMA-ES sigma and p_succ adaptation"))
C.append(code("display(Image(f'{OUT}/H_o11_diag.png'))"))

C.append(md("""\
## Conclusions

1. **sep-CMA-ES (diagonal covariance, k=0) remains the best choice** of the
   five conditions tested, on both models, at the full Q=2000 budget.
2. **VkD's extra rank-k directions (k=1,2,3) consistently cost a small
   amount** (0.0-0.3% median) rather than helping, on the standard model,
   and are noise-level neutral on the robust model. The V matrices grow to
   a consistent ~1.05/0.80/0.64 magnitude pattern in every condition/model,
   suggesting this is an artifact of the (self-normalised) update dynamics
   rather than image-specific structure being captured.
3. **(1+1)-CMA-ES is clearly worse on the standard model** (-6.6% median,
   only 36% win rate vs sep) despite ~16x more generations (478 vs 30).
   On the robust model it's roughly competitive by win-rate (58.9%) but not
   by median best_l2, due to a skewed distribution.
4. **Net recommendation**: keep sep-CMA-ES (`evolba_vkd(vk_rank=0)`, i.e.
   the existing `evolba_tuned`) as the production configuration. Neither
   VkD nor (1+1) justify their added complexity for this attack setting.

## Caveats / possible follow-ups

- The carrier config (`xi_step_scale=0.5, tau=3, ...`) was tuned *for*
  sep-CMA-ES in Stage 1; it's possible VkD or (1+1) would benefit from their
  own hyperparameter tuning (e.g. a different `cv` schedule, or `xi_step_scale`
  for o11). This study answers "does VkD/( 1+1) help as a drop-in
  replacement under sep's tuned settings", not "what is VkD/(1+1)'s ceiling
  under its own best settings".
- `o11`'s sigma adaptation was shown (in validation) to be nearly inert at
  n=3072 (`d_damp ~ 1537`), so its fixed `sigma0 = init_l2/n` choice is
  doing most of the work; a per-image or per-model sigma0 might change the
  picture.
"""))

nb = nbformat.v4.new_notebook()
nb['cells'] = C
nb['metadata'] = {
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
}
with open(NB_PATH, 'w') as f:
    nbformat.write(nb, f)
print(f'Wrote {NB_PATH}')
