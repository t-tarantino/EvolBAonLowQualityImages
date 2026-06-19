#!/usr/bin/env python3
"""
make_study1.py
Generate STAGE_1/study1_hyperparameter_tuning.ipynb

Six conditions isolate each proposed improvement from evolba_baseline:
  baseline    paper defaults (xi_corr=F, bs=26, tau=3, lam=28, cmu=1.0)
  xi_fix      fix step-size scale mismatch only
  cmu_01      re-tune covariance learning rate only
  efficient   reduce wasted binary-search queries + backtracks only
  lam_small   smaller population → more generations in same budget
  all_fixes   all improvements combined
"""
import nbformat, os

NB_PATH    = 'STAGE_1/study1_hyperparameter_tuning.ipynb'
OUTPUT_DIR = 'STAGE_1/outputs/study1'

def code(src): return nbformat.v4.new_code_cell(src)
def md(src):   return nbformat.v4.new_markdown_cell(src)

C = []

# ── Cell 0 ── Title ───────────────────────────────────────────────────────────
C.append(md("""\
# Stage 1 — Study 1: Hyperparameter Tuning of the EvolBA Baseline

**Goal:** identify which algorithm parameters hurt the baseline most and quantify
the gain from fixing each one individually and all at once.

**Background (from Phase 0):**
- ξ-shrink accounts for ~75 % of all queries: one scalar ξ does two jobs
  that differ by a factor of √n ≈ 55 for CIFAR-10.
- `cmu_scale = 1.0` was tuned for VGG19/224×224 (n ≈ 150 k), not CIFAR-10 (n = 3 072).
- `BS_STEPS = 26` exceeds float32 precision — queries beyond ~23 change nothing.
- `TAU = 3` wastes up to 3 × BS_STEPS queries on backtracks mostly caused by the ξ mismatch.
- `λ = 28` (auto): all offspring always land adversarial, so the sign-flip for
  failed offspring never fires. Fewer offspring → more generations in the same budget.

**Six conditions** (change one knob at a time, then everything at once):

| Condition | xi_fix | cmu | bs_steps | tau | lam |
|---|---|---|---|---|---|
| `baseline`  | ✗ | 1.0 | 26 | 3 | 28 |
| `xi_fix`    | ✓ | 1.0 | 26 | 3 | 28 |
| `cmu_01`    | ✗ | 0.1 | 26 | 3 | 28 |
| `efficient` | ✗ | 1.0 | 15 | 1 | 28 |
| `lam_small` | ✗ | 1.0 | 26 | 3 | 14 |
| `all_fixes` | ✓ | 0.1 | 15 | 1 | 28 |

**Sample:** 200 CIFAR-10 images (20 per class), 2 models, 2 000 queries/run → 2 400 runs total (~5 h on GPU).

**How to read the run count:** `6 conditions × 200 images × 2 models = 2 400`.
Each of the 200 images is run once per condition per model — that is the only source of repetition.
"""))

# ── Cell 1 ── Imports + conditions ───────────────────────────────────────────
C.append(code("""\
import os, sys, time, warnings, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from tqdm.auto import tqdm

warnings.filterwarnings('ignore')
sys.path.insert(0, '..')
sys.path.insert(0, '.')

from evolba_tuned import evolba_tuned

OUTPUT_DIR = 'outputs/study1'
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_Q    = 2000
N_IMG    = 200  # 20 per class
SNAP_QS  = [250, 500, 750, 1000, 1500, 2000]
CIFAR_CLASSES = ['airplane','automobile','bird','cat','deer',
                 'dog','frog','horse','ship','truck']

CONDITIONS = {
    'baseline':  dict(xi_correction=False, bs_steps=26, tau=3, lam_override=None, cmu_scale=1.0),
    'xi_fix':    dict(xi_correction=True,  bs_steps=26, tau=3, lam_override=None, cmu_scale=1.0),
    'cmu_01':    dict(xi_correction=False, bs_steps=26, tau=3, lam_override=None, cmu_scale=0.1),
    'efficient': dict(xi_correction=False, bs_steps=15, tau=1, lam_override=None, cmu_scale=1.0),
    'lam_small': dict(xi_correction=False, bs_steps=26, tau=3, lam_override=14,   cmu_scale=1.0),
    'all_fixes': dict(xi_correction=True,  bs_steps=15, tau=1, lam_override=None, cmu_scale=0.1),
}
COND_ORDER  = list(CONDITIONS.keys())
COND_COLORS = {
    'baseline':  '#888888',
    'xi_fix':    '#2196F3',
    'cmu_01':    '#FF9800',
    'efficient': '#4CAF50',
    'lam_small': '#9C27B0',
    'all_fixes': '#F44336',
}
n_runs = len(CONDITIONS) * N_IMG * len(['standard','robust'])
print(f'{len(CONDITIONS)} conditions  ×  {N_IMG} images  ×  2 models  =  {n_runs} runs')
print(f'Max queries per run: {MAX_Q}  |  Snapshot query counts: {SNAP_QS}')
print(f'Estimated wall time: ~{n_runs * 7.3 / 3600:.1f} h  (7.3 s/run on GPU)')
"""))

# ── Cell 2 ── Load models ─────────────────────────────────────────────────────
C.append(code("""\
from robustbench.utils import load_model

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', device)

model_std = load_model('Standard', dataset='cifar10',
                       threat_model='Linf').to(device).eval()
model_rob = load_model('Wang2023Better_WRN-28-10', dataset='cifar10',
                       threat_model='Linf').to(device).eval()

def make_oracle(model):
    def oracle(x_chw):
        with torch.no_grad():
            t = torch.from_numpy(x_chw[None].astype(np.float32)).to(device)
            return int(model(t).argmax(1).item())
    return oracle

MODELS = {'standard': make_oracle(model_std), 'robust': make_oracle(model_rob)}
print('Models loaded.')
"""))

# ── Cell 3 ── Load images ─────────────────────────────────────────────────────
C.append(code("""\
import torchvision

ds = torchvision.datasets.CIFAR10(root='/tmp/cifar10', train=False, download=True)

per_class = N_IMG // 10  # 20 per class
images, labels = [], []
counts = [0] * 10
for img_pil, label in ds:
    if counts[label] >= per_class:
        continue
    x = np.array(img_pil, dtype=np.float32).transpose(2, 0, 1) / 255.0
    if MODELS['standard'](x) == label:
        images.append(x)
        labels.append(label)
        counts[label] += 1
    if sum(counts) == N_IMG:
        break

images = np.stack(images)
labels = np.array(labels)
print(f'Collected {len(images)} images ({per_class}/class).')
"""))

# ── Cell 4 ── Run all conditions (with cache) ─────────────────────────────────
C.append(code("""\
RESULTS_FILE = f'{OUTPUT_DIR}/results.parquet'
TRAJ_FILE    = f'{OUTPUT_DIR}/trajectories.pkl'

def l2_at_q(trajectory, q_thresh, init_l2):
    \"\"\"L2 of the last generation that finished before q_thresh queries.\"\"\"
    val = init_l2
    for q, l2 in trajectory:
        if q <= q_thresh:
            val = l2
        else:
            break
    return val

if os.path.exists(RESULTS_FILE) and os.path.exists(TRAJ_FILE):
    print('Results already cached — skipping run.')
else:
    rows, all_traj = [], {}
    total = len(CONDITIONS) * N_IMG * len(MODELS)
    t0 = time.time()

    with tqdm(total=total, desc='runs') as pbar:
        for cname, cparams in CONDITIONS.items():
            for mname, oracle in MODELS.items():
                for img_idx in range(N_IMG):
                    x_orig = images[img_idx]
                    y_true = int(labels[img_idx])
                    seed   = img_idx * 100 + list(CONDITIONS).index(cname) * 10

                    result = evolba_tuned(oracle, x_orig, y_true,
                                         max_queries=MAX_Q, seed=seed, **cparams)
                    traj = result['trajectory']
                    key  = (cname, mname, img_idx)
                    all_traj[key] = traj

                    row = dict(
                        condition  = cname,
                        model      = mname,
                        image_idx  = img_idx,
                        true_class = int(y_true),
                        success    = result['success'],
                        queries    = result['queries'],
                        init_l2    = result.get('init_l2', float('nan')),
                        best_l2    = result.get('best_l2', float('nan')),
                    )
                    il2 = row['init_l2']
                    for sq in SNAP_QS:
                        row[f'l2_at_{sq}'] = l2_at_q(traj, sq, il2)
                    row['improvement_ratio'] = (
                        (il2 - row['best_l2']) / il2
                        if il2 > 0 and not np.isnan(il2) else float('nan')
                    )
                    rows.append(row)
                    pbar.update(1)

    elapsed = time.time() - t0
    df = pd.DataFrame(rows)
    df.to_parquet(RESULTS_FILE, index=False)
    with open(TRAJ_FILE, 'wb') as f:
        pickle.dump(all_traj, f)
    print(f'Done in {elapsed:.0f}s  ({elapsed/total:.1f}s/run)')
"""))

# ── Cell 5 ── Load results + summary table ────────────────────────────────────
C.append(code("""\
df = pd.read_parquet(RESULTS_FILE)
with open(TRAJ_FILE, 'rb') as f:
    all_traj = pickle.load(f)

ok = df[df.success]
print(f'{len(df)} runs  |  {df.success.mean()*100:.1f}% success overall')
print()

summary = ok.groupby(['condition','model']).agg(
    median_init_l2 = ('init_l2',  'median'),
    median_best_l2 = ('best_l2',  'median'),
    median_IR      = ('improvement_ratio', 'median'),
    n              = ('best_l2',  'count'),
).round(3)
print('=== Median results per condition × model ===')
print(summary.to_string())
"""))

# ── Cell 6 ── Plot A: final L2 per condition ──────────────────────────────────
C.append(code("""\
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

for ax, mname in zip(axes, ['standard', 'robust']):
    sub = ok[ok.model == mname]
    data   = [sub[sub.condition == c]['best_l2'].dropna().values for c in COND_ORDER]
    medians = [np.median(d) if len(d) else np.nan for d in data]

    parts = ax.violinplot(
        [d if len(d) else [np.nan] for d in data],
        positions=range(len(COND_ORDER)),
        showmedians=True, showextrema=False,
    )
    for i, (pc, cname) in enumerate(zip(parts['bodies'], COND_ORDER)):
        pc.set_facecolor(COND_COLORS[cname])
        pc.set_alpha(0.6)
    parts['cmedians'].set_color('black')
    parts['cmedians'].set_linewidth(2)

    # Annotate medians
    for i, med in enumerate(medians):
        if not np.isnan(med):
            ax.text(i, med + 0.05, f'{med:.2f}', ha='center', va='bottom',
                    fontsize=8, fontweight='bold')

    ax.set_xticks(range(len(COND_ORDER)))
    ax.set_xticklabels(COND_ORDER, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Final L2  (lower = better)')
    ax.set_title(f'{mname} model')
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('A: Final L2 per condition  —  lower is better', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/A_final_l2.png', dpi=130, bbox_inches='tight')
plt.show()
print('Saved A_final_l2.png')
"""))

C.append(md("**Analysis A — Final L2 per condition.**\n\n*(fill in after run)*"))

# ── Cell 8 ── Plot B: L2 vs query budget curves ───────────────────────────────
C.append(code("""\
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, mname in zip(axes, ['standard', 'robust']):
    sub_df = ok[ok.model == mname]

    for cname in COND_ORDER:
        sub_c = sub_df[sub_df.condition == cname]
        curves = []
        for _, row in sub_c.iterrows():
            key  = (cname, mname, int(row['image_idx']))
            traj = all_traj.get(key, [])
            il2  = row['init_l2']
            curve = [l2_at_q(traj, sq, il2) for sq in SNAP_QS]
            curves.append(curve)

        if not curves:
            continue
        arr = np.array(curves)
        med = np.median(arr, axis=0)
        p25 = np.percentile(arr, 25, axis=0)
        p75 = np.percentile(arr, 75, axis=0)

        ax.plot(SNAP_QS, med, color=COND_COLORS[cname], lw=2, label=cname, marker='o', ms=4)
        ax.fill_between(SNAP_QS, p25, p75, color=COND_COLORS[cname], alpha=0.12)

    ax.set_xlabel('Oracle queries used')
    ax.set_ylabel('Median L2  (lower = better)')
    ax.set_title(f'{mname} model')
    ax.legend(fontsize=8, frameon=True)
    ax.grid(alpha=0.3)

plt.suptitle('B: Convergence curves — median L2 vs query budget  (shading = IQR)', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/B_convergence.png', dpi=130, bbox_inches='tight')
plt.show()
print('Saved B_convergence.png')
"""))

C.append(md("**Analysis B — Convergence curves.**\n\n*(fill in after run)*"))

# ── Cell 10 ── Plot C: improvement ratio ─────────────────────────────────────
C.append(code("""\
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, mname in zip(axes, ['standard', 'robust']):
    sub = ok[ok.model == mname]
    x_pos = np.arange(len(COND_ORDER))
    meds  = [sub[sub.condition==c]['improvement_ratio'].median() for c in COND_ORDER]
    bars  = ax.bar(x_pos, meds,
                   color=[COND_COLORS[c] for c in COND_ORDER], alpha=0.85)

    for bar, med in zip(bars, meds):
        if not np.isnan(med):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.005,
                    f'{med:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(COND_ORDER, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Median IR = (init_L2 − final_L2) / init_L2  (higher = better)')
    ax.set_ylim(0, 1)
    ax.set_title(f'{mname} model')
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('C: Improvement Ratio per condition  —  higher is better', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/C_improvement_ratio.png', dpi=130, bbox_inches='tight')
plt.show()
print('Saved C_improvement_ratio.png')
"""))

C.append(md("**Analysis C — Improvement Ratio.**\n\n*(fill in after run)*"))

# ── Cell 12 ── Plot D: query efficiency (L2 per 100 queries) ─────────────────
C.append(code("""\
# How much does each condition improve L2 per 100 queries?
# Efficiency = (l2_at_500 - l2_at_3000) / (3000 - 500) * 100
# Higher = more L2 reduction per 100 queries = more query-efficient.

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, mname in zip(axes, ['standard', 'robust']):
    sub = ok[ok.model == mname]
    effs = []
    for cname in COND_ORDER:
        sc = sub[sub.condition == cname]
        dl2 = sc['l2_at_500'] - sc['l2_at_3000']
        eff = (dl2 / 2500.0 * 100).median()
        effs.append(eff)

    bars = ax.bar(range(len(COND_ORDER)), effs,
                  color=[COND_COLORS[c] for c in COND_ORDER], alpha=0.85)
    for bar, e in zip(bars, effs):
        if not np.isnan(e):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.0001,
                    f'{e:.4f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(range(len(COND_ORDER)))
    ax.set_xticklabels(COND_ORDER, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('ΔL2 per 100 queries  (higher = more efficient)')
    ax.set_title(f'{mname} model')
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('D: Query efficiency — L2 reduction per 100 oracle queries', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/D_query_efficiency.png', dpi=130, bbox_inches='tight')
plt.show()
print('Saved D_query_efficiency.png')
"""))

C.append(md("**Analysis D — Query efficiency.**\n\n*(fill in after run)*"))

# ── Cell 14 ── Summary ────────────────────────────────────────────────────────
C.append(md("## Summary & Conclusions\n\n*(fill in after run)*"))

# ─────────────────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(NB_PATH), exist_ok=True)
nb = nbformat.v4.new_notebook()
nb.cells = C
with open(NB_PATH, 'w') as f:
    nbformat.write(nb, f)
print(f'Wrote {NB_PATH}  ({len(C)} cells)')
