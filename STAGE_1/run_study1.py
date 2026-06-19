#!/usr/bin/env python3
"""
run_study1.py — standalone runner for Study 1: hyperparameter tuning.

Usage:
    python run_study1.py              # full run  (N=200, Q=2000)
    python run_study1.py --mock       # quick check (N=10, Q=300, 2 conds, 1 model)
"""
import os, sys, time, warnings, pickle, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from evolba_tuned import evolba_tuned

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--mock', action='store_true',
                    help='quick sanity check (N=10, Q=300)')
args = parser.parse_args()

MOCK      = args.mock
MAX_Q     = 300   if MOCK else 2000
N_IMG     = 10    if MOCK else 200    # images per run  (per_class = N_IMG // 10)
SNAP_QS   = [100, 200, 300] if MOCK else [250, 500, 750, 1000, 1500, 2000]
TAG       = 'MOCK' if MOCK else 'FULL'

OUTPUT_DIR  = f'outputs/study1_{TAG.lower()}'
RESULTS_FILE = f'{OUTPUT_DIR}/results.parquet'
TRAJ_FILE    = f'{OUTPUT_DIR}/trajectories.pkl'
LOG_FILE     = f'{OUTPUT_DIR}/run.log'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Conditions ────────────────────────────────────────────────────────────────
ALL_CONDITIONS = {
    'baseline':  dict(xi_correction=False, bs_steps=26, tau=3, lam_override=None, cmu_scale=1.0),
    'xi_fix':    dict(xi_correction=True,  bs_steps=26, tau=3, lam_override=None, cmu_scale=1.0),
    'cmu_01':    dict(xi_correction=False, bs_steps=26, tau=3, lam_override=None, cmu_scale=0.1),
    'efficient': dict(xi_correction=False, bs_steps=15, tau=1, lam_override=None, cmu_scale=1.0),
    'lam_small': dict(xi_correction=False, bs_steps=26, tau=3, lam_override=14,   cmu_scale=1.0),
    'all_fixes': dict(xi_correction=True,  bs_steps=15, tau=1, lam_override=None, cmu_scale=0.1),
}
COND_COLORS = {
    'baseline':'#888888','xi_fix':'#2196F3','cmu_01':'#FF9800',
    'efficient':'#4CAF50','lam_small':'#9C27B0','all_fixes':'#F44336',
}

CONDITIONS  = {k: ALL_CONDITIONS[k] for k in (['baseline','all_fixes'] if MOCK else ALL_CONDITIONS)}
MODEL_NAMES = ['standard'] if MOCK else ['standard', 'robust']

# ── Logging helper ─────────────────────────────────────────────────────────────
log_f = open(LOG_FILE, 'w', buffering=1)
def log(msg):
    ts = time.strftime('%H:%M:%S')
    line = f'[{ts}] {msg}'
    print(line, flush=True)
    log_f.write(line + '\n')

# ── Models ────────────────────────────────────────────────────────────────────
from robustbench.utils import load_model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
log(f'Device: {device}')

model_map = {
    'standard': ('Standard', 'Linf'),
    'robust':   ('Wang2023Better_WRN-28-10', 'Linf'),
}
MODELS = {}
for mname in MODEL_NAMES:
    arch, threat = model_map[mname]
    m = load_model(arch, dataset='cifar10', threat_model=threat).to(device).eval()
    def _make_oracle(model=m):
        def oracle(x_chw):
            with torch.no_grad():
                t = torch.from_numpy(x_chw[None].astype(np.float32)).to(device)
                return int(model(t).argmax(1).item())
        return oracle
    MODELS[mname] = _make_oracle()
    log(f'Loaded model: {mname}')

# ── Images ────────────────────────────────────────────────────────────────────
import torchvision
ds = torchvision.datasets.CIFAR10(root='/tmp/cifar10', train=False, download=True)
per_class = N_IMG // 10
images, labels = [], []
counts = [0] * 10
for img_pil, label in ds:
    if counts[label] >= per_class:
        continue
    x = np.array(img_pil, dtype=np.float32).transpose(2,0,1) / 255.0
    if MODELS['standard'](x) == label:
        images.append(x); labels.append(label); counts[label] += 1
    if sum(counts) == N_IMG:
        break
images = np.stack(images)
labels = np.array(labels)
log(f'Images: {len(images)} ({per_class}/class)  |  label dist: {counts}')

# ── Summary ───────────────────────────────────────────────────────────────────
COND_ORDER  = list(CONDITIONS.keys())
total_runs  = len(CONDITIONS) * N_IMG * len(MODEL_NAMES)
log(f'=== {TAG} RUN ===')
log(f'{len(CONDITIONS)} conditions × {N_IMG} images × {len(MODEL_NAMES)} model(s) = {total_runs} runs')
log(f'Max queries per run: {MAX_Q}  |  snapshots at: {SNAP_QS}')
log(f'Output dir: {OUTPUT_DIR}')
log(f'Results: {RESULTS_FILE}')

# ── Utility ───────────────────────────────────────────────────────────────────
def l2_at_q(trajectory, q_thresh, init_l2):
    val = init_l2
    for q, l2 in trajectory:
        if q <= q_thresh:
            val = l2
        else:
            break
    return val

# ── Main loop ─────────────────────────────────────────────────────────────────
rows, all_traj = [], {}
t_start  = time.time()
run_idx  = 0
run_times = []

for cname, cparams in CONDITIONS.items():
    for mname in MODEL_NAMES:
        oracle = MODELS[mname]
        cond_t0 = time.time()
        cond_l2s = []

        for img_idx in range(N_IMG):
            x_orig = images[img_idx]
            y_true = int(labels[img_idx])
            seed   = img_idx * 100 + COND_ORDER.index(cname) * 10

            t0 = time.time()
            result = evolba_tuned(oracle, x_orig, y_true,
                                  max_queries=MAX_Q, seed=seed, **cparams)
            elapsed = time.time() - t0
            run_times.append(elapsed)
            run_idx += 1

            traj      = result['trajectory']
            il2       = result.get('init_l2', float('nan'))
            best_l2   = result.get('best_l2', float('nan'))
            cond_l2s.append(best_l2)
            key = (cname, mname, img_idx)
            all_traj[key] = traj

            row = dict(condition=cname, model=mname, image_idx=img_idx,
                       true_class=int(y_true), success=result['success'],
                       queries=result['queries'], init_l2=il2, best_l2=best_l2)
            for sq in SNAP_QS:
                row[f'l2_at_{sq}'] = l2_at_q(traj, sq, il2)
            row['improvement_ratio'] = (il2 - best_l2) / il2 if il2 > 0 else float('nan')
            rows.append(row)

            # ── per-run line (every image) ─────────────────────────────────
            done_pct = run_idx / total_runs * 100
            remaining_s = (np.mean(run_times) * (total_runs - run_idx)) if run_times else 0
            eta = time.strftime('%H:%M:%S', time.localtime(time.time() + remaining_s))
            log(f'[{run_idx:4d}/{total_runs}  {done_pct:5.1f}%]  '
                f'{cname:<10s} | {mname:<8s} | img {img_idx:3d} '
                f'(class {y_true}) | init_l2={il2:.3f} best_l2={best_l2:.3f} '
                f'q={result["queries"]:4d}  t={elapsed:.1f}s  ETA {eta}')

        # ── per-condition summary ──────────────────────────────────────────
        cond_med = float(np.nanmedian(cond_l2s))
        cond_elapsed = time.time() - cond_t0
        log(f'  --> {cname}/{mname}: median_best_l2={cond_med:.4f}  '
            f'({N_IMG} runs in {cond_elapsed:.0f}s)')

        # ── checkpoint after each condition×model ─────────────────────────
        df_tmp = pd.DataFrame(rows)
        df_tmp.to_parquet(RESULTS_FILE + '.tmp', index=False)
        log(f'  --> checkpoint saved ({len(rows)} rows)')

# ── Final save ────────────────────────────────────────────────────────────────
df = pd.DataFrame(rows)
df.to_parquet(RESULTS_FILE, index=False)
with open(TRAJ_FILE, 'wb') as f:
    pickle.dump(all_traj, f)
total_elapsed = time.time() - t_start
log(f'Saved {RESULTS_FILE} ({len(df)} rows)')
log(f'Saved {TRAJ_FILE}')
log(f'Total wall time: {total_elapsed/3600:.2f}h  ({total_elapsed:.0f}s)')

# ── Console summary table ─────────────────────────────────────────────────────
log('\n=== FINAL SUMMARY ===')
ok = df[df.success]
summary = ok.groupby(['condition','model']).agg(
    n            = ('best_l2', 'count'),
    median_init  = ('init_l2', 'median'),
    median_best  = ('best_l2', 'median'),
    median_IR    = ('improvement_ratio', 'median'),
).round(4)
log('\n' + summary.to_string())

# ── Plots ─────────────────────────────────────────────────────────────────────
log('Generating plots...')

# Plot A: final L2 violin
fig, axes = plt.subplots(1, len(MODEL_NAMES), figsize=(7*len(MODEL_NAMES), 5), sharey=False)
if len(MODEL_NAMES) == 1:
    axes = [axes]
for ax, mname in zip(axes, MODEL_NAMES):
    sub = ok[ok.model == mname]
    data = [sub[sub.condition==c]['best_l2'].dropna().values for c in COND_ORDER]
    parts = ax.violinplot([d if len(d) else [np.nan] for d in data],
                          positions=range(len(COND_ORDER)),
                          showmedians=True, showextrema=False)
    for pc, cname in zip(parts['bodies'], COND_ORDER):
        pc.set_facecolor(COND_COLORS[cname]); pc.set_alpha(0.6)
    parts['cmedians'].set_color('black'); parts['cmedians'].set_linewidth(2)
    for i, d in enumerate(data):
        med = np.median(d) if len(d) else np.nan
        if not np.isnan(med):
            ax.text(i, med+0.05, f'{med:.3f}', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(range(len(COND_ORDER)))
    ax.set_xticklabels(COND_ORDER, rotation=30, ha='right')
    ax.set_ylabel('Final L2'); ax.set_title(f'{mname}'); ax.grid(axis='y', alpha=0.3)
plt.suptitle('A: Final L2 per condition  (lower = better)', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/A_final_l2.png', dpi=130, bbox_inches='tight')
plt.close()
log(f'Saved A_final_l2.png')

# Plot B: convergence curves
fig, axes = plt.subplots(1, len(MODEL_NAMES), figsize=(7*len(MODEL_NAMES), 5))
if len(MODEL_NAMES) == 1:
    axes = [axes]
for ax, mname in zip(axes, MODEL_NAMES):
    sub_df = ok[ok.model == mname]
    for cname in COND_ORDER:
        sub_c = sub_df[sub_df.condition == cname]
        curves = []
        for _, row in sub_c.iterrows():
            key  = (cname, mname, int(row['image_idx']))
            traj = all_traj.get(key, [])
            curve = [l2_at_q(traj, sq, row['init_l2']) for sq in SNAP_QS]
            curves.append(curve)
        if not curves: continue
        arr = np.array(curves)
        med = np.median(arr, axis=0)
        p25 = np.percentile(arr, 25, axis=0)
        p75 = np.percentile(arr, 75, axis=0)
        ax.plot(SNAP_QS, med, color=COND_COLORS[cname], lw=2,
                label=cname, marker='o', ms=4)
        ax.fill_between(SNAP_QS, p25, p75, color=COND_COLORS[cname], alpha=0.12)
    ax.set_xlabel('Oracle queries'); ax.set_ylabel('Median L2')
    ax.set_title(f'{mname}'); ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.suptitle('B: Convergence  (median ± IQR)', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/B_convergence.png', dpi=130, bbox_inches='tight')
plt.close()
log(f'Saved B_convergence.png')

# Plot C: improvement ratio
fig, axes = plt.subplots(1, len(MODEL_NAMES), figsize=(7*len(MODEL_NAMES), 5))
if len(MODEL_NAMES) == 1:
    axes = [axes]
for ax, mname in zip(axes, MODEL_NAMES):
    sub = ok[ok.model == mname]
    meds = [sub[sub.condition==c]['improvement_ratio'].median() for c in COND_ORDER]
    bars = ax.bar(range(len(COND_ORDER)), meds,
                  color=[COND_COLORS[c] for c in COND_ORDER], alpha=0.85)
    for bar, m in zip(bars, meds):
        if not np.isnan(m):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                    f'{m:.3f}', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(range(len(COND_ORDER)))
    ax.set_xticklabels(COND_ORDER, rotation=30, ha='right')
    ax.set_ylim(0,1); ax.set_ylabel('Median IR'); ax.set_title(f'{mname}'); ax.grid(axis='y', alpha=0.3)
plt.suptitle('C: Improvement Ratio  (higher = better)', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/C_improvement_ratio.png', dpi=130, bbox_inches='tight')
plt.close()
log(f'Saved C_improvement_ratio.png')

# Plot D: query efficiency
fig, axes = plt.subplots(1, len(MODEL_NAMES), figsize=(7*len(MODEL_NAMES), 5))
if len(MODEL_NAMES) == 1:
    axes = [axes]
first_q = SNAP_QS[0]; last_q = SNAP_QS[-1]
for ax, mname in zip(axes, MODEL_NAMES):
    sub = ok[ok.model == mname]
    effs = []
    for cname in COND_ORDER:
        sc = sub[sub.condition == cname]
        dl2 = sc[f'l2_at_{first_q}'] - sc[f'l2_at_{last_q}']
        eff = (dl2 / (last_q - first_q) * 100).median()
        effs.append(eff)
    bars = ax.bar(range(len(COND_ORDER)), effs,
                  color=[COND_COLORS[c] for c in COND_ORDER], alpha=0.85)
    for bar, e in zip(bars, effs):
        if not np.isnan(e):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.0001,
                    f'{e:.4f}', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(range(len(COND_ORDER)))
    ax.set_xticklabels(COND_ORDER, rotation=30, ha='right')
    ax.set_ylabel('ΔL2 / 100 queries'); ax.set_title(f'{mname}'); ax.grid(axis='y', alpha=0.3)
plt.suptitle('D: Query efficiency', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/D_query_efficiency.png', dpi=130, bbox_inches='tight')
plt.close()
log(f'Saved D_query_efficiency.png')

log('\nAll done.')
log_f.close()
