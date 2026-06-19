#!/usr/bin/env python3
"""
run_study7.py — Study 7: does the tuning compound? baseline vs fully-tuned.

Studies 1-6 settled on, in isolation:
  - xi_step_scale=0.5   (Study 3: directed-step scale, vs baseline's xi_step=xi)
  - bs_adaptive=True    (Study 5: lossless early-stop binary search, ~5-7% saved)
  - tau=0               (Study 6: no backtracking, + best_l2 running-min fix)
  - cmu_scale=1.0       (Study 3: already == baseline's default, unchanged)
  - lam_override=None   (NOT tuned yet -- left at baseline's default, 28 for CIFAR-10)

This study runs each image through BOTH a faithful reproduction of
evolba_baseline()'s hyperparameters and the fully-assembled tuned config
(both via evolba_tuned(), so the best_l2 reporting fix applies equally to
both -- it isolates exactly the 3 hyperparameter changes above):

  BASELINE: xi_step_scale=1.0, tau=3, bs_steps=26, bs_adaptive=False, cmu_scale=1.0
  TUNED:    xi_step_scale=0.5, tau=0, bs_adaptive=True, bs_cap=26,    cmu_scale=1.0

(lam_override=None in both -- left for a future study.)

Question: do the individually-validated improvements stack into a clear
net win, or do they interact/cancel?

Same seed across conditions for a given image (paired comparison).

Usage:
    python run_study7.py           # full run (N=200, Q=2000)
    python run_study7.py --mock    # quick check (N=10, Q=300)
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
parser.add_argument('--mock', action='store_true')
args   = parser.parse_args()

MOCK     = args.mock
MAX_Q    = 300  if MOCK else 2000
N_IMG    = 10   if MOCK else 200
SNAP_QS  = [100, 200, 300] if MOCK else [250, 500, 750, 1000, 1500, 2000]
TAG      = 'MOCK' if MOCK else 'FULL'

OUTPUT_DIR    = f'outputs/study7_{TAG.lower()}'
RESULTS_FILE  = f'{OUTPUT_DIR}/results.parquet'
TRAJ_FILE     = f'{OUTPUT_DIR}/trajectories.pkl'
LOG_FILE      = f'{OUTPUT_DIR}/run.log'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Conditions ────────────────────────────────────────────────────────────────
CONDITIONS = {
    'baseline': dict(xi_step_scale=1.0, tau=3, bs_steps=26, bs_adaptive=False,
                      cmu_scale=1.0, lam_override=None),
    'tuned':    dict(xi_step_scale=0.5, tau=0, bs_adaptive=True, bs_cap=26,
                      cmu_scale=1.0, lam_override=None),
}
COND_ORDER  = ['baseline', 'tuned']
COND_COLORS = {'baseline': '#9E9E9E', 'tuned': '#1E88E5'}
MODEL_NAMES = ['standard', 'robust']

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
total_runs = len(CONDITIONS) * N_IMG * len(MODEL_NAMES)
log(f'=== {TAG} RUN ===')
log(f'{len(CONDITIONS)} conditions ({COND_ORDER}) x {N_IMG} images x '
    f'{len(MODEL_NAMES)} models = {total_runs} runs')
for cname, cparams in CONDITIONS.items():
    log(f'{cname}: {cparams}')
log(f'Max queries per run: {MAX_Q}  |  snapshots at: {SNAP_QS}')
log(f'Output dir: {OUTPUT_DIR}')

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

for cname in COND_ORDER:
    cparams = CONDITIONS[cname]
    for mname in MODEL_NAMES:
        oracle = MODELS[mname]
        cond_t0 = time.time()
        cond_l2s = []

        for img_idx in range(N_IMG):
            x_orig = images[img_idx]
            y_true = int(labels[img_idx])
            seed   = img_idx * 100   # same seed across conditions for a given image

            t0 = time.time()
            result = evolba_tuned(oracle, x_orig, y_true,
                                  max_queries=MAX_Q, seed=seed, **cparams)
            elapsed = time.time() - t0
            run_times.append(elapsed)
            run_idx += 1

            traj    = result['trajectory']
            il2     = result.get('init_l2', float('nan'))
            best_l2 = result.get('best_l2', float('nan'))
            fin_l2  = result.get('final_l2', float('nan'))
            n_gen   = result.get('n_generations', 0)
            cond_l2s.append(best_l2)

            key = (cname, mname, img_idx)
            all_traj[key] = traj

            row = dict(condition=cname, model=mname, image_idx=img_idx,
                       true_class=int(y_true), success=result['success'],
                       queries=result['queries'], init_l2=il2,
                       best_l2=best_l2, final_l2=fin_l2,
                       n_generations=n_gen,
                       backtracks_total=result.get('backtracks', 0),
                       bs_calls=result.get('bs_calls', 0),
                       bs_queries_actual=result.get('bs_queries_actual', 0))
            for sq in SNAP_QS:
                row[f'l2_at_{sq}'] = l2_at_q(traj, sq, il2)
            row['improvement_ratio'] = (
                (il2 - best_l2) / il2 if (il2 > 0 and not np.isnan(il2)) else float('nan'))
            rows.append(row)

            done_pct = run_idx / total_runs * 100
            remaining_s = (np.mean(run_times) * (total_runs - run_idx)) if run_times else 0
            eta = time.strftime('%H:%M:%S', time.localtime(time.time() + remaining_s))
            log(f'[{run_idx:4d}/{total_runs}  {done_pct:5.1f}%]  '
                f'{cname:<8s} | {mname:<8s} | img {img_idx:3d} '
                f'(class {y_true}) | init_l2={il2:.3f} best_l2={best_l2:.3f} '
                f'final_l2={fin_l2:.3f} q={result["queries"]:4d} gen={n_gen:3d} '
                f't={elapsed:.1f}s  ETA {eta}')

        cond_med = float(np.nanmedian(cond_l2s))
        cond_elapsed = time.time() - cond_t0
        log(f'  --> {cname}/{mname}: median_best_l2={cond_med:.4f}  '
            f'({N_IMG} runs in {cond_elapsed:.0f}s)')

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
    pd.MultiIndex.from_product([COND_ORDER, MODEL_NAMES], names=['condition','model']))
log('\n' + summary.to_string())

# Paired win-rate (per image, tuned vs baseline best_l2)
log('\n=== PAIRED COMPARISON (per image, tuned vs baseline) ===')
for mname in MODEL_NAMES:
    base = ok[(ok.condition=='baseline')&(ok.model==mname)].set_index('image_idx')['best_l2']
    tun  = ok[(ok.condition=='tuned')   &(ok.model==mname)].set_index('image_idx')['best_l2']
    common = base.index.intersection(tun.index)
    b, t_ = base.loc[common], tun.loc[common]
    win_rate = float((t_ < b).mean())
    rel_impr = float(np.median((b - t_) / b))
    log(f'{mname}: n={len(common)}  tuned-wins={win_rate*100:.1f}%  '
        f'median relative improvement={rel_impr*100:.1f}%')

# ── Plots ─────────────────────────────────────────────────────────────────────
log('Generating plots...')

# Plot A: final L2 (best_l2) violin per condition
fig, axes = plt.subplots(1, len(MODEL_NAMES), figsize=(7*len(MODEL_NAMES), 5))
axes = np.atleast_1d(axes)
for ax, mname in zip(axes, MODEL_NAMES):
    sub = ok[ok.model == mname]
    data = [sub[sub.condition==c]['best_l2'].dropna().values for c in COND_ORDER]
    parts = ax.violinplot(data, positions=range(len(COND_ORDER)),
                          showmedians=True, showextrema=False)
    for pc, c in zip(parts['bodies'], COND_ORDER):
        pc.set_facecolor(COND_COLORS[c]); pc.set_alpha(0.7)
    parts['cmedians'].set_color('black'); parts['cmedians'].set_linewidth(2)
    for i, d in enumerate(data):
        med = np.nanmedian(d)
        ax.text(i, med, f'{med:.3f}', ha='center', va='bottom', fontsize=10)
    ax.set_xticks(range(len(COND_ORDER))); ax.set_xticklabels(COND_ORDER, fontsize=10)
    ax.set_ylabel('best_l2 (lower=better)')
    ax.set_title(mname); ax.grid(axis='y', alpha=0.3)
plt.suptitle('A: Final (best) L2 -- baseline vs fully-tuned', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/A_final_l2.png', dpi=130, bbox_inches='tight'); plt.close()
log('Saved A_final_l2.png')

# Plot B: convergence curves
fig, axes = plt.subplots(1, len(MODEL_NAMES), figsize=(7*len(MODEL_NAMES), 5))
axes = np.atleast_1d(axes)
for ax, mname in zip(axes, MODEL_NAMES):
    sub_df = ok[ok.model == mname]
    for cname in COND_ORDER:
        sub_c = sub_df[sub_df.condition == cname]
        curves = []
        for _, row in sub_c.iterrows():
            key   = (cname, mname, int(row['image_idx']))
            traj  = all_traj.get(key, [])
            curve = [l2_at_q(traj, sq, row['init_l2']) for sq in SNAP_QS]
            curves.append(curve)
        if not curves: continue
        arr = np.array(curves)
        med = np.median(arr, axis=0)
        ax.plot(SNAP_QS, med, color=COND_COLORS[cname], lw=2,
                label=cname, marker='o', ms=4)
    ax.set_xlabel('Oracle queries'); ax.set_ylabel('Median L2 (current, not best)')
    ax.set_title(mname); ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.suptitle('B: Convergence (median L2 at query checkpoints)', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/B_convergence.png', dpi=130, bbox_inches='tight'); plt.close()
log('Saved B_convergence.png')

# Plot C: paired per-image scatter (baseline vs tuned best_l2)
fig, axes = plt.subplots(1, len(MODEL_NAMES), figsize=(6*len(MODEL_NAMES), 6))
axes = np.atleast_1d(axes)
for ax, mname in zip(axes, MODEL_NAMES):
    base = ok[(ok.condition=='baseline')&(ok.model==mname)].set_index('image_idx')['best_l2']
    tun  = ok[(ok.condition=='tuned')   &(ok.model==mname)].set_index('image_idx')['best_l2']
    common = base.index.intersection(tun.index)
    b, t_ = base.loc[common].values, tun.loc[common].values
    ax.scatter(b, t_, s=18, alpha=0.6, color=COND_COLORS['tuned'])
    lim = [0, max(b.max(), t_.max()) * 1.05]
    ax.plot(lim, lim, 'k--', lw=1, label='y = x (no change)')
    win_rate = float((t_ < b).mean())
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel('baseline best_l2'); ax.set_ylabel('tuned best_l2')
    ax.set_title(f'{mname}  (tuned better in {win_rate*100:.0f}% of images)')
    ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.set_aspect('equal')
plt.suptitle('C: Per-image paired comparison (points below diagonal = tuned wins)', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/C_paired.png', dpi=130, bbox_inches='tight'); plt.close()
log('Saved C_paired.png')

# Plot D: improvement ratio + generations + bs query cost
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
metrics = [('improvement_ratio', 'Improvement ratio (higher=better)'),
           ('n_generations', 'n_generations'),
           ('bs_queries_actual', 'Binary-search queries used')]
for ax, (metric, ylabel) in zip(axes, metrics):
    x = np.arange(len(MODEL_NAMES))
    w = 0.35
    for i, cname in enumerate(COND_ORDER):
        vals = [ok[(ok.model==m)&(ok.condition==cname)][metric].median() for m in MODEL_NAMES]
        offset = (i - 0.5) * w
        bars = ax.bar(x + offset, vals, w, label=cname, color=COND_COLORS[cname], alpha=0.85)
        for xi_, v in zip(x + offset, vals):
            ax.text(xi_, v, f'{v:.3f}' if metric=='improvement_ratio' else f'{v:.0f}',
                    ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(MODEL_NAMES)
    ax.set_ylabel(ylabel); ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
plt.suptitle('D: Summary -- improvement ratio, generations, binary-search cost', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/D_summary.png', dpi=130, bbox_inches='tight'); plt.close()
log('Saved D_summary.png')

log('\nAll done.')
log_f.close()
