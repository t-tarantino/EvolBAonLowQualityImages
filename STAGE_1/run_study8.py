#!/usr/bin/env python3
"""
run_study8.py — Study 8: lam x xi_step_scale joint sweep (3x3 grid).

Study 7 found that the carrier config's xi_step_scale=0.5 (tuned at
lam_override=14 in Studies 3/5/6) does not transfer to lam_override=None
(=28, CIFAR-10 default): generations roughly double (33-34 vs 16-20.5) but
the improvement ratio does NOT improve, and is even slightly worse on the
robust model. This suggested `lam` and `xi_step_scale` interact -- the
choice of step size depends on how many generations you'll get.

This study runs a 3x3 grid:

    lam_override  in {14, 28, 42}
    xi_step_scale in {0.5, 0.75, 1.0}

All other hyperparameters held at the EC1 carrier config:
    tau=3, bs_adaptive=True, bs_cap=26, cmu_scale=1.0

Same seed across conditions for a given image (paired comparison).

Usage:
    python run_study8.py           # full run (N=100, Q=2000, 9x100x2=1800 runs)
    python run_study8.py --mock    # quick check (N=10, Q=300, 9x10x2=180 runs)
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
N_IMG    = 10   if MOCK else 100
SNAP_QS  = [100, 200, 300] if MOCK else [250, 500, 750, 1000, 1500, 2000]
TAG      = 'MOCK' if MOCK else 'FULL'

OUTPUT_DIR    = f'outputs/study8_{TAG.lower()}'
RESULTS_FILE  = f'{OUTPUT_DIR}/results.parquet'
TRAJ_FILE     = f'{OUTPUT_DIR}/trajectories.pkl'
LOG_FILE      = f'{OUTPUT_DIR}/run.log'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Conditions: 3x3 grid ──────────────────────────────────────────────────────
LAM_VALUES = [14, 28, 42]
XI_VALUES  = [0.5, 0.75, 1.0]

def cond_name(lam, xi):
    return f'L{lam}_X{int(round(xi*100)):03d}'

COND_ORDER = [cond_name(lam, xi) for lam in LAM_VALUES for xi in XI_VALUES]
CONDITIONS = {
    cond_name(lam, xi): dict(xi_step_scale=xi, tau=3, bs_adaptive=True, bs_cap=26,
                              cmu_scale=1.0, lam_override=lam)
    for lam in LAM_VALUES for xi in XI_VALUES
}
MODEL_NAMES = ['standard', 'robust']

# Color by lam (one colormap family per lam), shaded by xi
LAM_CMAPS = {14: 'Blues', 28: 'Oranges', 42: 'Greens'}
def cond_color(lam, xi):
    cmap = plt.get_cmap(LAM_CMAPS[lam])
    shade = 0.4 + 0.5 * (XI_VALUES.index(xi) / (len(XI_VALUES) - 1))
    return cmap(shade)
COND_COLORS = {cond_name(lam, xi): cond_color(lam, xi) for lam in LAM_VALUES for xi in XI_VALUES}

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
                       lam=cparams['lam_override'], xi_step_scale=cparams['xi_step_scale'],
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
                f'{cname:<9s} | {mname:<8s} | img {img_idx:3d} '
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

# ── Plots ─────────────────────────────────────────────────────────────────────
log('Generating plots...')

# Plot A: heatmap of median_IR over the lam x xi grid, per model
fig, axes = plt.subplots(1, len(MODEL_NAMES), figsize=(6.5*len(MODEL_NAMES), 5.5))
axes = np.atleast_1d(axes)
for ax, mname in zip(axes, MODEL_NAMES):
    grid = np.full((len(LAM_VALUES), len(XI_VALUES)), np.nan)
    for i, lam in enumerate(LAM_VALUES):
        for j, xi in enumerate(XI_VALUES):
            cname = cond_name(lam, xi)
            vals = ok[(ok.condition==cname)&(ok.model==mname)]['improvement_ratio']
            grid[i, j] = vals.median()
    im = ax.imshow(grid, cmap='viridis', aspect='auto')
    ax.set_xticks(range(len(XI_VALUES))); ax.set_xticklabels(XI_VALUES)
    ax.set_yticks(range(len(LAM_VALUES))); ax.set_yticklabels(LAM_VALUES)
    ax.set_xlabel('xi_step_scale'); ax.set_ylabel('lam')
    ax.set_title(mname)
    for i in range(len(LAM_VALUES)):
        for j in range(len(XI_VALUES)):
            ax.text(j, i, f'{grid[i,j]:.3f}', ha='center', va='center',
                    color='white', fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax, label='median improvement ratio')
plt.suptitle('A: Median improvement ratio over the lam x xi_step_scale grid', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/A_ir_heatmap.png', dpi=130, bbox_inches='tight'); plt.close()
log('Saved A_ir_heatmap.png')

# Plot B: interaction plot -- median IR vs xi_step_scale, one line per lam, per model
fig, axes = plt.subplots(1, len(MODEL_NAMES), figsize=(7*len(MODEL_NAMES), 5))
axes = np.atleast_1d(axes)
for ax, mname in zip(axes, MODEL_NAMES):
    for lam in LAM_VALUES:
        ys = []
        for xi in XI_VALUES:
            cname = cond_name(lam, xi)
            vals = ok[(ok.condition==cname)&(ok.model==mname)]['improvement_ratio']
            ys.append(vals.median())
        ax.plot(XI_VALUES, ys, marker='o', label=f'lam={lam}',
                color=plt.get_cmap(LAM_CMAPS[lam])(0.7))
    ax.set_xlabel('xi_step_scale'); ax.set_ylabel('median improvement ratio')
    ax.set_title(mname); ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.suptitle('B: Interaction -- does the best xi_step_scale shift with lam?', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/B_interaction.png', dpi=130, bbox_inches='tight'); plt.close()
log('Saved B_interaction.png')

# Plot C: violin of best_l2 per condition (grouped by lam), per model
fig, axes = plt.subplots(1, len(MODEL_NAMES), figsize=(11*len(MODEL_NAMES), 5))
axes = np.atleast_1d(axes)
for ax, mname in zip(axes, MODEL_NAMES):
    sub = ok[ok.model == mname]
    data = [sub[sub.condition==c]['best_l2'].dropna().values for c in COND_ORDER]
    parts = ax.violinplot(data, positions=range(len(COND_ORDER)),
                          showmedians=True, showextrema=False)
    for pc, c in zip(parts['bodies'], COND_ORDER):
        pc.set_facecolor(COND_COLORS[c]); pc.set_alpha(0.8)
    parts['cmedians'].set_color('black'); parts['cmedians'].set_linewidth(2)
    for i, d in enumerate(data):
        med = np.nanmedian(d)
        ax.text(i, med, f'{med:.3f}', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(range(len(COND_ORDER)))
    ax.set_xticklabels(COND_ORDER, fontsize=9, rotation=30)
    ax.set_ylabel('best_l2 (lower=better)')
    ax.set_title(mname); ax.grid(axis='y', alpha=0.3)
plt.suptitle('C: Final (best) L2 across the 3x3 grid', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/C_final_l2.png', dpi=130, bbox_inches='tight'); plt.close()
log('Saved C_final_l2.png')

# Plot D: heatmap of median n_generations over the grid, per model (sanity check)
fig, axes = plt.subplots(1, len(MODEL_NAMES), figsize=(6.5*len(MODEL_NAMES), 5.5))
axes = np.atleast_1d(axes)
for ax, mname in zip(axes, MODEL_NAMES):
    grid = np.full((len(LAM_VALUES), len(XI_VALUES)), np.nan)
    for i, lam in enumerate(LAM_VALUES):
        for j, xi in enumerate(XI_VALUES):
            cname = cond_name(lam, xi)
            vals = ok[(ok.condition==cname)&(ok.model==mname)]['n_generations']
            grid[i, j] = vals.median()
    im = ax.imshow(grid, cmap='magma', aspect='auto')
    ax.set_xticks(range(len(XI_VALUES))); ax.set_xticklabels(XI_VALUES)
    ax.set_yticks(range(len(LAM_VALUES))); ax.set_yticklabels(LAM_VALUES)
    ax.set_xlabel('xi_step_scale'); ax.set_ylabel('lam')
    ax.set_title(mname)
    for i in range(len(LAM_VALUES)):
        for j in range(len(XI_VALUES)):
            ax.text(j, i, f'{grid[i,j]:.0f}', ha='center', va='center',
                    color='white', fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax, label='median n_generations')
plt.suptitle('D: Median generations over the grid (sanity check -- lam should dominate)', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/D_generations_heatmap.png', dpi=130, bbox_inches='tight'); plt.close()
log('Saved D_generations_heatmap.png')

log('\nAll done.')
log_f.close()
