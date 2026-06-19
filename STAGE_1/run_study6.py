#!/usr/bin/env python3
"""
run_study6.py — Study 6: backtrack budget (`tau`) — regression & recovery.

Background: each generation, the directed step + binary-search projection
produces a candidate `m_new`. If `||m_new - x_orig|| > dist_to_orig` (the
step REGRESSED -- moved farther from x_orig than `m` was), the algorithm
halves `xi_step` and re-projects, up to `tau` times, then accepts whatever
comes out regardless (m = m_new unconditionally -- see CHANGES.md).

The FIRST proposal (before any backtracking) is independent of `tau` -- it's
a fixed property of the current direction/step-size estimate. `tau` only
controls the RESPONSE: how hard we try to turn a bad proposal into a
non-regression before accepting it, at the cost of up to `tau` extra
binary_search calls (bs_steps queries each) per generation.

This study sweeps tau in {0,1,2,3} (shared HPARAMS otherwise:
xi_step_scale=0.5, lam_override=14, cmu_scale=1.0, bs_steps=15,
bs_adaptive=False) and, per generation, records:
  dist_to_orig        L2 of m at the START of the generation
  l2_pre_backtrack    L2 of the FIRST proposal (before backtracking)
  l2_post_backtrack   L2 of m at the END of the generation (after <=tau backtracks)
  backtracks          how many backtracks were used

From this we derive, per run:
  n_regress_proposed  # gens where l2_pre_backtrack > dist_to_orig (~tau-independent)
  n_net_regress       # gens where l2_post_backtrack > dist_to_orig (m ended up worse)
  n_recovered         # of those, how many gens later does L2 drop back <= dist_to_orig
  n_never_recovered   # ... and how many never do, within the query budget
  median_recovery_queries

Question: does higher tau meaningfully reduce net regressions / improve
final L2, or do most regressions self-heal regardless of tau (in which case
tau's extra backtrack queries are wasted)?

Same seed across tau conditions for a given image (only tau differs).

Usage:
    python run_study6.py           # full run (N=200, Q=2000)
    python run_study6.py --mock    # quick check (N=10, Q=300)
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

OUTPUT_DIR    = f'outputs/study6_{TAG.lower()}'
RESULTS_FILE  = f'{OUTPUT_DIR}/results.parquet'
TRAJ_FILE     = f'{OUTPUT_DIR}/trajectories.pkl'
GENINFO_FILE  = f'{OUTPUT_DIR}/gen_info.pkl'
LOG_FILE      = f'{OUTPUT_DIR}/run.log'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Conditions ────────────────────────────────────────────────────────────────
# Shared "settled" hyperparameters from Study 3/4 (single config for both models),
# bs_steps=15 fixed (decoupled from the still-open bs_adaptive question, Study 5).
HPARAMS = dict(xi_step_scale=0.5, lam_override=14, cmu_scale=1.0,
               bs_steps=15, bs_adaptive=False)

CONDITIONS = {f'tau{t}': dict(tau=t, **HPARAMS) for t in (0, 1, 2, 3)}
COND_ORDER  = ['tau0', 'tau1', 'tau2', 'tau3']
COND_COLORS = {'tau0': '#9E9E9E', 'tau1': '#64B5F6', 'tau2': '#1E88E5', 'tau3': '#0D47A1'}
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
log(f'Shared hyperparams: {HPARAMS}')
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

def analyze_regressions(gen_info, trajectory):
    """Per-run regression/recovery stats. See module docstring for definitions."""
    n_gen = len(gen_info)
    n_regress_proposed = 0
    n_net_regress      = 0
    n_recovered        = 0
    recovery_queries   = []
    for i, g in enumerate(gen_info):
        if g['l2_pre_backtrack'] > g['dist_to_orig']:
            n_regress_proposed += 1
        if g['l2_post_backtrack'] > g['dist_to_orig']:
            n_net_regress += 1
            target = g['dist_to_orig']
            for j in range(i + 1, n_gen):
                if trajectory[j][1] <= target:
                    n_recovered += 1
                    recovery_queries.append(trajectory[j][0] - trajectory[i][0])
                    break
    return dict(
        n_regress_proposed=n_regress_proposed,
        n_net_regress=n_net_regress,
        n_recovered=n_recovered,
        n_never_recovered=n_net_regress - n_recovered,
        median_recovery_queries=(float(np.median(recovery_queries))
                                  if recovery_queries else float('nan')),
    )

# ── Main loop ─────────────────────────────────────────────────────────────────
rows, all_traj, all_gen_info = [], {}, {}
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
            seed   = img_idx * 100   # same seed across tau conditions for a given image

            gen_info = []
            t0 = time.time()
            result = evolba_tuned(oracle, x_orig, y_true,
                                  max_queries=MAX_Q, seed=seed,
                                  collect_gen_info=gen_info, **cparams)
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
            all_traj[key]     = traj
            all_gen_info[key] = gen_info

            stats = analyze_regressions(gen_info, traj)
            bt_total = result.get('backtracks', 0)

            row = dict(condition=cname, model=mname, image_idx=img_idx,
                       true_class=int(y_true), success=result['success'],
                       queries=result['queries'], init_l2=il2,
                       best_l2=best_l2, final_l2=fin_l2,
                       n_generations=n_gen,
                       backtracks_total=bt_total,
                       backtrack_per_gen=(bt_total/n_gen if n_gen else float('nan')),
                       extra_bt_queries=bt_total * HPARAMS['bs_steps'],
                       n_regress_proposed=stats['n_regress_proposed'],
                       n_net_regress=stats['n_net_regress'],
                       n_recovered=stats['n_recovered'],
                       n_never_recovered=stats['n_never_recovered'],
                       regress_rate=(stats['n_regress_proposed']/n_gen if n_gen else float('nan')),
                       net_regress_rate=(stats['n_net_regress']/n_gen if n_gen else float('nan')),
                       recovery_rate=(stats['n_recovered']/stats['n_net_regress']
                                      if stats['n_net_regress'] else float('nan')),
                       median_recovery_queries=stats['median_recovery_queries'])
            for sq in SNAP_QS:
                row[f'l2_at_{sq}'] = l2_at_q(traj, sq, il2)
            row['improvement_ratio'] = (
                (il2 - best_l2) / il2 if (il2 > 0 and not np.isnan(il2)) else float('nan'))
            rows.append(row)

            done_pct = run_idx / total_runs * 100
            remaining_s = (np.mean(run_times) * (total_runs - run_idx)) if run_times else 0
            eta = time.strftime('%H:%M:%S', time.localtime(time.time() + remaining_s))
            log(f'[{run_idx:4d}/{total_runs}  {done_pct:5.1f}%]  '
                f'{cname:<6s} | {mname:<8s} | img {img_idx:3d} '
                f'(class {y_true}) | init_l2={il2:.3f} best_l2={best_l2:.3f} '
                f'final_l2={fin_l2:.3f} q={result["queries"]:4d} gen={n_gen:3d} '
                f'net_regress={stats["n_net_regress"]:2d} '
                f'recov={stats["n_recovered"]:2d} '
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
with open(GENINFO_FILE, 'wb') as f:
    pickle.dump(all_gen_info, f)
total_elapsed = time.time() - t_start
log(f'Saved {RESULTS_FILE} ({len(df)} rows)')
log(f'Saved {TRAJ_FILE}')
log(f'Saved {GENINFO_FILE}')
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
    regress_rate    = ('regress_rate', 'median'),
    net_regress_rate= ('net_regress_rate', 'median'),
    recovery_rate   = ('recovery_rate', 'median'),
    median_recov_q  = ('median_recovery_queries', 'median'),
    bt_per_gen      = ('backtrack_per_gen', 'median'),
).round(4)
summary = summary.reindex(
    pd.MultiIndex.from_product([COND_ORDER, MODEL_NAMES], names=['condition','model'])
)
log('\n' + summary.to_string())

# ── Plots ─────────────────────────────────────────────────────────────────────
log('Generating plots...')

def violin_panel(metric, title, fname, ylabel, ylim=None):
    fig, axes = plt.subplots(1, len(MODEL_NAMES), figsize=(7*len(MODEL_NAMES), 5))
    axes = np.atleast_1d(axes)
    for ax, mname in zip(axes, MODEL_NAMES):
        sub = ok[ok.model == mname]
        data = [sub[sub.condition==c][metric].dropna().values for c in COND_ORDER]
        parts = ax.violinplot([d if len(d) else [np.nan] for d in data],
                              positions=range(len(COND_ORDER)),
                              showmedians=True, showextrema=False)
        for pc, c in zip(parts['bodies'], COND_ORDER):
            pc.set_facecolor(COND_COLORS[c]); pc.set_alpha(0.7)
        parts['cmedians'].set_color('black'); parts['cmedians'].set_linewidth(2)
        for i, d in enumerate(data):
            med = np.nanmedian(d) if len(d) else np.nan
            if not np.isnan(med):
                ax.text(i, med, f'{med:.3f}', ha='center', va='bottom', fontsize=9)
        ax.set_xticks(range(len(COND_ORDER)))
        ax.set_xticklabels(COND_ORDER, fontsize=10)
        ax.set_ylabel(ylabel)
        if ylim: ax.set_ylim(*ylim)
        ax.set_title(mname); ax.grid(axis='y', alpha=0.3)
    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/{fname}', dpi=130, bbox_inches='tight'); plt.close()
    log(f'Saved {fname}')

# Plot A: final L2 (best_l2) violin per tau
violin_panel('best_l2', 'A: Final (best) L2 vs tau', 'A_final_l2.png', 'best_l2 (lower=better)')

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

# Plot C: regression rates -- proposed (~tau-independent) vs net (tau-dependent)
fig, axes = plt.subplots(1, len(MODEL_NAMES), figsize=(7*len(MODEL_NAMES), 5))
axes = np.atleast_1d(axes)
for ax, mname in zip(axes, MODEL_NAMES):
    sub = ok[ok.model == mname]
    proposed = [sub[sub.condition==c]['regress_rate'].median() for c in COND_ORDER]
    net      = [sub[sub.condition==c]['net_regress_rate'].median() for c in COND_ORDER]
    x = np.arange(len(COND_ORDER))
    w = 0.35
    ax.bar(x - w/2, proposed, w, label='proposed (pre-backtrack)', color='#FFB74D', alpha=0.85)
    ax.bar(x + w/2, net,      w, label='net (post-backtrack)',     color='#E53935', alpha=0.85)
    for xi_, v in zip(x - w/2, proposed):
        ax.text(xi_, v, f'{v:.2f}', ha='center', va='bottom', fontsize=8)
    for xi_, v in zip(x + w/2, net):
        ax.text(xi_, v, f'{v:.2f}', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(COND_ORDER, fontsize=10)
    ax.set_ylabel('Fraction of generations'); ax.set_ylim(0, 1)
    ax.set_title(mname); ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)
plt.suptitle('C: Regression rate -- proposed vs surviving backtracking', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/C_regression_rates.png', dpi=130, bbox_inches='tight'); plt.close()
log('Saved C_regression_rates.png')

# Plot D: recovery rate and median recovery cost (queries)
fig, axes = plt.subplots(1, len(MODEL_NAMES), figsize=(7*len(MODEL_NAMES), 5))
axes = np.atleast_1d(axes)
for ax, mname in zip(axes, MODEL_NAMES):
    sub = ok[ok.model == mname]
    rec_rate = [sub[sub.condition==c]['recovery_rate'].median() for c in COND_ORDER]
    bars = ax.bar(range(len(COND_ORDER)), rec_rate,
                  color=[COND_COLORS[c] for c in COND_ORDER], alpha=0.85)
    for bar, v, cname in zip(bars, rec_rate, COND_ORDER):
        rq = sub[sub.condition==cname]['median_recovery_queries'].median()
        label = f'{v:.2f}' if not np.isnan(v) else 'n/a'
        if not np.isnan(rq):
            label += f'\n(~{rq:.0f}q)'
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(),
                label, ha='center', va='bottom', fontsize=8)
    ax.set_xticks(range(len(COND_ORDER))); ax.set_xticklabels(COND_ORDER, fontsize=10)
    ax.set_ylim(0, 1.15); ax.set_ylabel('Recovery rate (fraction of net-regressions)')
    ax.set_title(f'{mname}  (label = median queries-to-recover)')
    ax.grid(axis='y', alpha=0.3)
plt.suptitle('D: Of the regressions that survive backtracking, how many recover later?', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/D_recovery.png', dpi=130, bbox_inches='tight'); plt.close()
log('Saved D_recovery.png')

# Plot E: backtracking cost (extra queries spent) vs generations completed
fig, axes = plt.subplots(1, len(MODEL_NAMES), figsize=(7*len(MODEL_NAMES), 5))
axes = np.atleast_1d(axes)
for ax, mname in zip(axes, MODEL_NAMES):
    sub = ok[ok.model == mname]
    bt_q = [sub[sub.condition==c]['extra_bt_queries'].median() for c in COND_ORDER]
    gens = [sub[sub.condition==c]['n_generations'].median() for c in COND_ORDER]
    ax2 = ax.twinx()
    bars = ax.bar(range(len(COND_ORDER)), bt_q,
                  color=[COND_COLORS[c] for c in COND_ORDER], alpha=0.6, label='extra BT queries')
    ax2.plot(range(len(COND_ORDER)), gens, color='black', marker='o', label='n_generations')
    for bar, v in zip(bars, bt_q):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(), f'{v:.0f}',
                ha='center', va='bottom', fontsize=8)
    for xi_, v in zip(range(len(COND_ORDER)), gens):
        ax2.text(xi_, v, f'{v:.0f}', ha='center', va='bottom', fontsize=8, color='black')
    ax.set_xticks(range(len(COND_ORDER))); ax.set_xticklabels(COND_ORDER, fontsize=10)
    ax.set_ylabel('Median extra backtrack queries (per run)')
    ax2.set_ylabel('Median n_generations')
    ax.set_title(mname); ax.grid(axis='y', alpha=0.3)
plt.suptitle('E: Backtracking cost vs generations completed (within Q budget)', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/E_backtrack_cost.png', dpi=130, bbox_inches='tight'); plt.close()
log('Saved E_backtrack_cost.png')

log('\nAll done.')
log_f.close()
