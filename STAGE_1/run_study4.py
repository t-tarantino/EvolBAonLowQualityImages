#!/usr/bin/env python3
"""
run_study4.py — Study 4: Multi-init Phase 1 (Study 2) + corrected Phase 3
hyperparameters (Study 3).

Two conditions, both using Study 3's corrected, model-conditional
hyperparameters (bs_steps=15, tau=1, lam_override=14, cmu_scale=1.0,
xi_step_scale=1.0 for standard / 0.5 for robust):
  baseline    uniform-random Phase 1 (current default)
  multi_init  try all 8 Phase-1 strategies, keep the one closest to x_orig

Study 2 found multi-init cuts Phase-1 init L2 by ~40% (standard) but Phase 3
(running with the *broken* xi_step_scale~=0.018) made almost no further
progress (IR~=0.01). Study 3 found the corrected hyperparameters restore
Phase 3's progress (IR~=0.41 standard / 0.066 robust from a uniform-random
start). This study tests whether the two gains compound.

Usage:
    python run_study4.py           # full run (N=100, Q=2000)
    python run_study4.py --mock    # quick check (N=10, Q=300)
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

from evolba_tuned  import evolba_tuned
from evolba_baseline import binary_search, uniform_random_init
from phase1_zoo    import INIT_ZOO

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--mock', action='store_true')
args   = parser.parse_args()

MOCK     = args.mock
MAX_Q    = 300  if MOCK else 2000
N_IMG    = 10   if MOCK else 100
SNAP_QS  = [100, 200, 300] if MOCK else [250, 500, 750, 1000, 1500, 2000]
TAG      = 'MOCK' if MOCK else 'FULL'

OUTPUT_DIR   = f'outputs/study4_{TAG.lower()}'
RESULTS_FILE = f'{OUTPUT_DIR}/results.parquet'
TRAJ_FILE    = f'{OUTPUT_DIR}/trajectories.pkl'
INIT_FILE    = f'{OUTPUT_DIR}/init_breakdown.pkl'   # which init won per image
LOG_FILE     = f'{OUTPUT_DIR}/run.log'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Study 3's corrected, model-conditional hyperparameters.
HPARAMS_BY_MODEL = {
    'standard': dict(xi_step_scale=1.0, bs_steps=15, tau=1, lam_override=14, cmu_scale=1.0),
    'robust':   dict(xi_step_scale=0.5, bs_steps=15, tau=1, lam_override=14, cmu_scale=1.0),
}

# ── Logging ───────────────────────────────────────────────────────────────────
log_f = open(LOG_FILE, 'w', buffering=1)
def log(msg):
    ts   = time.strftime('%H:%M:%S')
    line = f'[{ts}] {msg}'
    print(line, flush=True)
    log_f.write(line + '\n')

# ── Models ────────────────────────────────────────────────────────────────────
from robustbench.utils import load_model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
log(f'Device: {device}')

model_std = load_model('Standard', dataset='cifar10',
                       threat_model='Linf').to(device).eval()
model_rob = load_model('Wang2023Better_WRN-28-10', dataset='cifar10',
                       threat_model='Linf').to(device).eval()

def _mk_oracle(model):
    def oracle(x_chw):
        with torch.no_grad():
            t = torch.from_numpy(x_chw[None].astype(np.float32)).to(device)
            return int(model(t).argmax(1).item())
    return oracle

MODELS     = {'standard': _mk_oracle(model_std), 'robust': _mk_oracle(model_rob)}
MODEL_NAMES = list(MODELS.keys())
log('Models loaded.')

# ── Images ────────────────────────────────────────────────────────────────────
import torchvision
ds        = torchvision.datasets.CIFAR10(root='/tmp/cifar10', train=False, download=True)
per_class = N_IMG // 10
images, labels = [], []
counts = [0] * 10
for img_pil, label in ds:
    if counts[label] >= per_class: continue
    x = np.array(img_pil, dtype=np.float32).transpose(2,0,1) / 255.0
    if MODELS['standard'](x) == label:
        images.append(x); labels.append(label); counts[label] += 1
    if sum(counts) == N_IMG: break
images = np.stack(images)
labels = np.array(labels)
log(f'Images: {len(images)} ({per_class}/class)  label dist: {counts}')

# ── Conditions ────────────────────────────────────────────────────────────────
CONDITIONS = ['baseline', 'multi_init']
COND_COLORS = {'baseline': '#888888', 'multi_init': '#E91E63'}

total_runs = len(CONDITIONS) * N_IMG * len(MODEL_NAMES)
log(f'=== {TAG} RUN ===')
log(f'{len(CONDITIONS)} conditions × {N_IMG} images × {len(MODEL_NAMES)} models = {total_runs} runs')
log(f'Max queries: {MAX_Q}  |  Hyperparams: {HPARAMS_BY_MODEL}')
log(f'Init zoo ({len(INIT_ZOO)}): {list(INIT_ZOO.keys())}')

# ── Utility ───────────────────────────────────────────────────────────────────
def l2_at_q(traj, q_thresh, init_l2):
    val = init_l2
    for q, l2 in traj:
        if q <= q_thresh: val = l2
        else: break
    return val

# ── Per-image multi-init probe (separate from evolba_tuned loop) ───────────────
def run_multi_init_probe(oracle, x_orig, y_true, rng, bs_steps=15):
    """
    Try every init strategy, binary-search each to the boundary,
    return (best_boundary_img, {name: l2 or None}, queries_used).

    `queries_used` counts every oracle call made during the probe — this is
    charged against the run's total query budget so multi_init and baseline
    are compared at equal *total* oracle cost.
    """
    x_orig_flat = x_orig.flatten().astype(np.float64)
    qcount = [0]
    def counted_oracle(x):
        qcount[0] += 1
        return oracle(x)

    best_x, best_l2 = None, float('inf')
    breakdown = {}
    for name, fn in INIT_ZOO.items():
        x_adv = fn(counted_oracle, x_orig, y_true, rng)
        if x_adv is None:
            breakdown[name] = None
            continue
        x_bnd = binary_search(counted_oracle, x_adv, x_orig, y_true, n_steps=bs_steps)
        l2    = float(np.linalg.norm(x_bnd.flatten() - x_orig_flat))
        breakdown[name] = l2
        if l2 < best_l2:
            best_l2 = l2
            best_x  = x_bnd
    return best_x, breakdown, qcount[0]

# ── Main loop ─────────────────────────────────────────────────────────────────
rows, all_traj, all_breakdowns = [], {}, {}
t_start, run_idx, run_times    = time.time(), 0, []

for cname in CONDITIONS:
    for mname in MODEL_NAMES:
        oracle  = MODELS[mname]
        HPARAMS = HPARAMS_BY_MODEL[mname]
        cond_t0  = time.time()
        cond_l2s = []

        for img_idx in range(N_IMG):
            x_orig  = images[img_idx]
            y_true  = int(labels[img_idx])
            seed    = img_idx * 100 + CONDITIONS.index(cname) * 10
            rng     = np.random.default_rng(seed)

            t0 = time.time()

            if cname == 'baseline':
                result = evolba_tuned(oracle, x_orig, y_true,
                                      max_queries=MAX_Q, seed=seed, **HPARAMS)
                breakdown, probe_q = None, 0

            else:  # multi_init
                # Phase 1: probe all strategies, pick the best boundary point.
                # Probe queries are charged against the total MAX_Q budget.
                best_x1, breakdown, probe_q = run_multi_init_probe(
                    oracle, x_orig, y_true, rng, bs_steps=HPARAMS['bs_steps'])

                # If every strategy failed fall back to uniform random
                if best_x1 is None:
                    best_x1 = uniform_random_init(oracle, x_orig.shape, y_true, rng)

                # Phase 3: evolba_tuned with best Phase-1 point as init_fn,
                # given the *remaining* query budget.
                def _init_fn(q, shape, yt, r, _x=best_x1):
                    return _x

                remaining_q = max(MAX_Q - probe_q, 50)
                result = evolba_tuned(oracle, x_orig, y_true,
                                      max_queries=remaining_q, seed=seed,
                                      init_fn=_init_fn, **HPARAMS)
                result['queries'] += probe_q  # total cost incl. Phase-1 probe

            elapsed = time.time() - t0
            run_times.append(elapsed)
            run_idx += 1

            traj    = result['trajectory']
            il2     = result.get('init_l2', float('nan'))
            best_l2 = result.get('best_l2', float('nan'))
            cond_l2s.append(best_l2)

            key = (cname, mname, img_idx)
            all_traj[key]        = traj
            all_breakdowns[key]  = breakdown    # None for baseline

            row = dict(condition=cname, model=mname, image_idx=img_idx,
                       true_class=int(y_true), success=result['success'],
                       queries=result['queries'], probe_queries=probe_q,
                       init_l2=il2, best_l2=best_l2)
            for sq in SNAP_QS:
                row[f'l2_at_{sq}'] = l2_at_q(traj, sq, il2)
            row['improvement_ratio'] = (
                (il2 - best_l2) / il2 if (il2 > 0 and not np.isnan(il2)) else float('nan'))

            # Which init won? (multi_init only)
            if breakdown:
                valid = {k: v for k, v in breakdown.items() if v is not None}
                winner = min(valid, key=valid.get) if valid else 'none'
                row['init_winner']    = winner
                row['init_winner_l2'] = valid.get(winner, float('nan'))
                row['n_inits_success']= len(valid)
                for k, v in breakdown.items():
                    row[f'init_{k}_l2'] = v if v is not None else float('nan')
            else:
                row['init_winner']    = 'uniform_random'
                row['init_winner_l2'] = il2
                row['n_inits_success']= float('nan')
                for k in INIT_ZOO:
                    row[f'init_{k}_l2'] = float('nan')

            rows.append(row)

            # ── per-run log line ──────────────────────────────────────────
            done_pct    = run_idx / total_runs * 100
            remain_s    = np.mean(run_times) * (total_runs - run_idx)
            eta         = time.strftime('%H:%M:%S', time.localtime(time.time() + remain_s))
            winner_str  = (f'  winner={row["init_winner"]}({row["init_winner_l2"]:.3f})'
                           if cname == 'multi_init' else '')
            log(f'[{run_idx:4d}/{total_runs}  {done_pct:5.1f}%]  '
                f'{cname:<10s} | {mname:<8s} | img {img_idx:3d} '
                f'(cls {y_true}) | init_l2={il2:.3f} best_l2={best_l2:.3f} '
                f'q={result["queries"]:4d}  t={elapsed:.1f}s  ETA {eta}'
                f'{winner_str}')

        cond_elapsed = time.time() - cond_t0
        log(f'  --> {cname}/{mname}: median_best_l2='
            f'{float(np.nanmedian(cond_l2s)):.4f}  ({N_IMG} runs in {cond_elapsed:.0f}s)')

        # ── checkpoint ────────────────────────────────────────────────────
        pd.DataFrame(rows).to_parquet(RESULTS_FILE + '.tmp', index=False)
        log(f'  --> checkpoint saved ({len(rows)} rows)')

# ── Final save ────────────────────────────────────────────────────────────────
df = pd.DataFrame(rows)
df.to_parquet(RESULTS_FILE, index=False)
with open(TRAJ_FILE, 'wb') as f: pickle.dump(all_traj, f)
with open(INIT_FILE, 'wb') as f: pickle.dump(all_breakdowns, f)
total_elapsed = time.time() - t_start
log(f'Saved {RESULTS_FILE} ({len(df)} rows)')
log(f'Total wall time: {total_elapsed/3600:.2f}h')

# ── Summary table ─────────────────────────────────────────────────────────────
log('\n=== FINAL SUMMARY ===')
ok = df[df.success]
summary = ok.groupby(['condition','model']).agg(
    n          = ('best_l2', 'count'),
    median_init= ('init_l2', 'median'),
    median_best= ('best_l2', 'median'),
    median_IR  = ('improvement_ratio', 'median'),
).round(4)
log('\n' + summary.to_string())

# Init-winner breakdown (multi_init only)
mi = ok[ok.condition == 'multi_init']
if len(mi):
    log('\n--- Multi-init winner breakdown ---')
    for mname in MODEL_NAMES:
        sub = mi[mi.model == mname]
        counts = sub['init_winner'].value_counts()
        log(f'  {mname}: {counts.to_dict()}')
    log('\n--- Median init L2 per strategy (multi_init) ---')
    init_cols = [f'init_{k}_l2' for k in INIT_ZOO]
    for mname in MODEL_NAMES:
        sub = mi[mi.model == mname]
        meds = sub[init_cols].median().sort_values()
        log(f'  {mname}:')
        for col, val in meds.items():
            name = col.replace('init_','').replace('_l2','')
            log(f'    {name:<15s}: {val:.4f}')

# ── Plots ─────────────────────────────────────────────────────────────────────
log('Generating plots...')

# Plot A: init_l2 comparison (phase-1 quality)
fig, axes = plt.subplots(1, len(MODEL_NAMES), figsize=(7*len(MODEL_NAMES), 5))
axes = np.atleast_1d(axes)
for ax, mname in zip(axes, MODEL_NAMES):
    sub = ok[ok.model == mname]
    data = [sub[sub.condition==c]['init_l2'].dropna().values for c in CONDITIONS]
    parts = ax.violinplot([d if len(d) else [np.nan] for d in data],
                          positions=range(len(CONDITIONS)),
                          showmedians=True, showextrema=False)
    for pc, c in zip(parts['bodies'], CONDITIONS):
        pc.set_facecolor(COND_COLORS[c]); pc.set_alpha(0.6)
    parts['cmedians'].set_color('black'); parts['cmedians'].set_linewidth(2)
    for i, d in enumerate(data):
        med = np.median(d) if len(d) else np.nan
        if not np.isnan(med):
            ax.text(i, med+0.05, f'{med:.3f}', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(range(len(CONDITIONS)))
    ax.set_xticklabels(CONDITIONS, fontsize=10)
    ax.set_ylabel('Phase-1 init L2  (lower = better start)')
    ax.set_title(mname); ax.grid(axis='y', alpha=0.3)
plt.suptitle('A: Phase-1 init L2  —  does multi-init find a closer boundary point?', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/A_init_l2.png', dpi=130, bbox_inches='tight'); plt.close()
log('Saved A_init_l2.png')

# Plot B: final L2 comparison
fig, axes = plt.subplots(1, len(MODEL_NAMES), figsize=(7*len(MODEL_NAMES), 5))
axes = np.atleast_1d(axes)
for ax, mname in zip(axes, MODEL_NAMES):
    sub  = ok[ok.model == mname]
    data = [sub[sub.condition==c]['best_l2'].dropna().values for c in CONDITIONS]
    parts = ax.violinplot([d if len(d) else [np.nan] for d in data],
                          positions=range(len(CONDITIONS)),
                          showmedians=True, showextrema=False)
    for pc, c in zip(parts['bodies'], CONDITIONS):
        pc.set_facecolor(COND_COLORS[c]); pc.set_alpha(0.6)
    parts['cmedians'].set_color('black'); parts['cmedians'].set_linewidth(2)
    for i, d in enumerate(data):
        med = np.median(d) if len(d) else np.nan
        if not np.isnan(med):
            ax.text(i, med+0.05, f'{med:.3f}', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(range(len(CONDITIONS)))
    ax.set_xticklabels(CONDITIONS, fontsize=10)
    ax.set_ylabel('Final L2  (lower = better AE)')
    ax.set_title(mname); ax.grid(axis='y', alpha=0.3)
plt.suptitle('B: Final L2  —  does better Phase 1 + corrected Phase 3 compound?', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/B_final_l2.png', dpi=130, bbox_inches='tight'); plt.close()
log('Saved B_final_l2.png')

# Plot C: convergence curves
fig, axes = plt.subplots(1, len(MODEL_NAMES), figsize=(7*len(MODEL_NAMES), 5))
axes = np.atleast_1d(axes)
for ax, mname in zip(axes, MODEL_NAMES):
    sub_df = ok[ok.model == mname]
    for cname in CONDITIONS:
        sub_c  = sub_df[sub_df.condition == cname]
        curves = []
        for _, row in sub_c.iterrows():
            key   = (cname, mname, int(row['image_idx']))
            traj  = all_traj.get(key, [])
            curve = [l2_at_q(traj, sq, row['init_l2']) for sq in SNAP_QS]
            curves.append(curve)
        if not curves: continue
        arr = np.array(curves)
        med = np.median(arr, axis=0)
        p25 = np.percentile(arr, 25, axis=0)
        p75 = np.percentile(arr, 75, axis=0)
        ax.plot(SNAP_QS, med, color=COND_COLORS[cname], lw=2,
                label=cname, marker='o', ms=4)
        ax.fill_between(SNAP_QS, p25, p75, color=COND_COLORS[cname], alpha=0.15)
    ax.set_xlabel('Oracle queries'); ax.set_ylabel('Median L2')
    ax.set_title(mname); ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.suptitle('C: Convergence  (median ± IQR)', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/C_convergence.png', dpi=130, bbox_inches='tight'); plt.close()
log('Saved C_convergence.png')

# Plot D: which init strategy wins (multi_init only)
mi = ok[ok.condition == 'multi_init']
if len(mi):
    fig, axes = plt.subplots(1, len(MODEL_NAMES), figsize=(10*len(MODEL_NAMES), 5))
    axes = np.atleast_1d(axes)
    init_names = list(INIT_ZOO.keys())
    palette = plt.cm.tab10(np.linspace(0, 1, len(init_names)))

    for ax, mname in zip(axes, MODEL_NAMES):
        sub    = mi[mi.model == mname]
        counts = sub['init_winner'].value_counts().reindex(init_names, fill_value=0)
        bars   = ax.bar(range(len(init_names)), counts.values,
                        color=palette, alpha=0.85)
        for bar, cnt in zip(bars, counts.values):
            if cnt > 0:
                ax.text(bar.get_x()+bar.get_width()/2,
                        bar.get_height()+0.2, str(cnt),
                        ha='center', va='bottom', fontsize=9)
        ax.set_xticks(range(len(init_names)))
        ax.set_xticklabels(init_names, rotation=35, ha='right', fontsize=9)
        ax.set_ylabel('# images won'); ax.set_title(mname); ax.grid(axis='y', alpha=0.3)

    plt.suptitle('D: Which Phase-1 strategy wins (= lowest init L2)', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/D_init_winner.png', dpi=130, bbox_inches='tight'); plt.close()
    log('Saved D_init_winner.png')

    # Plot E: per-strategy median init L2 heatmap
    init_cols = [f'init_{k}_l2' for k in init_names]
    fig, axes = plt.subplots(1, len(MODEL_NAMES), figsize=(10*len(MODEL_NAMES), 4))
    axes = np.atleast_1d(axes)
    for ax, mname in zip(axes, MODEL_NAMES):
        sub  = mi[mi.model == mname]
        meds = [sub[c].median() for c in init_cols]
        bars = ax.bar(range(len(init_names)), meds, color=palette, alpha=0.85)
        for bar, med in zip(bars, meds):
            if not np.isnan(med):
                ax.text(bar.get_x()+bar.get_width()/2,
                        bar.get_height()+0.02,
                        f'{med:.2f}', ha='center', va='bottom', fontsize=8)
        ax.set_xticks(range(len(init_names)))
        ax.set_xticklabels(init_names, rotation=35, ha='right', fontsize=9)
        ax.set_ylabel('Median Phase-1 boundary L2'); ax.set_title(mname); ax.grid(axis='y', alpha=0.3)
    plt.suptitle('E: Median Phase-1 L2 per strategy after boundary projection', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/E_init_l2_per_strategy.png', dpi=130, bbox_inches='tight'); plt.close()
    log('Saved E_init_l2_per_strategy.png')

log('\nAll done.')
log_f.close()
