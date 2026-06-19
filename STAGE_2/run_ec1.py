#!/usr/bin/env python3
"""
run_ec1.py — Stage 2, Study EC1: CMA-ES variant comparison.

5 conditions, all sharing the same "carrier" hyperparameters (the Study 7
"tuned except tau=3" config: xi_step_scale=0.5, bs_adaptive=True,
cmu_scale=1.0, lam_override=None, tau=3):

  sep   : evolba_vkd(vk_rank=0)  -- sep-CMA-ES (== evolba_tuned, validated)
  vd1   : evolba_vkd(vk_rank=1)  -- VD-CMA
  vd2   : evolba_vkd(vk_rank=2)  -- VkD-CMA, k=2
  vd3   : evolba_vkd(vk_rank=3)  -- VkD-CMA, k=3
  o11   : evolba_one_plus_one()  -- (1+1)-CMA-ES, structurally different
          generation loop (1 query/gen, elitist), bs_adaptive=True

Full CMA-ES and LM-CMA were considered and discarded -- see CHANGES.md.

Usage:
    python run_ec1.py           # full run (N=100, Q=2000)
    python run_ec1.py --mock    # quick check (N=10, Q=300)
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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'STAGE_1'))
sys.path.insert(0, os.path.dirname(__file__))

from evolba_ec import evolba_vkd, evolba_one_plus_one

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--mock', action='store_true')
args   = parser.parse_args()

MOCK     = args.mock
MAX_Q    = 300  if MOCK else 2000
N_IMG    = 10   if MOCK else 100
SNAP_QS  = [100, 200, 300] if MOCK else [250, 500, 750, 1000, 1500, 2000]
TAG      = 'MOCK' if MOCK else 'FULL'

OUTPUT_DIR     = f'outputs/ec1_{TAG.lower()}'
RESULTS_FILE   = f'{OUTPUT_DIR}/results.parquet'
TRAJ_FILE      = f'{OUTPUT_DIR}/trajectories.pkl'
VNORMS_FILE    = f'{OUTPUT_DIR}/v_norms.pkl'
O11_DIAG_FILE  = f'{OUTPUT_DIR}/o11_diag.pkl'
LOG_FILE       = f'{OUTPUT_DIR}/run.log'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Conditions ────────────────────────────────────────────────────────────────
CARRIER = dict(xi_step_scale=0.5, bs_adaptive=True, cmu_scale=1.0,
               lam_override=None, tau=3)
VK_RANK = {'sep': 0, 'vd1': 1, 'vd2': 2, 'vd3': 3}
COND_ORDER  = ['sep', 'vd1', 'vd2', 'vd3', 'o11']
COND_COLORS = {'sep': '#9E9E9E', 'vd1': '#90CAF9', 'vd2': '#42A5F5',
               'vd3': '#1565C0', 'o11': '#E53935'}
MODEL_NAMES = ['standard', 'robust']


def run_condition(cname, oracle, x_orig, y_true, seed):
    if cname == 'o11':
        return evolba_one_plus_one(oracle, x_orig, y_true, max_queries=MAX_Q,
                                    bs_adaptive=True, seed=seed)
    return evolba_vkd(oracle, x_orig, y_true, vk_rank=VK_RANK[cname],
                      max_queries=MAX_Q, seed=seed, **CARRIER)


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
total_runs = len(COND_ORDER) * N_IMG * len(MODEL_NAMES)
log(f'=== {TAG} RUN ===')
log(f'{len(COND_ORDER)} conditions ({COND_ORDER}) x {N_IMG} images x '
    f'{len(MODEL_NAMES)} models = {total_runs} runs')
log(f'Carrier config (sep/vd1/vd2/vd3): {CARRIER}')
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
rows, all_traj, v_norms_store, o11_diag = [], {}, {}, {}
t_start  = time.time()
run_idx  = 0
run_times = []

for cname in COND_ORDER:
    for mname in MODEL_NAMES:
        oracle = MODELS[mname]
        cond_t0 = time.time()
        cond_l2s = []

        for img_idx in range(N_IMG):
            x_orig = images[img_idx]
            y_true = int(labels[img_idx])
            seed   = img_idx * 100   # same seed across conditions for a given image

            t0 = time.time()
            result = run_condition(cname, oracle, x_orig, y_true, seed)
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
            if cname in ('vd1', 'vd2', 'vd3'):
                v_norms_store[key] = result.get('v_norms', [])
            if cname == 'o11':
                o11_diag[(mname, img_idx)] = dict(
                    sigma_trajectory=result.get('sigma_trajectory', []),
                    p_succ_trajectory=result.get('p_succ_trajectory', []),
                    n_successes=result.get('n_successes', 0),
                )

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
                f'{cname:<5s} | {mname:<8s} | img {img_idx:3d} '
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
with open(VNORMS_FILE, 'wb') as f:
    pickle.dump(v_norms_store, f)
with open(O11_DIAG_FILE, 'wb') as f:
    pickle.dump(o11_diag, f)
total_elapsed = time.time() - t_start
log(f'Saved {RESULTS_FILE} ({len(df)} rows)')
log(f'Saved {TRAJ_FILE}, {VNORMS_FILE}, {O11_DIAG_FILE}')
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

# Plot A: final (best) L2 violin per condition
fig, axes = plt.subplots(1, len(MODEL_NAMES), figsize=(9*len(MODEL_NAMES), 5))
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
plt.suptitle('A: Final (best) L2 -- sep / VD-1/2/3 / (1+1)', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/A_final_l2.png', dpi=130, bbox_inches='tight'); plt.close()
log('Saved A_final_l2.png')

# Plot B: convergence curves
fig, axes = plt.subplots(1, len(MODEL_NAMES), figsize=(9*len(MODEL_NAMES), 5))
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

# Plot D: improvement ratio / generations / bs query cost, all 5 conditions
fig, axes = plt.subplots(1, 3, figsize=(20, 5))
metrics = [('improvement_ratio', 'Improvement ratio (higher=better)'),
           ('n_generations', 'n_generations'),
           ('bs_queries_actual', 'Binary-search queries used')]
for ax, (metric, ylabel) in zip(axes, metrics):
    x = np.arange(len(MODEL_NAMES))
    w = 0.8 / len(COND_ORDER)
    for i, cname in enumerate(COND_ORDER):
        vals = [ok[(ok.model==m)&(ok.condition==cname)][metric].median() for m in MODEL_NAMES]
        offset = (i - (len(COND_ORDER)-1)/2) * w
        bars = ax.bar(x + offset, vals, w, label=cname, color=COND_COLORS[cname], alpha=0.85)
        for xi_, v in zip(x + offset, vals):
            ax.text(xi_, v, f'{v:.3f}' if metric=='improvement_ratio' else f'{v:.0f}',
                    ha='center', va='bottom', fontsize=7, rotation=90)
    ax.set_xticks(x); ax.set_xticklabels(MODEL_NAMES)
    ax.set_ylabel(ylabel); ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)
plt.suptitle('D: Summary -- improvement ratio, generations, binary-search cost', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/D_summary.png', dpi=130, bbox_inches='tight'); plt.close()
log('Saved D_summary.png')

# Plot F: median best_l2 and IR vs k (sep=0, vd1=1, vd2=2, vd3=3), (1+1) as reference line
fig, axes = plt.subplots(1, len(MODEL_NAMES), figsize=(7*len(MODEL_NAMES), 5))
axes = np.atleast_1d(axes)
K_CONDS = ['sep', 'vd1', 'vd2', 'vd3']
for ax, mname in zip(axes, MODEL_NAMES):
    sub = ok[ok.model == mname]
    ks  = [VK_RANK[c] for c in K_CONDS]
    med_best = [sub[sub.condition==c]['best_l2'].median() for c in K_CONDS]
    med_ir   = [sub[sub.condition==c]['improvement_ratio'].median() for c in K_CONDS]
    ax2 = ax.twinx()
    l1, = ax.plot(ks, med_best, 'o-', color='#1565C0', label='median best_l2')
    l2, = ax2.plot(ks, med_ir, 's--', color='#E53935', label='median IR')
    o11_best = sub[sub.condition=='o11']['best_l2'].median()
    o11_ir   = sub[sub.condition=='o11']['improvement_ratio'].median()
    ax.axhline(o11_best, color='#1565C0', ls=':', alpha=0.5)
    ax2.axhline(o11_ir, color='#E53935', ls=':', alpha=0.5)
    ax.text(ks[-1], o11_best, '  (1+1) best_l2', color='#1565C0', fontsize=8, va='bottom')
    ax2.text(ks[-1], o11_ir, '  (1+1) IR', color='#E53935', fontsize=8, va='top')
    ax.set_xlabel('VkD rank k'); ax.set_xticks(ks)
    ax.set_ylabel('median best_l2', color='#1565C0')
    ax2.set_ylabel('median improvement_ratio', color='#E53935')
    ax.set_title(mname); ax.grid(alpha=0.3)
    ax.legend(handles=[l1, l2], fontsize=9)
plt.suptitle('F: Effect of VkD rank k on final L2 / improvement ratio', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/F_k_sweep.png', dpi=130, bbox_inches='tight'); plt.close()
log('Saved F_k_sweep.png')

# Plot G: median V column norms over generations, per k (vd1/2/3), per model
fig, axes = plt.subplots(1, len(MODEL_NAMES), figsize=(7*len(MODEL_NAMES), 5))
axes = np.atleast_1d(axes)
for ax, mname in zip(axes, MODEL_NAMES):
    for cname in ('vd1', 'vd2', 'vd3'):
        k = VK_RANK[cname]
        runs = [v_norms_store[(cname, mname, i)] for i in range(N_IMG)
                if (cname, mname, i) in v_norms_store and v_norms_store[(cname, mname, i)]]
        if not runs:
            continue
        max_len = max(len(r) for r in runs)
        # column-0 norm (the dominant VkD direction), padded with NaN
        col0 = np.full((len(runs), max_len), np.nan)
        for i, r in enumerate(runs):
            col0[i, :len(r)] = [g[0] for g in r]
        med = np.nanmedian(col0, axis=0)
        ax.plot(np.arange(1, max_len+1), med, color=COND_COLORS[cname],
                lw=2, label=f'{cname} (k={k}), col 0')
    ax.set_xlabel('Generation'); ax.set_ylabel('median ||V[:,0]|| across images')
    ax.set_title(mname); ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.suptitle('G: VkD dominant-direction norm over generations', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/G_v_norms.png', dpi=130, bbox_inches='tight'); plt.close()
log('Saved G_v_norms.png')

# Plot H: (1+1) sigma and p_succ trajectories over generations, per model
fig, axes = plt.subplots(2, len(MODEL_NAMES), figsize=(7*len(MODEL_NAMES), 8))
axes = np.atleast_2d(axes)
for col, mname in enumerate(MODEL_NAMES):
    runs = [o11_diag[(mname, i)] for i in range(N_IMG) if (mname, i) in o11_diag]
    for row, (field, ylabel) in enumerate(
            [('sigma_trajectory', 'sigma'), ('p_succ_trajectory', 'p_succ')]):
        seqs = [r[field] for r in runs if r[field]]
        if not seqs:
            continue
        max_len = max(len(s) for s in seqs)
        arr = np.full((len(seqs), max_len), np.nan)
        for i, s in enumerate(seqs):
            arr[i, :len(s)] = s
        med = np.nanmedian(arr, axis=0)
        ax = axes[row, col]
        ax.plot(np.arange(1, max_len+1), med, color=COND_COLORS['o11'], lw=2)
        if field == 'p_succ_trajectory':
            ax.axhline(2.0/11.0, color='k', ls='--', lw=1, label='p_target=2/11')
            ax.legend(fontsize=8)
        ax.set_xlabel('Generation'); ax.set_ylabel(f'median {ylabel}')
        ax.set_title(f'{mname}: {ylabel}'); ax.grid(alpha=0.3)
plt.suptitle('H: (1+1)-CMA-ES sigma and p_succ adaptation', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/H_o11_diag.png', dpi=130, bbox_inches='tight'); plt.close()
log('Saved H_o11_diag.png')

log('\nAll done.')
log_f.close()
