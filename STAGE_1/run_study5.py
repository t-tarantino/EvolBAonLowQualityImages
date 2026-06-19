#!/usr/bin/env python3
"""
run_study5.py — Study 5: adaptive binary-search early-stop (bs_adaptive).

Background: `binary_search` bisects the segment [x_orig, x_adv], spending
one oracle query per step. After enough steps, `mid = 0.5*(lo+hi)` rounds
(in float32) to exactly `lo` or `hi` -- the bracket can't shrink further, so
by the loop invariant (`lo` correctly classified, `hi` adversarial) querying
`mid` is *guaranteed* to return the same label as `lo`/`hi`. That step is a
provable no-op.

`binary_search_adaptive(n_steps=bs_cap)` checks for this BEFORE querying and
breaks early -- lossless (returns the exact same `hi` as the fixed-step
version), but typically uses fewer queries. Crucially, the segment length
`L = ||x_adv - x_orig||` shrinks over the course of an attack, so the
saturation point shrinks too: early generations may have headroom for *more*
than 15 useful steps (up to the float32 ceiling, ~23-26), while late
generations may saturate in *fewer* than 15 -- meaning a fixed bs_steps=15
both under-uses precision early and wastes queries late.

Two conditions, both using the Study-3/4 shared config
(xi_step_scale=0.5, tau=1, lam_override=14, cmu_scale=1.0):
  bs15_fixed     binary_search(n_steps=15)         (current default)
  bs26_adaptive  binary_search_adaptive(n_steps=26) (lossless early-stop)

Plus an offline validation pass: sample (x_adv, x_orig, y_true) pairs
collected during the bs26_adaptive runs, and for each pair confirm
binary_search(26) and binary_search_adaptive(26) return identical `hi`
(never breaks), recording how many queries were actually spent (vs 26).

Usage:
    python run_study5.py           # full run (N=100, Q=2000)
    python run_study5.py --mock    # quick check (N=10, Q=300)
"""
import os, sys, time, warnings, pickle, argparse, random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from evolba_tuned   import evolba_tuned
from evolba_baseline import binary_search, binary_search_adaptive

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--mock', action='store_true')
args   = parser.parse_args()

MOCK     = args.mock
MAX_Q    = 300  if MOCK else 2000
N_IMG    = 10   if MOCK else 100
SNAP_QS  = [100, 200, 300] if MOCK else [250, 500, 750, 1000, 1500, 2000]
TAG      = 'MOCK' if MOCK else 'FULL'
VAL_N    = 30   if MOCK else 300   # number of (x_adv,x_orig,y_true) pairs to validate

OUTPUT_DIR   = f'outputs/study5_{TAG.lower()}'
RESULTS_FILE = f'{OUTPUT_DIR}/results.parquet'
TRAJ_FILE    = f'{OUTPUT_DIR}/trajectories.pkl'
VAL_FILE     = f'{OUTPUT_DIR}/validation.parquet'
LOG_FILE     = f'{OUTPUT_DIR}/run.log'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Conditions ────────────────────────────────────────────────────────────────
# Shared "settled" hyperparameters from Study 3/4 (single config for both models).
HPARAMS = dict(xi_step_scale=0.5, tau=1, lam_override=14, cmu_scale=1.0)

CONDITIONS = {
    'bs15_fixed':    dict(bs_steps=15, bs_adaptive=False, **HPARAMS),
    'bs26_adaptive': dict(bs_adaptive=True, bs_cap=26, **HPARAMS),
}
COND_ORDER  = ['bs15_fixed', 'bs26_adaptive']
COND_COLORS = {'bs15_fixed': '#888888', 'bs26_adaptive': '#4CAF50'}
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

# ── Main loop ─────────────────────────────────────────────────────────────────
rows, all_traj = [], {}
bs_pairs = []   # (x_adv, x_orig, y_true, model) collected during bs26_adaptive runs
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
            seed   = img_idx * 100 + COND_ORDER.index(cname) * 10

            collect = [] if cname == 'bs26_adaptive' else None
            t0 = time.time()
            result = evolba_tuned(oracle, x_orig, y_true,
                                  max_queries=MAX_Q, seed=seed,
                                  collect_bs_pairs=collect, **cparams)
            elapsed = time.time() - t0
            run_times.append(elapsed)
            run_idx += 1

            # tag freshly-collected pairs with the model they came from
            if collect:
                bs_pairs.extend((x_adv, x_orig_p, y_true_p, mname)
                                for x_adv, x_orig_p, y_true_p in collect)

            traj    = result['trajectory']
            il2     = result.get('init_l2', float('nan'))
            best_l2 = result.get('best_l2', float('nan'))
            n_gen   = result.get('n_generations', 0)
            bs_calls = result.get('bs_calls', 0)
            bs_q     = result.get('bs_queries_actual', 0)
            cond_l2s.append(best_l2)
            key = (cname, mname, img_idx)
            all_traj[key] = traj

            row = dict(condition=cname, model=mname, image_idx=img_idx,
                       true_class=int(y_true), success=result['success'],
                       queries=result['queries'], init_l2=il2, best_l2=best_l2,
                       n_generations=n_gen,
                       bs_calls=bs_calls, bs_queries_actual=bs_q,
                       bs_queries_per_call=(bs_q/bs_calls if bs_calls else float('nan')),
                       bs_saved_vs_26=(bs_calls*26 - bs_q))
            for sq in SNAP_QS:
                row[f'l2_at_{sq}'] = l2_at_q(traj, sq, il2)
            row['improvement_ratio'] = (
                (il2 - best_l2) / il2 if (il2 > 0 and not np.isnan(il2)) else float('nan'))
            rows.append(row)

            done_pct = run_idx / total_runs * 100
            remaining_s = (np.mean(run_times) * (total_runs - run_idx)) if run_times else 0
            eta = time.strftime('%H:%M:%S', time.localtime(time.time() + remaining_s))
            log(f'[{run_idx:4d}/{total_runs}  {done_pct:5.1f}%]  '
                f'{cname:<14s} | {mname:<8s} | img {img_idx:3d} '
                f'(class {y_true}) | init_l2={il2:.3f} best_l2={best_l2:.3f} '
                f'q={result["queries"]:4d} gen={n_gen:3d} '
                f'bs_calls={bs_calls:4d} bs_q/call={row["bs_queries_per_call"]:.2f} '
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
    n              = ('best_l2', 'count'),
    median_init    = ('init_l2', 'median'),
    median_best    = ('best_l2', 'median'),
    median_IR      = ('improvement_ratio', 'median'),
    median_gen     = ('n_generations', 'median'),
    median_bs_q_per_call = ('bs_queries_per_call', 'median'),
    median_bs_calls= ('bs_calls', 'median'),
).round(4)
summary = summary.reindex(
    pd.MultiIndex.from_product([COND_ORDER, MODEL_NAMES], names=['condition','model'])
)
log('\n' + summary.to_string())

# ── Offline validation: binary_search(26) vs binary_search_adaptive(26) ────────
log(f'\n=== VALIDATION: collected {len(bs_pairs)} (x_adv,x_orig,y_true,model) pairs '
    f'from bs26_adaptive runs ===')

sample = random.sample(bs_pairs, min(VAL_N, len(bs_pairs))) if bs_pairs else []
val_rows = []
for x_adv, x_orig_p, y_true_p, mname in sample:
    oracle = MODELS[mname]

    cnt_fixed = [0]
    def q_fixed(img):
        cnt_fixed[0] += 1
        return oracle(img)
    hi_fixed = binary_search(q_fixed, x_adv, x_orig_p, y_true_p, n_steps=26)

    cnt_adapt = [0]
    def q_adapt(img):
        cnt_adapt[0] += 1
        return oracle(img)
    hi_adapt, n_actual = binary_search_adaptive(q_adapt, x_adv, x_orig_p, y_true_p, n_steps=26)

    match = bool(np.array_equal(hi_fixed, hi_adapt))
    L = float(np.linalg.norm((x_adv.astype(np.float64) - x_orig_p.astype(np.float64)).flatten()))
    val_rows.append(dict(model=mname, L=L, n_fixed=cnt_fixed[0], n_actual=n_actual,
                         saved=26 - n_actual, match=match))

val_df = pd.DataFrame(val_rows)
val_df.to_parquet(VAL_FILE, index=False)

if len(val_df):
    match_rate = val_df['match'].mean()
    log(f'Validated {len(val_df)} pairs: match_rate={match_rate*100:.2f}% '
        f'(hi_adaptive == hi_fixed26 in all cases means it never breaks)')
    n_mismatch = (~val_df['match']).sum()
    if n_mismatch:
        log(f'!!! {n_mismatch} MISMATCHES found -- inspect validation.parquet !!!')
    log(f'Median queries actually spent (cap=26): {val_df["n_actual"].median():.1f}  '
        f'(median saved vs 26: {val_df["saved"].median():.1f})')
    # bin by L (log-spaced) to show how savings vary with segment length
    val_df['L_bin'] = pd.cut(val_df['L'], bins=np.geomspace(
        max(val_df['L'].min(), 1e-4), val_df['L'].max() + 1e-9, 6))
    log('\nSaved (vs cap=26) by segment length L:')
    log(val_df.groupby('L_bin', observed=True)['saved'].agg(['count','median']).to_string())
else:
    log('No bs_pairs collected -- skipping validation.')

# ── Plots ─────────────────────────────────────────────────────────────────────
log('Generating plots...')

# Plot A: final L2 violin per condition
fig, axes = plt.subplots(1, len(MODEL_NAMES), figsize=(7*len(MODEL_NAMES), 5))
axes = np.atleast_1d(axes)
for ax, mname in zip(axes, MODEL_NAMES):
    sub = ok[ok.model == mname]
    data = [sub[sub.condition==c]['best_l2'].dropna().values for c in COND_ORDER]
    parts = ax.violinplot([d if len(d) else [np.nan] for d in data],
                          positions=range(len(COND_ORDER)),
                          showmedians=True, showextrema=False)
    for pc, c in zip(parts['bodies'], COND_ORDER):
        pc.set_facecolor(COND_COLORS[c]); pc.set_alpha(0.6)
    parts['cmedians'].set_color('black'); parts['cmedians'].set_linewidth(2)
    for i, d in enumerate(data):
        med = np.median(d) if len(d) else np.nan
        if not np.isnan(med):
            ax.text(i, med+0.05, f'{med:.3f}', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(range(len(COND_ORDER)))
    ax.set_xticklabels(COND_ORDER, fontsize=10)
    ax.set_ylabel('Final L2 (lower = better)')
    ax.set_title(mname); ax.grid(axis='y', alpha=0.3)
plt.suptitle('A: Final L2 -- bs15_fixed vs bs26_adaptive', fontsize=12)
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
    ax.set_xlabel('Oracle queries'); ax.set_ylabel('Median L2')
    ax.set_title(mname); ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.suptitle('B: Convergence (median L2)', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/B_convergence.png', dpi=130, bbox_inches='tight'); plt.close()
log('Saved B_convergence.png')

# Plot C: queries-per-binary-search-call distribution
fig, axes = plt.subplots(1, len(MODEL_NAMES), figsize=(7*len(MODEL_NAMES), 5))
axes = np.atleast_1d(axes)
for ax, mname in zip(axes, MODEL_NAMES):
    sub = ok[ok.model == mname]
    data = [sub[sub.condition==c]['bs_queries_per_call'].dropna().values for c in COND_ORDER]
    parts = ax.violinplot([d if len(d) else [np.nan] for d in data],
                          positions=range(len(COND_ORDER)),
                          showmedians=True, showextrema=False)
    for pc, c in zip(parts['bodies'], COND_ORDER):
        pc.set_facecolor(COND_COLORS[c]); pc.set_alpha(0.6)
    parts['cmedians'].set_color('black'); parts['cmedians'].set_linewidth(2)
    ax.axhline(26, color='red', ls='--', lw=1, label='cap=26')
    for i, d in enumerate(data):
        med = np.median(d) if len(d) else np.nan
        if not np.isnan(med):
            ax.text(i, med+0.3, f'{med:.2f}', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(range(len(COND_ORDER)))
    ax.set_xticklabels(COND_ORDER, fontsize=10)
    ax.set_ylabel('Queries per binary_search call')
    ax.set_title(mname); ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)
plt.suptitle('C: Per-call binary-search query cost', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/C_bs_queries_per_call.png', dpi=130, bbox_inches='tight'); plt.close()
log('Saved C_bs_queries_per_call.png')

# Plot D: validation -- queries actually spent (cap=26) vs segment length L
if len(val_df):
    fig, ax = plt.subplots(figsize=(7, 5))
    for mname, color in zip(MODEL_NAMES, ['#2196F3', '#F44336']):
        sub = val_df[val_df.model == mname]
        ax.scatter(sub['L'], sub['n_actual'], s=12, alpha=0.5, color=color, label=mname)
    ax.axhline(26, color='gray', ls='--', lw=1, label='cap=26')
    ax.axhline(15, color='black', ls=':', lw=1, label='old fixed bs_steps=15')
    ax.set_xscale('log')
    ax.set_xlabel('Segment length L = ||x_adv - x_orig||  (log scale)')
    ax.set_ylabel('Queries actually spent (binary_search_adaptive, cap=26)')
    ax.set_title('D: Adaptive binary-search cost vs segment length')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/D_validation_savings.png', dpi=130, bbox_inches='tight'); plt.close()
    log('Saved D_validation_savings.png')

# Plot E: improvement ratio bar
fig, axes = plt.subplots(1, len(MODEL_NAMES), figsize=(7*len(MODEL_NAMES), 5))
axes = np.atleast_1d(axes)
for ax, mname in zip(axes, MODEL_NAMES):
    sub = ok[ok.model == mname]
    meds = [sub[sub.condition==c]['improvement_ratio'].median() for c in COND_ORDER]
    bars = ax.bar(range(len(COND_ORDER)), meds,
                  color=[COND_COLORS[c] for c in COND_ORDER], alpha=0.85)
    for bar, mval in zip(bars, meds):
        if not np.isnan(mval):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                    f'{mval:.3f}', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(range(len(COND_ORDER)))
    ax.set_xticklabels(COND_ORDER, fontsize=10)
    ax.set_ylim(0,1); ax.set_ylabel('Median improvement ratio')
    ax.set_title(mname); ax.grid(axis='y', alpha=0.3)
plt.suptitle('E: Improvement ratio -- bs15_fixed vs bs26_adaptive', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/E_improvement_ratio.png', dpi=130, bbox_inches='tight'); plt.close()
log('Saved E_improvement_ratio.png')

log('\nAll done.')
log_f.close()
