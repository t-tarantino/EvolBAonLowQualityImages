#!/usr/bin/env python3
"""
make_final_plots.py — generate every FINAL_RUN plot from saved results only.

Never touches the model or the oracle: everything here is a pure function of
results.parquet / snapshots.npz / snapshots_meta.csv, so re-styling a plot never
requires rerunning run_final.py.

Every figure is a single axes (no subplot grids) per project convention.

Usage:
    python make_final_plots.py --run final_q2000_n500
    python make_final_plots.py --run final_mock
"""
import os, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                    'dog', 'frog', 'horse', 'ship', 'truck']
MODEL_COLORS = {'standard': '#1976D2', 'robust': '#D32F2F'}
MODEL_ORDER  = ['standard', 'robust']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, help='output subdir under outputs/, e.g. final_q2000_n500')
    args = parser.parse_args()

    OUT = os.path.join(os.path.dirname(__file__), 'outputs', args.run)
    PLOTS = os.path.join(OUT, 'plots')
    os.makedirs(PLOTS, exist_ok=True)

    df = pd.read_parquet(f'{OUT}/results.parquet')
    print(f'Loaded {len(df)} rows from {OUT}/results.parquet')

    # ── Plot 1: mean L2 vs cumulative queries (both models, one axes) ────────
    q_grid = np.linspace(0, df.apply(lambda r: r.phase1_queries + r.queries_phase3, axis=1).max(), 200)
    fig, ax = plt.subplots(figsize=(7, 5))
    for mname in MODEL_ORDER:
        sub = df[df.model == mname]
        curves = []
        for _, row in sub.iterrows():
            qs = [0] + [row.phase1_queries + q for q in row.queries_history]
            l2 = [row.phase1_l2] + list(row.l2_history)
            curves.append(np.interp(q_grid, qs, l2, right=l2[-1]))
        arr  = np.stack(curves)
        mean, std = arr.mean(0), arr.std(0)
        ax.plot(q_grid, mean, color=MODEL_COLORS[mname], lw=2, label=mname)
        ax.fill_between(q_grid, mean - std, mean + std, color=MODEL_COLORS[mname], alpha=0.15)
    ax.set_xlabel('total queries (Phase 1 + Phase 3)')
    ax.set_ylabel('mean L2 to original')
    ax.set_title('L2 vs. query budget')
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f'{PLOTS}/l2_vs_query.png', dpi=130)
    plt.close(fig)
    print('Saved l2_vs_query.png')

    # ── Plot 2: mean L2 vs generation index (both models, one axes) ──────────
    max_gen = int(df.l2_history.apply(len).max())
    gen_grid = np.arange(max_gen)
    fig, ax = plt.subplots(figsize=(7, 5))
    for mname in MODEL_ORDER:
        sub = df[df.model == mname]
        curves = []
        for _, row in sub.iterrows():
            l2 = np.array(row.l2_history, dtype=np.float64)
            padded = np.full(max_gen, l2[-1])
            padded[:len(l2)] = l2
            curves.append(padded)
        arr  = np.stack(curves)
        mean, std = arr.mean(0), arr.std(0)
        ax.plot(gen_grid, mean, color=MODEL_COLORS[mname], lw=2, label=mname)
        ax.fill_between(gen_grid, mean - std, mean + std, color=MODEL_COLORS[mname], alpha=0.15)
    ax.set_xlabel('generation')
    ax.set_ylabel('mean L2 to original')
    ax.set_title('L2 vs. generation')
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f'{PLOTS}/l2_vs_generation.png', dpi=130)
    plt.close(fig)
    print('Saved l2_vs_generation.png')

    # ── Plot 3: best_l2 distribution (both models overlaid, one axes) ────────
    fig, ax = plt.subplots(figsize=(7, 5))
    for mname in MODEL_ORDER:
        vals = df[df.model == mname]['best_l2'].values
        ax.hist(vals, bins=30, color=MODEL_COLORS[mname], alpha=0.55, label=mname, density=False)
    ax.set_xlabel('best L2'); ax.set_ylabel('count')
    ax.set_title('Distribution of best L2 (final adversarial perturbation size)')
    ax.legend(); ax.grid(alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(f'{PLOTS}/best_l2_distribution.png', dpi=130)
    plt.close(fig)
    print('Saved best_l2_distribution.png')

    # ── Plot 4: IR_phase3 distribution (both models overlaid, one axes) ──────
    fig, ax = plt.subplots(figsize=(7, 5))
    for mname in MODEL_ORDER:
        vals = df[df.model == mname]['IR_phase3'].values
        ax.hist(vals, bins=30, color=MODEL_COLORS[mname], alpha=0.55, label=mname, density=False)
    ax.set_xlabel('IR_phase3  =  (phase1_l2 - best_l2) / phase1_l2')
    ax.set_ylabel('count')
    ax.set_title('Distribution of Phase 3 improvement ratio')
    ax.legend(); ax.grid(alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(f'{PLOTS}/IR_phase3_distribution.png', dpi=130)
    plt.close(fig)
    print('Saved IR_phase3_distribution.png')

    # ── Plot 5: query-cost breakdown (grouped bars, one axes) ─────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    cats = ['phase1_queries', 'q_offspring_eval', 'q_xi_shrink', 'q_theta_bs']
    cat_labels = ['Phase 1\n(init)', 'offspring\neval', 'xi-shrink\nloop', 'theta_bs\n(pull+backtrack)']
    x = np.arange(len(cats))
    width = 0.35
    for i, mname in enumerate(MODEL_ORDER):
        means = [df[df.model == mname][c].mean() for c in cats]
        ax.bar(x + (i - 0.5) * width, means, width=width, color=MODEL_COLORS[mname],
               alpha=0.8, label=mname)
    ax.set_xticks(x); ax.set_xticklabels(cat_labels)
    ax.set_ylabel('mean queries per image')
    ax.set_title('Query-cost breakdown')
    ax.legend(); ax.grid(alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(f'{PLOTS}/query_cost_breakdown.png', dpi=130)
    plt.close(fig)
    print('Saved query_cost_breakdown.png')

    # ── Plot 6: per-image snapshot sequence (one PNG per (image, model),  ────
    #            checkpoints of that single attack shown in order) ──────────
    # Note: 'p3_0' (Phase 3 at 0% of its budget) is pixel-identical to
    # 'phase1_boundary' (Phase 3 starts exactly where Phase 1+2 left off,
    # theta_m=0), so it is dropped here as a redundant column rather than
    # plotted twice. 'phase1_boundary' is the boundary point reached after
    # Phase 1 (corruption-based initial adversarial example) *and* Phase 2
    # (binary search onto the decision boundary) -- not Phase 1 alone.
    CHECKPOINT_ORDER = ['original', 'phase1_boundary', 'p3_25', 'p3_50', 'p3_75', 'p3_100']
    CHECKPOINT_LABELS = ['Original', 'Phase 1+2\nboundary', 'Phase 3\n25%', '50%', '75%', '100%']

    snap_meta_path = f'{OUT}/snapshots_meta.csv'
    snap_npz_path  = f'{OUT}/snapshots.npz'
    if os.path.exists(snap_meta_path) and os.path.exists(snap_npz_path):
        meta = pd.read_csv(snap_meta_path)
        arrays = np.load(snap_npz_path)
        snap_dir = os.path.join(PLOTS, 'snapshots')
        os.makedirs(snap_dir, exist_ok=True)

        n_seq = 0
        for (img_idx, mname), group in meta.groupby(['image_idx', 'model']):
            by_cp = {row.checkpoint: row for _, row in group.iterrows()}
            present = [cp for cp in CHECKPOINT_ORDER if cp in by_cp]
            if not present:
                continue

            fig, axes = plt.subplots(1, len(present), figsize=(2.3 * len(present), 2.9))
            if len(present) == 1:
                axes = [axes]
            for ax, cp in zip(axes, present):
                row = by_cp[cp]
                key = f'{img_idx}_{mname}_{cp}'
                if key not in arrays:
                    ax.axis('off'); continue
                img = arrays[key].transpose(1, 2, 0)
                ax.imshow(np.clip(img, 0, 1))
                ax.set_xticks([]); ax.set_yticks([])
                true_name = CIFAR10_CLASSES[int(row.true_label)]
                pred_name = CIFAR10_CLASSES[int(row.label)]
                status = 'correct' if row.label == row.true_label else f'-> {pred_name}'
                cp_label = CHECKPOINT_LABELS[CHECKPOINT_ORDER.index(cp)]
                ax.set_title(f'{cp_label}\n{status}\nL2={row.l2:.2f} SSIM={row.ssim:.2f}\nq={int(row.queries)}',
                             fontsize=7.5)
            fig.suptitle(f'Image {img_idx} | {mname} model | true class = {CIFAR10_CLASSES[int(group.iloc[0].true_label)]}',
                         fontsize=10)
            fig.tight_layout()
            fname = f'{snap_dir}/img{img_idx}_{mname}_sequence.png'
            fig.savefig(fname, dpi=130, bbox_inches='tight')
            plt.close(fig)
            n_seq += 1
        print(f'Saved {n_seq} snapshot-sequence plots to {snap_dir}/')

        # ── Plot 7: 5x7 grid, one figure per model, rows = selected images ────
        GRID_IMAGES = [55, 222, 388, 166, 0]
        for mname in MODEL_ORDER:
            fig, axes = plt.subplots(len(GRID_IMAGES), len(CHECKPOINT_ORDER),
                                      figsize=(2.2 * len(CHECKPOINT_ORDER), 2.6 * len(GRID_IMAGES)))
            for row_i, img_idx in enumerate(GRID_IMAGES):
                sub = meta[(meta.image_idx == img_idx) & (meta.model == mname)]
                by_cp = {row.checkpoint: row for _, row in sub.iterrows()}
                true_label = int(sub.iloc[0].true_label) if len(sub) else None
                for col_i, cp in enumerate(CHECKPOINT_ORDER):
                    ax = axes[row_i, col_i]
                    if cp not in by_cp:
                        ax.axis('off'); continue
                    row = by_cp[cp]
                    key = f'{img_idx}_{mname}_{cp}'
                    if key not in arrays:
                        ax.axis('off'); continue
                    img = arrays[key].transpose(1, 2, 0)
                    ax.imshow(np.clip(img, 0, 1))
                    ax.set_xticks([]); ax.set_yticks([])
                    pred_name = CIFAR10_CLASSES[int(row.label)]
                    status = 'correct' if row.label == row.true_label else f'-> {pred_name}'
                    if row_i == 0:
                        ax.set_title(CHECKPOINT_LABELS[col_i], fontsize=9)
                    ax.text(0.5, -0.12, f'{status}\nL2={row.l2:.2f} SSIM={row.ssim:.2f}',
                            transform=ax.transAxes, ha='center', va='top', fontsize=7)
                    if col_i == 0:
                        ax.set_ylabel(f'img {img_idx}\n{CIFAR10_CLASSES[true_label]}', fontsize=9)
            fig.suptitle(f'Snapshot grid | {mname} model', fontsize=13)
            fig.tight_layout()
            fname = f'{PLOTS}/snapshot_grid_{mname}.png'
            fig.savefig(fname, dpi=130, bbox_inches='tight')
            plt.close(fig)
            print(f'Saved snapshot_grid_{mname}.png')
    else:
        print('No snapshot data found, skipping snapshot plots.')

    print('\nAll plots written to', PLOTS)


if __name__ == '__main__':
    main()
