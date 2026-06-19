#!/usr/bin/env python3
"""
exp_subspace_perf_visual.py — Stage 3: DCT vs random subspace, performance +
visual diagnostics, on a much larger image set.

Same k-dim subspace port of evolba_baseline's Algorithm 1 as
exp_subspace_random_vs_dct.py (theta-space CMA-ES, theta_orig binary-search
target, residual_norm floor). Additions in this script:

  - l2_history / queries_history: per-generation L2-to-x_orig and cumulative
    query count, saved for every run -> convergence curves (L2 vs queries).
  - best_l2 (= final_dist, the algorithm's reported output) and IR, as before.
  - Visual progression: for a handful of images, at the largest k, snapshot
    the reconstructed pixel image at 5 uniform query-fraction checkpoints
    (0%, 25%, 50%, 75%, 100% of the budget) for both DCT and random bases.

Usage:
    python exp_subspace_perf_visual.py            # N=150, Q=2000 (~4.5h)
    python exp_subspace_perf_visual.py --mock     # N=6,  Q=200, k=[30,480]
"""
import os, sys, time, warnings, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from evolba_baseline import (
    objective, uniform_random_init, binary_search,
    sep_cmaes_weights, update_diagonal_covariance,
    mean_shift_direction, TAU, BS_STEPS,
)
from attacks.utils.subspace import dct_basis, random_basis


# ── Instrumented Algorithm-1 main loop, in k-dim subspace coordinates ───────

def run_subspace(oracle_fn, x_orig, y_true, basis, max_queries, seed,
                  snapshot_fracs=None, x_b_override=None):
    """
    x_b_override: (C,H,W) float32 boundary point from an external Phase 1.
    When provided, uniform_random_init + binary_search are skipped entirely
    and no Phase-1 queries are charged.
    """
    rng = np.random.default_rng(seed)
    shape = x_orig.shape
    k = basis.shape[0]
    x_orig_flat = x_orig.flatten().astype(np.float64)

    queries = [0]
    def query(img):
        queries[0] += 1
        return oracle_fn(img)

    if x_b_override is not None:
        x_b = x_b_override.flatten().astype(np.float64)
    else:
        x0 = uniform_random_init(query, shape, y_true, rng)
        if x0 is None:
            return None
        x_tilde0 = binary_search(query, x0, x_orig, y_true)
        x_b = x_tilde0.flatten().astype(np.float64)

    init_dist = float(np.linalg.norm(x_b - x_orig_flat))

    delta = x_orig_flat - x_b
    theta_orig = basis @ delta
    residual = delta - basis.T @ theta_orig
    residual_norm = float(np.linalg.norm(residual))

    def to_pixel(theta):
        return np.clip(x_b + basis.T @ theta, 0.0, 1.0)

    theta_m = np.zeros(k)
    D = np.ones(k, dtype=np.float64)
    lam = 4 + int(3 * np.log(k))
    mu = lam
    weights, mueff = sep_cmaes_weights(mu)

    c1  = 2.0 / ((k + 1.3) ** 2 + mueff)
    cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((k + 2.0) ** 2 + mueff))
    cmu = cmu * (k + 2.0) / 3.0

    def theta_binary_search(theta_adv, n_steps=BS_STEPS):
        lo, hi = theta_orig.copy(), theta_adv.copy()
        for _ in range(n_steps):
            mid = 0.5 * (lo + hi)
            img = to_pixel(mid).reshape(shape).astype(np.float32)
            if query(img) != y_true:
                hi = mid
            else:
                lo = mid
        return hi

    # ── L2-history / snapshot bookkeeping ───────────────────────────────────
    l2_history = [init_dist]
    queries_history = [0]

    snapshots = None
    snap_targets = []
    snap_next = 0
    if snapshot_fracs is not None:
        snapshots = []
        snap_targets = [int(round(f * max_queries)) for f in snapshot_fracs]
        if snap_targets[0] <= 0:
            snapshots.append((0, init_dist, to_pixel(theta_m).reshape(shape).astype(np.float32)))
            snap_next = 1

    t = 1
    while queries[0] < max_queries:
        m_pixel = to_pixel(theta_m)
        dist_to_orig = float(np.linalg.norm(m_pixel - x_orig_flat))
        xi = dist_to_orig / np.sqrt(t)

        zs = rng.standard_normal((lam, k))
        theta_cand = theta_m + xi * D * zs

        labels = np.empty(lam, dtype=np.int64)
        l2s    = np.empty(lam, dtype=np.float64)
        for i in range(lam):
            x_cand    = to_pixel(theta_cand[i]).reshape(shape).astype(np.float32)
            labels[i] = query(x_cand)
            l2s[i]    = np.linalg.norm(x_cand.flatten().astype(np.float64) - x_orig_flat)
            if queries[0] >= max_queries:
                lam_eff = i + 1
                zs, theta_cand, labels, l2s = zs[:lam_eff], theta_cand[:lam_eff], labels[:lam_eff], l2s[:lam_eff]
                break

        is_adv  = labels != y_true
        fitness = np.array([objective(l2, adv) for l2, adv in zip(l2s, is_adv)])

        w_eff = weights[:len(fitness)] / weights[:len(fitness)].sum()
        v = mean_shift_direction(zs, fitness, is_adv, w_eff)
        D = update_diagonal_covariance(D, zs, fitness, w_eff, cmu)

        # ── xi-shrink while-loop ─────────────────────────────────────────────
        n_shrink = 0
        theta_shifted = theta_m + xi * v
        while query(to_pixel(theta_shifted).reshape(shape).astype(np.float32)) == y_true:
            xi /= 2.0
            n_shrink += 1
            theta_shifted = theta_m + xi * v
            if queries[0] >= max_queries:
                break

        theta_new = theta_binary_search(theta_shifted)

        # ── backtracking while-loop ──────────────────────────────────────────
        n_backtrack = 0
        new_dist = float(np.linalg.norm(to_pixel(theta_new) - x_orig_flat))
        while (new_dist > dist_to_orig and n_backtrack < TAU and queries[0] < max_queries):
            xi /= 2.0
            theta_shifted = theta_m + xi * v
            theta_new = theta_binary_search(theta_shifted)
            new_dist = float(np.linalg.norm(to_pixel(theta_new) - x_orig_flat))
            n_backtrack += 1

        theta_m = theta_new
        l2_history.append(new_dist)
        queries_history.append(queries[0])

        if snapshots is not None:
            while snap_next < len(snap_targets) and queries[0] >= snap_targets[snap_next]:
                snapshots.append((queries[0], new_dist, to_pixel(theta_m).reshape(shape).astype(np.float32)))
                snap_next += 1

        t += 1

    if snapshots is not None:
        while snap_next < len(snap_targets):
            snapshots.append((queries[0], l2_history[-1], to_pixel(theta_m).reshape(shape).astype(np.float32)))
            snap_next += 1

    final_dist = l2_history[-1]
    best_l2    = float(min(l2_history))   # running minimum, not just final
    return dict(
        init_dist=init_dist, final_dist=final_dist, best_l2=best_l2,
        residual_norm=residual_norm,
        n_gens=len(l2_history) - 1, queries_used=queries[0],
        l2_history=l2_history, queries_history=queries_history,
        snapshots=snapshots,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mock', action='store_true', help='quick check (N=6, Q=200, k=[30,480])')
    args = parser.parse_args()

    MOCK  = args.mock
    N_IMG = 6    if MOCK else 150
    MAX_Q = 200  if MOCK else 2000
    K_VALUES = [30, 480] if MOCK else [30, 60, 120, 240, 480]
    TAG   = 'mock' if MOCK else f'perf_q{MAX_Q}'

    VISUAL_K = max(K_VALUES)
    VISUAL_IMG_INDICES = [0, 1] if MOCK else [0, 30, 60, 90, 120]
    SNAPSHOT_FRACS = [0.0, 0.25, 0.5, 0.75, 1.0]

    OUT = f'outputs/exp_subspace_{TAG}'
    os.makedirs(OUT, exist_ok=True)

    # ── Model + images ───────────────────────────────────────────────────────
    from robustbench.utils import load_model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    model = load_model('Standard', dataset='cifar10', threat_model='Linf').to(device).eval()
    def oracle(x_chw):
        with torch.no_grad():
            t = torch.from_numpy(x_chw[None].astype(np.float32)).to(device)
            return int(model(t).argmax(1).item())

    import torchvision
    ds = torchvision.datasets.CIFAR10(root='/tmp/cifar10', train=False, download=True)
    per_class = max(1, N_IMG // 10)
    images, labels = [], []
    counts = [0] * 10
    for img_pil, label in ds:
        if counts[label] >= per_class:
            continue
        x = np.array(img_pil, dtype=np.float32).transpose(2, 0, 1) / 255.0
        if oracle(x) == label:
            images.append(x); labels.append(label); counts[label] += 1
        if sum(counts) >= N_IMG:
            break
    images = np.stack(images)[:N_IMG]
    labels = np.array(labels)[:N_IMG]
    print(f'Images: {len(images)}  |  label dist: {counts}')

    shape_chw = images[0].shape

    # ── Run ──────────────────────────────────────────────────────────────────
    rows = []
    all_snapshots = {}  # (basis, k, img_idx) -> list of (queries, l2, image)
    t0 = time.time()
    for k in K_VALUES:
        bases = {
            'dct':    dct_basis(shape_chw, k),
            'random': random_basis(shape_chw, k, seed=k),
        }
        for bname, basis in bases.items():
            for img_idx in range(N_IMG):
                x_orig = images[img_idx]
                y_true = int(labels[img_idx])
                seed = img_idx * 100
                want_snapshots = (k == VISUAL_K and img_idx in VISUAL_IMG_INDICES)
                res = run_subspace(oracle, x_orig, y_true, basis, MAX_Q, seed,
                                    snapshot_fracs=SNAPSHOT_FRACS if want_snapshots else None)
                if res is None:
                    continue

                snapshots = res.pop('snapshots')
                if snapshots is not None:
                    all_snapshots[(bname, k, img_idx)] = snapshots

                res['basis'] = bname
                res['k'] = k
                res['image_idx'] = img_idx
                res['IR'] = (res['init_dist'] - res['final_dist']) / res['init_dist']
                rows.append(res)
            print(f'  k={k:4d}  basis={bname:6s}  done  ({time.time()-t0:.1f}s elapsed)')

    df = pd.DataFrame(rows)
    df.to_parquet(f'{OUT}/results.parquet', index=False)
    print(f'\nTotal time: {time.time()-t0:.1f}s  |  {len(df)} run-records saved to {OUT}/results.parquet')

    # ── Summary table ───────────────────────────────────────────────────────
    print('\n=== Summary by (k, basis) ===')
    summary = df.groupby(['k', 'basis']).agg(
        n           = ('IR', 'count'),
        mean_IR     = ('IR', 'mean'),
        median_IR   = ('IR', 'median'),
        mean_best_l2 = ('best_l2', 'mean'),
        mean_residual_norm = ('residual_norm', 'mean'),
        mean_n_gens        = ('n_gens', 'mean'),
    ).round(4)
    print(summary.to_string())
    summary.to_csv(f'{OUT}/summary.csv')

    # ── Convergence plot: mean L2 vs queries, per k, DCT vs random ──────────
    q_grid = np.arange(0, MAX_Q + 1, max(1, MAX_Q // 40))
    fig, axes = plt.subplots(1, len(K_VALUES), figsize=(4 * len(K_VALUES), 4), sharey=True)
    if len(K_VALUES) == 1:
        axes = [axes]
    for ax, k in zip(axes, K_VALUES):
        for bname, color in [('dct', '#2196F3'), ('random', '#888888')]:
            sub = df[(df.k == k) & (df.basis == bname)]
            curves = np.stack([
                np.interp(q_grid, row.queries_history, row.l2_history)
                for _, row in sub.iterrows()
            ])
            mean_curve = curves.mean(axis=0)
            std_curve  = curves.std(axis=0)
            ax.plot(q_grid, mean_curve, label=bname, color=color)
            ax.fill_between(q_grid, mean_curve - std_curve, mean_curve + std_curve,
                             color=color, alpha=0.15)
        ax.set_title(f'k={k}')
        ax.set_xlabel('queries')
        if k == K_VALUES[0]:
            ax.set_ylabel('L2 to x_orig')
        ax.legend()
    plt.tight_layout()
    plt.savefig(f'{OUT}/convergence_by_k.png', dpi=130, bbox_inches='tight')
    plt.close()
    print(f'Saved {OUT}/convergence_by_k.png')

    # ── IR vs k plot ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for bname, color in [('dct', '#2196F3'), ('random', '#888888')]:
        sub = summary.xs(bname, level='basis')
        ax.plot(sub.index, sub['mean_IR'], marker='o', label=bname, color=color)
    ax.set_xlabel('k (subspace dimension)')
    ax.set_ylabel('mean IR (improvement ratio)')
    ax.set_title('Improvement ratio vs k')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{OUT}/ir_vs_k.png', dpi=130, bbox_inches='tight')
    plt.close()
    print(f'Saved {OUT}/ir_vs_k.png')

    # ── Visual progression: one figure per image, DCT row + random row ──────
    snap_q = [int(round(f * MAX_Q)) for f in SNAPSHOT_FRACS]
    for img_idx in VISUAL_IMG_INDICES:
        fig, axes = plt.subplots(2, len(SNAPSHOT_FRACS),
                                  figsize=(2.2 * len(SNAPSHOT_FRACS), 4.6))
        for row_i, bname in enumerate(['dct', 'random']):
            key = (bname, VISUAL_K, img_idx)
            if key not in all_snapshots:
                continue
            for col_i, (q, l2, img) in enumerate(all_snapshots[key]):
                ax = axes[row_i, col_i]
                ax.imshow(np.clip(img.transpose(1, 2, 0), 0, 1))
                ax.set_title(f'q={q}\nL2={l2:.2f}', fontsize=9)
                ax.set_xticks([]); ax.set_yticks([])
                if col_i == 0:
                    ax.set_ylabel(bname, fontsize=11)
        fig.suptitle(f'image {img_idx}  (k={VISUAL_K})')
        plt.tight_layout()
        plt.savefig(f'{OUT}/visual_progression_img{img_idx}.png', dpi=130, bbox_inches='tight')
        plt.close()
    print(f'Saved visual_progression_img*.png for {VISUAL_IMG_INDICES}')

    print('\nDone.')


if __name__ == '__main__':
    main()
