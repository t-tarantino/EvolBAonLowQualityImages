#!/usr/bin/env python3
"""
exp_subspace_random_vs_dct.py — Stage 3: DCT vs. random subspace, vs. k.

Ports evolba_baseline's Algorithm 1 (Eq. 9-10 mean_shift_direction, Sep-CMA-ES
diagonal D, Eq. 4 xi schedule, binary search) to operate on k-dim subspace
coordinates `theta` instead of the full n=3072 pixel vector.

Phase 1/2 (uniform_random_init + binary_search) are unchanged and produce a
pixel-space boundary point x_b. From there:

  theta_orig = B @ (x_orig - x_b)   -- projection of x_orig onto the subspace,
                                        FIXED for the whole run: the binary-search
                                        target in theta-space.
  residual   = (x_orig - x_b) - B.T @ theta_orig
  residual_norm = ||residual||      -- the unreachable component: even with
                                        theta_m -> theta_orig, dist_to_orig can
                                        only shrink to residual_norm, never 0.

  to_pixel(theta) = clip(x_b + B.T @ theta, 0, 1)

Each generation samples zs ~ N(0, I_k), forms theta candidates, maps to pixel
space for oracle queries, computes v (mean_shift_direction) and D in R^k, then
shrinks/backtracks/binary-searches entirely in theta-space (binary search
interpolates between theta_orig and theta_shifted).

Compares DCT vs random bases at several k, on-policy, paired seeds
(seed = image_idx * 100, identical across (basis,k) for a given image).

Usage:
    python exp_subspace_random_vs_dct.py            # N=20, Q=2000
    python exp_subspace_random_vs_dct.py --mock     # N=3,  Q=200, k=[30]
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

def run_subspace(oracle_fn, x_orig, y_true, basis, max_queries, seed):
    rng = np.random.default_rng(seed)
    shape = x_orig.shape
    n = x_orig.size
    k = basis.shape[0]
    x_orig_flat = x_orig.flatten().astype(np.float64)

    queries = [0]
    def query(img):
        queries[0] += 1
        return oracle_fn(img)

    x0 = uniform_random_init(query, shape, y_true, rng)
    if x0 is None:
        return None
    x_tilde0 = binary_search(query, x0, x_orig, y_true)
    x_b = x_tilde0.flatten().astype(np.float64)

    init_dist = float(np.linalg.norm(x_b - x_orig_flat))

    # subspace target + unreachable residual (zero-query, closed form)
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

    gens = []
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

        gens.append(dict(
            gen=t, queries=queries[0], dist_to_orig=dist_to_orig, xi=xi,
            n_shrink=n_shrink, n_backtrack=n_backtrack, l2_new=new_dist,
        ))

        theta_m = theta_new
        t += 1

    final_dist = float(np.linalg.norm(to_pixel(theta_m) - x_orig_flat))
    return dict(
        init_dist=init_dist, final_dist=final_dist, residual_norm=residual_norm,
        n_gens=len(gens), queries_used=queries[0],
        mean_n_shrink=float(np.mean([g['n_shrink'] for g in gens])) if gens else 0.0,
        mean_n_backtrack=float(np.mean([g['n_backtrack'] for g in gens])) if gens else 0.0,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mock', action='store_true', help='quick check (N=3, Q=200, k=[30])')
    args = parser.parse_args()

    MOCK  = args.mock
    N_IMG = 3    if MOCK else 20
    MAX_Q = 200  if MOCK else 2000
    K_VALUES = [30] if MOCK else [30, 60, 120, 240, 480]
    TAG   = 'mock' if MOCK else f'full_q{MAX_Q}'

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
                res = run_subspace(oracle, x_orig, y_true, basis, MAX_Q, seed)
                if res is None:
                    continue
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
        mean_residual_norm = ('residual_norm', 'mean'),
        mean_final_dist    = ('final_dist', 'mean'),
        mean_n_shrink      = ('mean_n_shrink', 'mean'),
        mean_n_backtrack   = ('mean_n_backtrack', 'mean'),
        mean_n_gens        = ('n_gens', 'mean'),
    ).round(4)
    print(summary.to_string())
    summary.to_csv(f'{OUT}/summary.csv')

    # ── Plot: IR vs k, DCT vs random ────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for bname, color in [('dct', '#2196F3'), ('random', '#888888')]:
        sub = summary.xs(bname, level='basis')
        axes[0].plot(sub.index, sub['mean_IR'], marker='o', label=bname, color=color)
        axes[1].plot(sub.index, sub['mean_n_shrink'], marker='o', label=bname, color=color)
    axes[0].set_xlabel('k (subspace dimension)')
    axes[0].set_ylabel('mean IR (improvement ratio)')
    axes[0].set_title('Improvement ratio vs k')
    axes[0].legend()
    axes[1].set_xlabel('k (subspace dimension)')
    axes[1].set_ylabel('mean n_shrink per generation')
    axes[1].set_title('xi-shrink cost vs k')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(f'{OUT}/ir_and_shrink_vs_k.png', dpi=130, bbox_inches='tight')
    plt.close()
    print(f'Saved {OUT}/ir_and_shrink_vs_k.png')

    print('\nDone.')


if __name__ == '__main__':
    main()
