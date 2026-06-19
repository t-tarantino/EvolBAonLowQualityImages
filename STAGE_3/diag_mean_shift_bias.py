#!/usr/bin/env python3
"""
diag_mean_shift_bias.py — Stage 3 diagnostic: does v point away from x_orig?

Background
-----------
evolba_baseline.mean_shift_direction() (Eq. 9-10) negates the sampled
direction `z` of every non-adversarial offspring and adds it to `v`. Because
non-adversarial offspring are, almost by definition, the ones whose `z` had a
meaningful component TOWARD x_orig (that's how they crossed the boundary),
"-z" injects a component AWAY from x_orig into `v`. The hypothesis is that
this is the structural reason the xi-shrink while-loop (Alg. 1 lines 12-14)
fires so often (Stage 0: ~75% of every query).

This script tests that hypothesis by running 3 variants of the mean-shift
direction, end-to-end (on-policy — each variant drives its own trajectory),
and logging, every generation:
  - v_dot_u : cos-similarity between v and u=(x_orig-m)/||x_orig-m||
              (negative => v points away from x_orig)
  - n_shrink: how many times xi was halved in the xi-shrink while-loop
  - n_backtrack, frac_adv, dist_to_orig, xi

Variants
--------
  current : evolba_baseline.mean_shift_direction (validated, UNCHANGED)
  eq10    : the paper's Eq. 10 taken literally — reversed index pairing for
            the non-adversarial group (w_{l+1}<->worst non-adv ... w_mu<->best
            non-adv), vs. current's natural (non-reversed) pairing.
  drop    : non-adversarial offspring contribute NOTHING to v (only
            adversarial offspring, renormalised weights) — the "don't go
            there" alternative discussed in Stage 3 brainstorming.

Usage:
    python diag_mean_shift_bias.py            # N=20, Q=2000, standard model
    python diag_mean_shift_bias.py --mock     # N=3,  Q=200
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
    mean_shift_direction, TAU,
)


# ── Alternative mean-shift-direction variants ───────────────────────────────

def mean_shift_eq10(zs, fitness, is_adv, weights):
    """
    Literal Eq. 10: v = sum_{i<=l} w_i z_{i:mu} - sum_{j=l+1}^{mu} w_j z_{(mu-j+l+1):mu}

    For the non-adversarial group, as j goes l+1..mu (weight w_j decreasing),
    the paired z-rank (mu-j+l+1) goes mu..l+1 (decreasing) -- i.e. REVERSED
    relative to `current`'s natural (non-reversed) pairing:
      w_{l+1} (largest non-adv weight)  <-> z rank mu   (worst non-adv)
      w_mu    (smallest non-adv weight) <-> z rank l+1  (best non-adv)
    """
    order = np.argsort(fitness)
    is_adv_ord = is_adv[order]
    l  = int(is_adv_ord.sum())
    mu = len(order)

    v = np.zeros(zs.shape[1])
    if l > 0:
        v += (weights[:l, None] * zs[order[:l]]).sum(axis=0)
    if mu > l:
        non_adv_z = zs[order[l:][::-1]]   # z_mu, z_{mu-1}, ..., z_{l+1}
        non_adv_w = weights[l:mu]         # w_{l+1}, ..., w_mu (decreasing)
        v -= (non_adv_w[:, None] * non_adv_z).sum(axis=0)
    return v / (np.linalg.norm(v) + 1e-12)


def mean_shift_drop(zs, fitness, is_adv, weights):
    """
    "Don't go there": non-adversarial offspring contribute nothing to v.
    Only adversarial offspring are used, with their weights renormalised to
    sum to 1. If NO offspring are adversarial this generation, return a zero
    vector (a no-op move for this generation -- m stays on the boundary).
    """
    order = np.argsort(fitness)
    l = int(is_adv[order].sum())
    if l == 0:
        return np.zeros(zs.shape[1])
    w_adv = weights[:l]
    w_adv = w_adv / w_adv.sum()
    v = (w_adv[:, None] * zs[order[:l]]).sum(axis=0)
    return v / (np.linalg.norm(v) + 1e-12)


VARIANTS = {
    'current': mean_shift_direction,
    'eq10':    mean_shift_eq10,
    'drop':    mean_shift_drop,
}
VARIANT_COLORS = {'current': '#888888', 'eq10': '#2196F3', 'drop': '#4CAF50'}


# ── Instrumented Algorithm-1 main loop (mirrors evolba_baseline, + logging) ──

def run_instrumented(oracle_fn, x_orig, y_true, mean_shift_fn, max_queries, seed):
    rng = np.random.default_rng(seed)
    shape = x_orig.shape
    n = x_orig.size
    x_orig_flat = x_orig.flatten().astype(np.float64)

    queries = [0]
    def query(img):
        queries[0] += 1
        return oracle_fn(img)

    x0 = uniform_random_init(query, shape, y_true, rng)
    if x0 is None:
        return []
    x_tilde0 = binary_search(query, x0, x_orig, y_true)

    m   = x_tilde0.flatten().astype(np.float64)
    D   = np.ones(n, dtype=np.float64)
    lam = 4 + int(3 * np.log(n))
    mu  = lam
    weights, mueff = sep_cmaes_weights(mu)

    c1  = 2.0 / ((n + 1.3) ** 2 + mueff)
    cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((n + 2.0) ** 2 + mueff))
    cmu = cmu * (n + 2.0) / 3.0

    gens = []
    t = 1
    while queries[0] < max_queries:
        dist_to_orig = float(np.linalg.norm(m - x_orig_flat))
        if dist_to_orig < 1e-12:
            break
        xi = dist_to_orig / np.sqrt(t)

        zs = rng.standard_normal((lam, n))
        xs = np.clip(m + xi * D * zs, 0.0, 1.0)

        labels = np.empty(lam, dtype=np.int64)
        l2s    = np.empty(lam, dtype=np.float64)
        for k in range(lam):
            x_cand    = xs[k].reshape(shape).astype(np.float32)
            labels[k] = query(x_cand)
            l2s[k]    = np.linalg.norm(xs[k] - x_orig_flat)
            if queries[0] >= max_queries:
                lam_eff = k + 1
                zs, xs, labels, l2s = zs[:lam_eff], xs[:lam_eff], labels[:lam_eff], l2s[:lam_eff]
                break

        is_adv  = labels != y_true
        fitness = np.array([objective(l2, adv) for l2, adv in zip(l2s, is_adv)])

        w_eff = weights[:len(fitness)] / weights[:len(fitness)].sum()
        v = mean_shift_fn(zs, fitness, is_adv, w_eff)
        D = update_diagonal_covariance(D, zs, fitness, w_eff, cmu)

        # ── diagnostic: does v point toward (+) or away from (-) x_orig? ────
        u = x_orig_flat - m
        u_norm = np.linalg.norm(u)
        v_dot_u = float(np.dot(v, u) / (u_norm + 1e-12)) if u_norm > 1e-12 else 0.0

        # ── xi-shrink while-loop (Alg. 1 lines 12-14) ───────────────────────
        n_shrink = 0
        m_shifted = np.clip(m + xi * v, 0.0, 1.0)
        while query(m_shifted.reshape(shape).astype(np.float32)) == y_true:
            xi /= 2.0
            n_shrink += 1
            m_shifted = np.clip(m + xi * v, 0.0, 1.0)
            if queries[0] >= max_queries:
                break

        m_new = binary_search(query, m_shifted.reshape(shape).astype(np.float32),
                              x_orig, y_true)
        m_new = m_new.flatten().astype(np.float64)

        # ── backtracking while-loop (Alg. 1 lines 19-22) ────────────────────
        n_backtrack = 0
        while (np.linalg.norm(m_new - x_orig_flat) > dist_to_orig
               and n_backtrack < TAU and queries[0] < max_queries):
            xi /= 2.0
            cand  = np.clip(m + xi * v, 0.0, 1.0).reshape(shape).astype(np.float32)
            m_new = binary_search(query, cand, x_orig, y_true).flatten().astype(np.float64)
            n_backtrack += 1

        gens.append(dict(
            gen=t, queries=queries[0], dist_to_orig=dist_to_orig, xi=xi,
            v_dot_u=v_dot_u, frac_adv=float(is_adv.mean()),
            n_shrink=n_shrink, n_backtrack=n_backtrack,
            l2_new=float(np.linalg.norm(m_new - x_orig_flat)),
        ))

        m = m_new
        t += 1

    return gens


def main():
    # ── CLI ──────────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser()
    parser.add_argument('--mock', action='store_true', help='quick check (N=3, Q=200)')
    args = parser.parse_args()

    MOCK    = args.mock
    N_IMG   = 3    if MOCK else 20
    MAX_Q   = 200  if MOCK else 2000
    TAG     = 'mock' if MOCK else f'full_q{MAX_Q}'

    OUT = f'outputs/diag_mean_shift_{TAG}'
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

    # ── Run ──────────────────────────────────────────────────────────────────
    rows = []
    t0 = time.time()
    for img_idx in range(N_IMG):
        x_orig = images[img_idx]
        y_true = int(labels[img_idx])
        seed = img_idx * 100   # SAME seed across variants -> same init point, paired

        for vname, vfn in VARIANTS.items():
            gens = run_instrumented(oracle, x_orig, y_true, vfn, MAX_Q, seed)
            for g in gens:
                g['variant'] = vname
                g['image_idx'] = img_idx
                rows.append(g)
            print(f'  img {img_idx:2d}  variant={vname:8s}  gens={len(gens):3d}')

    df = pd.DataFrame(rows)
    df.to_parquet(f'{OUT}/results.parquet', index=False)
    print(f'\nTotal time: {time.time()-t0:.1f}s  |  {len(df)} generation-records saved to {OUT}/results.parquet')

    # ── Summary table ───────────────────────────────────────────────────────
    print('\n=== Summary by variant ===')
    summary = df.groupby('variant').agg(
        n_gens        = ('v_dot_u', 'count'),
        mean_v_dot_u  = ('v_dot_u', 'mean'),
        median_v_dot_u= ('v_dot_u', 'median'),
        frac_negative = ('v_dot_u', lambda s: float((s < 0).mean())),
        mean_n_shrink = ('n_shrink', 'mean'),
        mean_n_backtrack = ('n_backtrack', 'mean'),
    ).round(4)
    summary = summary.reindex(['current', 'eq10', 'drop'])
    print(summary.to_string())
    summary.to_csv(f'{OUT}/summary.csv')

    # ── Plot: histogram of v.u per variant ────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True, sharey=True)
    for ax, vname in zip(axes, ['current', 'eq10', 'drop']):
        sub = df[df.variant == vname]['v_dot_u']
        ax.hist(sub, bins=40, range=(-1, 1), color=VARIANT_COLORS[vname], alpha=0.8)
        ax.axvline(0, color='k', lw=1, ls='--')
        ax.set_title(f'{vname}\nmean={sub.mean():.3f}, P(v.u<0)={float((sub<0).mean()):.2f}')
        ax.set_xlabel('v . u   (- = away from x_orig, + = toward x_orig)')
    axes[0].set_ylabel('count (generations)')
    plt.suptitle('Distribution of v.u across all generations / images, per variant')
    plt.tight_layout()
    plt.savefig(f'{OUT}/v_dot_u_hist.png', dpi=130, bbox_inches='tight')
    plt.close()
    print(f'Saved {OUT}/v_dot_u_hist.png')

    print('\nDone.')


if __name__ == '__main__':
    main()
