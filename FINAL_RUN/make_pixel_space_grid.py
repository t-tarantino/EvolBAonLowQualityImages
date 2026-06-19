#!/usr/bin/env python3
"""
make_pixel_space_grid.py -- generates a FINAL_RUN-style snapshot grid for
Part 1's pixel-space attack (best validated configuration: xi_step_scale=0.5,
tau=0, lambda=14, adaptive binary search, Q=2000), on the SAME 5 images and
checkpoint scheme as FINAL_RUN/make_final_plots.py's Figures 2/3, so the two
are directly comparable.

No saved image checkpoints exist anywhere in STAGE_0/STAGE_1 for the
pixel-space studies (only scalar L2 trajectories), so this reruns the attack
from scratch on just 5 images x 2 models -- a few minutes, not a full study.

Usage: cd report && python make_pixel_space_grid.py
"""
import os, sys, time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torchvision

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from evolba_baseline import (
    uniform_random_init, binary_search_adaptive, objective,
    sep_cmaes_weights, mean_shift_direction, update_diagonal_covariance,
)

# ── Config: Part 1's best-validated pixel-space configuration ────────────────
LAM           = 14
TAU           = 0
XI_STEP_SCALE = 0.5
BS_CAP        = 26
MAX_Q         = 2000
GRID_IMAGES   = [55, 222, 388, 166, 0]   # same indices as FINAL_RUN Figures 2/3
SNAP_FRACS    = [0.25, 0.5, 0.75, 1.0]
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                    'dog', 'frog', 'horse', 'ship', 'truck']

OUT_DIR = os.path.join(os.path.dirname(__file__), 'figures')


def select_images(oracles, n_img, seed=0):
    ds = torchvision.datasets.CIFAR10(root='/tmp/cifar10', train=False, download=True)
    per_class = max(1, n_img // 10)
    images, labels_list = [], []
    counts = [0] * 10
    for img_pil, label in ds:
        if counts[label] >= per_class:
            continue
        x = np.array(img_pil, dtype=np.float32).transpose(2, 0, 1) / 255.0
        if oracles['standard'](x) == label and oracles['robust'](x) == label:
            images.append(x); labels_list.append(label); counts[label] += 1
        if sum(counts) >= n_img:
            break
    images = np.stack(images)[:n_img]
    labels = np.array(labels_list)[:n_img]
    return images, labels


def _sep_params(n, lam):
    weights, mueff = sep_cmaes_weights(lam)
    c1  = 2.0 / ((n + 1.3) ** 2 + mueff)
    cmu = min(1.0 - c1,
              2.0 * (mueff - 2.0 + 1.0 / mueff) / ((n + 2.0) ** 2 + mueff))
    cmu = cmu * (n + 2.0) / 3.0
    return weights, c1, cmu


def run_pixel_space_attack(oracle_fn, x_orig, y_true, max_queries, seed, snapshot_fracs):
    """Faithful re-implementation of evolba_baseline's loop at Part 1's best
    validated config (xi_step_scale=0.5, tau=0, lambda=14, adaptive BS),
    instrumented to capture image snapshots at given fractions of max_queries."""
    rng         = np.random.default_rng(seed)
    shape       = x_orig.shape
    n           = x_orig.size
    x_orig_flat = x_orig.flatten().astype(np.float64)

    queries = [0]
    def query(img):
        queries[0] += 1
        return oracle_fn(img)

    # Phase 1: uniform-random init (matches Studies 6/7's default, use_fractal_init=False)
    x0 = uniform_random_init(query, shape, y_true, rng)
    if x0 is None:
        return None

    # Phase 2: binary search onto the boundary (adaptive)
    x_b, _ = binary_search_adaptive(query, x0, x_orig, y_true, n_steps=BS_CAP)
    m  = x_b.flatten().astype(np.float64)
    init_dist = float(np.linalg.norm(m - x_orig_flat))

    D   = np.ones(n, dtype=np.float64)
    lam = LAM
    mu  = lam
    weights, c1, cmu = _sep_params(n, mu)

    l2_history = [init_dist]
    snap_targets = [int(round(f * max_queries)) for f in snapshot_fracs]
    snapshots, snap_next = [], 0

    t = 1
    while queries[0] < max_queries:
        dist_to_orig = float(np.linalg.norm(m - x_orig_flat))
        xi = XI_STEP_SCALE * dist_to_orig / np.sqrt(t)

        zs = rng.standard_normal((lam, n))
        xs = np.clip(m + xi * D * zs, 0.0, 1.0)

        labels = np.empty(lam, dtype=np.int64)
        l2s    = np.empty(lam, dtype=np.float64)
        lam_eff = lam
        for k in range(lam):
            x_cand = xs[k].reshape(shape).astype(np.float32)
            labels[k] = query(x_cand)
            l2s[k]    = np.linalg.norm(xs[k] - x_orig_flat)
            if queries[0] >= max_queries:
                lam_eff = k + 1
                break
        zs, xs, labels, l2s = zs[:lam_eff], xs[:lam_eff], labels[:lam_eff], l2s[:lam_eff]

        is_adv  = labels != y_true
        fitness = np.array([objective(l2, adv) for l2, adv in zip(l2s, is_adv)])
        w_eff   = weights[:len(fitness)] / weights[:len(fitness)].sum()
        v = mean_shift_direction(zs, fitness, is_adv, w_eff)
        D = update_diagonal_covariance(D, zs, fitness, w_eff, cmu)

        m_shifted = np.clip(m + xi * v, 0.0, 1.0)
        while query(m_shifted.reshape(shape).astype(np.float32)) == y_true:
            xi /= 2.0
            m_shifted = np.clip(m + xi * v, 0.0, 1.0)
            if queries[0] >= max_queries:
                break

        m_new, _ = binary_search_adaptive(query, m_shifted.reshape(shape).astype(np.float32),
                                           x_orig, y_true, n_steps=BS_CAP)
        m_new = m_new.flatten().astype(np.float64)
        # TAU=0: no backtracking, just keep the running best
        m = m_new
        l2_history.append(float(np.linalg.norm(m - x_orig_flat)))

        while snap_next < len(snap_targets) and queries[0] >= snap_targets[snap_next]:
            snapshots.append((queries[0], min(l2_history),
                               m.reshape(shape).astype(np.float32)))
            snap_next += 1
        t += 1

    while snap_next < len(snap_targets):
        snapshots.append((queries[0], min(l2_history), m.reshape(shape).astype(np.float32)))
        snap_next += 1

    return dict(x_orig=x_orig, x_boundary=x_b, init_dist=init_dist,
                best_l2=float(min(l2_history)), queries_used=queries[0],
                snapshots=snapshots)


def main():
    from robustbench.utils import load_model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    specs = [('standard', 'Standard', 'Linf'), ('robust', 'Wang2023Better_WRN-28-10', 'Linf')]
    oracles, models = {}, {}
    for mname, arch, threat in specs:
        m = load_model(arch, dataset='cifar10', threat_model=threat).to(device).eval()
        models[mname] = m
        def _make_oracle(model=m):
            def oracle(x):
                with torch.no_grad():
                    t = torch.from_numpy(x).unsqueeze(0).float().to(device)
                    return int(model(t).argmax(dim=1).item())
            return oracle
        oracles[mname] = _make_oracle()

    images, labels = select_images(oracles, 500, seed=0)
    print(f'Selected {len(images)} images; using indices {GRID_IMAGES}')

    for mname in ['standard', 'robust']:
        rows = []
        for img_idx in GRID_IMAGES:
            x_orig, y_true = images[img_idx], int(labels[img_idx])
            t0 = time.time()
            res = None
            for retry in range(5):
                res = run_pixel_space_attack(oracles[mname], x_orig, y_true,
                                              MAX_Q, seed=img_idx * 1000 + retry,
                                              snapshot_fracs=SNAP_FRACS)
                if res is not None:
                    break
                print(f'  [{mname}] img {img_idx}: Phase 1 failed (uniform-random init), retrying...')
            if res is None:
                print(f'  [{mname}] img {img_idx}: Phase 1 failed after 5 retries, skipping')
                continue
            print(f'  [{mname}] img {img_idx} (class {CIFAR10_CLASSES[y_true]}): '
                  f'init_l2={res["init_dist"]:.3f} best_l2={res["best_l2"]:.3f} '
                  f'({time.time()-t0:.1f}s)')
            rows.append((img_idx, y_true, res))

        # ── Plot grid: original | phase1+2 boundary | p3_25 | p3_50 | p3_75 | p3_100
        col_titles = ['Original', 'Phase 1+2\nboundary', 'Phase 3\n25%',
                      'Phase 3\n50%', 'Phase 3\n75%', 'Phase 3\n100%']
        fig, axes = plt.subplots(len(rows), len(col_titles),
                                  figsize=(2.2 * len(col_titles), 2.6 * len(rows)))
        for row_i, (img_idx, y_true, res) in enumerate(rows):
            imgs = [res['x_orig'], res['x_boundary']] + [s[2] for s in res['snapshots']]
            l2s  = [0.0, res['init_dist']] + [s[1] for s in res['snapshots']]
            for col_i, (img, l2v) in enumerate(zip(imgs, l2s)):
                ax = axes[row_i, col_i]
                ax.imshow(np.clip(img, 0, 1).transpose(1, 2, 0))
                ax.set_xticks([]); ax.set_yticks([])
                if col_i > 0:
                    ax.set_title(f'L2={l2v:.2f}', fontsize=8)
                if col_i == 0:
                    ax.set_ylabel(f'img {img_idx}\n{CIFAR10_CLASSES[y_true]}', fontsize=9)
                if row_i == 0:
                    ax.set_xlabel('')
        for col_i, title in enumerate(col_titles):
            axes[0, col_i].annotate(title, xy=(0.5, 1.25), xycoords='axes fraction',
                                     ha='center', fontsize=10, fontweight='bold')
        fig.suptitle(f'Pixel-space Sep-CMA-ES, best validated config '
                     f'($\\xi$=0.5, $\\tau$=0, $\\lambda$=14), {mname} model', y=1.04)
        fig.tight_layout()
        fname = f'{OUT_DIR}/pixel_space_snapshot_grid_{mname}.png'
        fig.savefig(fname, dpi=150, bbox_inches='tight')
        print(f'Saved {fname}')


if __name__ == '__main__':
    main()
