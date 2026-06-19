#!/usr/bin/env python3
"""
exp_full_cmaes_subspace.py — Stage 3: full CMA-ES variants in k=14 corruption subspace.

Phase 1: 11 corruptions (updated zoo — sharpen/saturation/gamma added,
         low_freq_rand removed).

Phase 3 direction basis (k ≤ 14):
  - Up to 11 corruption directions:
      boundary direction  if Phase 1 found an adversarial example for that type
      fixed-severity dir  (from DIRECTION_ZOO) otherwise
  - 3 fixed DCT band vectors:
      dct_low  (fy=1,  fx=0)  — smooth gradient
      dct_mid  (fy=8,  fx=8)  — mid frequency
      dct_high (fy=16, fx=16) — near Nyquist

Arms (3 new, reference lines from previous experiment):
  E_sep_k14       sep-CMA-ES, k≤14, lam=auto   — isolates effect of more directions
  E_full_k14      full CMA-ES, k≤14, lam=auto   — tests full covariance at fair lam
  E_full_k14_lam28 full CMA-ES, k≤14, lam=28   — tests larger population

References (from exp_corruption_subspace_q1500_n200, not re-run):
  standard: C_full=1.880, E_corr_sep_k6=1.906, E_rand=2.133
  robust:   C_full=4.981, E_corr_sep_k6=4.696, E_rand=5.161

Usage:
    python exp_full_cmaes_subspace.py            # full run (~1.75h)
    python exp_full_cmaes_subspace.py --mock     # N=8, Q=200
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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'STAGE_1'))

from evolba_baseline import (
    objective, uniform_random_init, binary_search,
    sep_cmaes_weights, mean_shift_direction, TAU, BS_STEPS,
)
from phase1_zoo import INIT_ZOO, DIRECTION_ZOO
from attacks.utils.subspace import corruption_basis
from exp_subspace_perf_visual import run_subspace   # for E_sep_k14
from exp_corruption_subspace import run_pixel_phase3, save_visual

MODEL_SPECS = [
    ('standard', 'Standard',                 'Linf'),
    ('robust',   'Wang2023Better_WRN-28-10', 'Linf'),
]

PREV_REF = {
    'standard': {'C_full': 1.880, 'E_corr_sep_k6': 1.906, 'E_rand': 2.133},
    'robust':   {'C_full': 4.981, 'E_corr_sep_k6': 4.696, 'E_rand': 5.161},
}


# ── DCT band direction vectors ────────────────────────────────────────────────

def dct_band_vector(shape_chw, fy, fx):
    """Single DCT-II basis vector for frequency (fy, fx), all channels combined."""
    C, H, W = shape_chw
    n = C * H * W
    ys = np.arange(H, dtype=np.float64)
    xs = np.arange(W, dtype=np.float64)
    v2d = (np.cos(np.pi * (2 * ys[:, None] + 1) * fy / (2 * H)) *
           np.cos(np.pi * (2 * xs[None, :] + 1) * fx / (2 * W)))
    v = np.zeros(n, dtype=np.float64)
    for c in range(C):
        v[c * H * W:(c + 1) * H * W] = v2d.flatten()
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-12 else v


# ── Phase 1: all corruptions + collect ALL direction vectors ──────────────────

def run_full_phase1_v2(oracle_fn, x_orig, y_true, seed):
    """
    Phase 1 with 11 corruptions (updated INIT_ZOO).

    For every corruption type:
      - If Phase 1 finds an adversarial example: use boundary direction
        d_j = x_boundary_j - x_orig  (refined, on the decision boundary)
      - Otherwise: use fixed-severity direction from DIRECTION_ZOO
        d_j = direction_fn(x_orig) - x_orig  (no oracle queries)

    Also appends 3 fixed DCT band vectors.

    Returns:
        best_init       (C,H,W) float32 — best adversarial pre-BS image
        best_boundary   (C,H,W) float32 — best boundary point
        best_l2         float
        phase1_queries  int
        winning_name    str
        per_corr        dict name -> l2 or None
        dir_vecs        dict name -> (n,) float64  direction vectors
    """
    rng         = np.random.default_rng(seed)
    x_orig_flat = x_orig.flatten().astype(np.float64)
    queries     = [0]

    def query(img):
        queries[0] += 1
        return oracle_fn(img)

    best_l2   = float('inf')
    best_bnd  = None
    best_init = None
    best_name = None
    per_corr  = {}
    dir_vecs  = {}

    for name, fn in INIT_ZOO.items():
        x_adv = fn(query, x_orig, y_true, rng)
        if x_adv is None:
            per_corr[name] = None
            # Fallback: fixed-severity direction (no oracle queries)
            if name in DIRECTION_ZOO:
                x_corrupt = DIRECTION_ZOO[name](x_orig)
                d = x_corrupt.flatten().astype(np.float64) - x_orig_flat
                if np.linalg.norm(d) > 1e-8:
                    dir_vecs[name] = d
        else:
            x_bnd = binary_search(query, x_adv, x_orig, y_true)
            l2    = float(np.linalg.norm(
                x_bnd.flatten().astype(np.float64) - x_orig_flat))
            per_corr[name] = l2
            dir_vecs[name] = x_bnd.flatten().astype(np.float64) - x_orig_flat
            if l2 < best_l2:
                best_l2   = l2
                best_bnd  = x_bnd
                best_init = x_adv
                best_name = name

    # Add 3 fixed DCT band vectors
    for band_name, fy, fx in [('dct_low', 1, 0), ('dct_mid', 8, 8), ('dct_high', 16, 16)]:
        dir_vecs[band_name] = dct_band_vector(x_orig.shape, fy, fx)

    if best_bnd is None:
        return None, None, None, queries[0], None, per_corr, dir_vecs
    return best_init, best_bnd, best_l2, queries[0], best_name, per_corr, dir_vecs


# ── Full rank-μ CMA-ES in k-dim subspace ──────────────────────────────────────

def run_full_cmaes_subspace(oracle_fn, x_orig, y_true, basis, max_queries, seed,
                             lam_override=None, x_b_override=None,
                             snapshot_fracs=None):
    """
    Full rank-μ CMA-ES in the k-dimensional subspace defined by `basis`.

    Algorithm 1 boundary-following structure (xi schedule, BS per generation,
    xi-shrink, tau=3 backtracking). Full k×k covariance matrix C replaces
    diagonal D; no rank-1 / evolution path (consistent with EvolBA xi schedule).

    Sampling:   xi * ys  where ys = (L @ zs.T).T,  L = chol(C)
    Mean shift: v = mean_shift_direction(ys, ...)   — in theta-space
    Cov update: rank-μ on ys:  C = (1-c1-cmu)*C + cmu * Σ wᵢ outer(ysᵢ, ysᵢ)
    """
    rng         = np.random.default_rng(seed)
    shape       = x_orig.shape
    k           = basis.shape[0]
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
        x_b = binary_search(query, x0, x_orig, y_true).flatten().astype(np.float64)

    init_dist  = float(np.linalg.norm(x_b - x_orig_flat))
    theta_orig = basis @ (x_orig_flat - x_b)

    def to_pixel(theta):
        return np.clip(x_b + basis.T @ theta, 0.0, 1.0)

    theta_m = np.zeros(k, dtype=np.float64)
    C       = np.eye(k,  dtype=np.float64)

    lam = lam_override if lam_override is not None else 4 + int(3 * np.log(k))
    mu  = lam
    weights, mueff = sep_cmaes_weights(mu)

    c1  = 2.0 / ((k + 1.3) ** 2 + mueff)
    cmu = min(1.0 - c1,
              2.0 * (mueff - 2.0 + 1.0 / mueff) / ((k + 2.0) ** 2 + mueff))
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

    l2_history      = [init_dist]
    queries_history = [0]
    snapshots       = None
    snap_targets    = []
    snap_next       = 0
    if snapshot_fracs is not None:
        snapshots    = []
        snap_targets = [int(round(f * max_queries)) for f in snapshot_fracs]
        if snap_targets[0] <= 0:
            snapshots.append((0, init_dist,
                              to_pixel(theta_m).reshape(shape).astype(np.float32)))
            snap_next = 1

    t = 1
    while queries[0] < max_queries:
        m_pixel      = to_pixel(theta_m)
        dist_to_orig = float(np.linalg.norm(m_pixel - x_orig_flat))
        xi           = dist_to_orig / np.sqrt(t)

        try:
            L = np.linalg.cholesky(C + 1e-8 * np.eye(k))
        except np.linalg.LinAlgError:
            C = np.eye(k)
            L = np.eye(k)

        zs = rng.standard_normal((lam, k))
        ys = (L @ zs.T).T
        theta_cand = theta_m + xi * ys

        labels = np.empty(lam, dtype=np.int64)
        l2s    = np.empty(lam, dtype=np.float64)
        for i in range(lam):
            x_cand    = to_pixel(theta_cand[i]).reshape(shape).astype(np.float32)
            labels[i] = query(x_cand)
            l2s[i]    = np.linalg.norm(
                x_cand.flatten().astype(np.float64) - x_orig_flat)
            if queries[0] >= max_queries:
                lam_eff = i + 1
                ys, zs      = ys[:lam_eff], zs[:lam_eff]
                theta_cand  = theta_cand[:lam_eff]
                labels, l2s = labels[:lam_eff], l2s[:lam_eff]
                break

        is_adv  = labels != y_true
        fitness = np.array([objective(l2, adv) for l2, adv in zip(l2s, is_adv)])

        w_eff = weights[:len(fitness)] / weights[:len(fitness)].sum()
        v     = mean_shift_direction(ys, fitness, is_adv, w_eff)

        # Rank-μ update on ys
        order  = np.argsort(fitness)
        ys_ord = ys[order]
        rankmu = np.zeros((k, k), dtype=np.float64)
        for i in range(len(w_eff)):
            rankmu += w_eff[i] * np.outer(ys_ord[i], ys_ord[i])
        C = (1.0 - c1 - cmu) * C + cmu * rankmu
        C = 0.5 * (C + C.T)

        theta_shifted = theta_m + xi * v
        while query(to_pixel(theta_shifted).reshape(shape).astype(np.float32)) == y_true:
            xi           /= 2.0
            theta_shifted = theta_m + xi * v
            if queries[0] >= max_queries:
                break

        theta_new = theta_binary_search(theta_shifted)

        n_backtrack = 0
        new_dist    = float(np.linalg.norm(to_pixel(theta_new) - x_orig_flat))
        while (new_dist > dist_to_orig and n_backtrack < TAU
               and queries[0] < max_queries):
            xi           /= 2.0
            theta_shifted = theta_m + xi * v
            theta_new     = theta_binary_search(theta_shifted)
            new_dist      = float(np.linalg.norm(to_pixel(theta_new) - x_orig_flat))
            n_backtrack  += 1

        theta_m = theta_new
        l2_history.append(new_dist)
        queries_history.append(queries[0])

        if snapshots is not None:
            while (snap_next < len(snap_targets)
                   and queries[0] >= snap_targets[snap_next]):
                snapshots.append((queries[0], new_dist,
                                  to_pixel(theta_m).reshape(shape).astype(np.float32)))
                snap_next += 1
        t += 1

    if snapshots is not None:
        while snap_next < len(snap_targets):
            snapshots.append((queries[0], l2_history[-1],
                              to_pixel(theta_m).reshape(shape).astype(np.float32)))
            snap_next += 1

    return dict(
        init_dist=init_dist, final_dist=l2_history[-1],
        best_l2=float(min(l2_history)),
        n_gens=len(l2_history) - 1, queries_used=queries[0],
        l2_history=l2_history, queries_history=queries_history,
        snapshots=snapshots,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mock', action='store_true',
                        help='quick check (N=8, Q=200)')
    args = parser.parse_args()

    MOCK    = args.mock
    N_IMG   = 8   if MOCK else 200
    Q_TOTAL = 200 if MOCK else 1500
    VISUAL_IMG_INDICES = [0, 1] if MOCK else [0, 40, 80, 120, 160]
    TAG     = 'mock' if MOCK else f'q{Q_TOTAL}_n{N_IMG}'

    OUT = os.path.join(os.path.dirname(__file__),
                       'outputs', f'exp_full_cmaes_subspace_{TAG}')
    os.makedirs(OUT, exist_ok=True)

    from robustbench.utils import load_model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    oracles = {}
    for mname, arch, threat in MODEL_SPECS:
        m = load_model(arch, dataset='cifar10', threat_model=threat).to(device).eval()
        def _make_oracle(model=m):
            def oracle(x_chw):
                with torch.no_grad():
                    t = torch.from_numpy(x_chw[None]).to(device)
                    return int(model(t).argmax(1).item())
            return oracle
        oracles[mname] = _make_oracle()
        print(f'Loaded {mname}: {arch}')

    import torchvision
    ds = torchvision.datasets.CIFAR10(root='/tmp/cifar10', train=False, download=True)
    per_class = max(1, N_IMG // 10)
    images, labels = [], []
    counts = [0] * 10
    for img_pil, label in ds:
        if counts[label] >= per_class:
            continue
        x = np.array(img_pil, dtype=np.float32).transpose(2, 0, 1) / 255.0
        if oracles['standard'](x) == label and oracles['robust'](x) == label:
            images.append(x); labels.append(label); counts[label] += 1
        if sum(counts) >= N_IMG:
            break
    images = np.stack(images)[:N_IMG]
    labels = np.array(labels)[:N_IMG]
    print(f'Images: {len(images)}  |  label dist: {counts}')

    rows     = []
    vis_data = {}
    t0       = time.time()

    for img_idx in range(len(images)):
        x_orig    = images[img_idx]
        y_true    = int(labels[img_idx])
        seed_base = img_idx * 1000
        want_vis  = img_idx in VISUAL_IMG_INDICES

        for mname, _, _ in MODEL_SPECS:
            oracle_fn = oracles[mname]

            (x_init, x_bnd, pl2, pq,
             win_name, per_corr, dir_vecs) = run_full_phase1_v2(
                oracle_fn, x_orig, y_true, seed=seed_base)

            if x_bnd is None:
                continue

            dvecs_list = list(dir_vecs.values())
            k_total    = len(dvecs_list)
            try:
                basis_k14 = corruption_basis(dvecs_list)
            except ValueError:
                continue

            q_phase3   = max(10, Q_TOTAL - pq)
            snap_fracs = ([1/3, 2/3, 1.0] if want_vis else None)
            snap_qs    = ([int(q_phase3 * f) for f in [1/3, 2/3, 1.0]]
                          if want_vis else None)

            # ── E_sep_k14: sep-CMA-ES, k≤14, lam=auto ────────────────────────
            res_sep = run_subspace(
                oracle_fn, x_orig, y_true, basis_k14, q_phase3,
                seed=seed_base + 1, x_b_override=x_bnd,
                snapshot_fracs=snap_fracs)

            # ── E_full_k14: full CMA-ES, k≤14, lam=auto ──────────────────────
            res_full = run_full_cmaes_subspace(
                oracle_fn, x_orig, y_true, basis_k14, q_phase3,
                seed=seed_base + 2, lam_override=None,
                x_b_override=x_bnd, snapshot_fracs=snap_fracs)

            # ── E_full_k14_lam28: full CMA-ES, k≤14, lam=28 ──────────────────
            res_full_lam28 = run_full_cmaes_subspace(
                oracle_fn, x_orig, y_true, basis_k14, q_phase3,
                seed=seed_base + 3, lam_override=28,
                x_b_override=x_bnd, snapshot_fracs=snap_fracs)

            arm_results = [
                ('E_sep_k14',        res_sep),
                ('E_full_k14',       res_full),
                ('E_full_k14_lam28', res_full_lam28),
            ]
            for arm_name, res in arm_results:
                if res is None:
                    continue
                ir_p3 = (pl2 - res['best_l2']) / pl2 if pl2 > 0 else 0.0
                row = dict(
                    model=mname, arm=arm_name,
                    image_idx=img_idx, y_true=y_true,
                    k_total=k_total,
                    phase1_l2=pl2, phase1_queries=pq,
                    winning_corruption=win_name,
                    best_l2=res['best_l2'],
                    final_l2=res.get('final_l2', res.get('final_dist',
                                                          res['best_l2'])),
                    IR_phase3=ir_p3,
                    n_gens=res['n_gens'],
                    queries_phase3=(res['queries_history'][-1]
                                    if 'queries_history' in res
                                    else res.get('queries_used', 0)),
                    **{f'p1_{n}': v for n, v in per_corr.items()},
                    l2_history=res['l2_history'],
                    queries_history=res['queries_history'],
                )
                rows.append(row)

                if want_vis:
                    snaps = res.get('snapshots') or []
                    x_best_vis = snaps[-1][2] if snaps else x_bnd
                    if len(snaps) < 3:
                        snaps = snaps + [(0, pl2, x_bnd)] * (3 - len(snaps))
                    vis_data[(img_idx, mname, arm_name)] = dict(
                        model_label=mname, arm_label=arm_name,
                        phase1_l2=pl2,
                        x_init=x_init, x_boundary=x_bnd,
                        snapshots=snaps[:3],
                        x_best=x_best_vis, best_l2=res['best_l2'],
                    )

        elapsed = time.time() - t0
        if (img_idx + 1) % 10 == 0 or img_idx < 3:
            print(f'  img {img_idx+1:3d}/{len(images)}  ({elapsed:.0f}s elapsed)')

    print(f'\nTotal time: {time.time()-t0:.1f}s  |  {len(rows)} rows')

    df = pd.DataFrame(rows)
    df.to_parquet(f'{OUT}/results.parquet', index=False)
    print(f'Saved {OUT}/results.parquet')
    df.drop(columns=['l2_history', 'queries_history']).to_csv(
        f'{OUT}/summary.csv', index=False)

    print('\n=== Summary by (model, arm) ===')
    plain   = df.drop(columns=['l2_history', 'queries_history'])
    summary = plain.groupby(['model', 'arm']).agg(
        n              = ('best_l2', 'count'),
        mean_phase1_l2 = ('phase1_l2', 'mean'),
        mean_best_l2   = ('best_l2', 'mean'),
        median_best_l2 = ('best_l2', 'median'),
        mean_IR_phase3 = ('IR_phase3', 'mean'),
        mean_n_gens    = ('n_gens', 'mean'),
        mean_k_total   = ('k_total', 'mean'),
    ).round(4)
    print(summary.to_string())

    arm_order  = ['E_sep_k14', 'E_full_k14', 'E_full_k14_lam28']
    arm_labels = {
        'E_sep_k14':        'E_sep_k14\nsep-CMA\nlam=auto',
        'E_full_k14':       'E_full_k14\nfull-CMA\nlam=auto',
        'E_full_k14_lam28': 'E_full_k14\nfull-CMA\nlam=28',
    }
    model_colors = {'standard': '#1976D2', 'robust': '#D32F2F'}

    # ── Plot 1: best_l2 by arm ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, (mname, _, _) in zip(axes, MODEL_SPECS):
        sub  = plain[plain.model == mname]
        data = [sub[sub.arm == a]['best_l2'].values for a in arm_order]
        bp   = ax.boxplot(data, labels=[arm_labels[a] for a in arm_order],
                          patch_artist=True,
                          medianprops=dict(color='white', lw=2))
        for patch in bp['boxes']:
            patch.set_facecolor(model_colors[mname]); patch.set_alpha(0.7)
        ref = PREV_REF[mname]
        ax.axhline(ref['E_corr_sep_k6'], color='green',  ls='--', lw=1.2,
                   label=f'E_corr_sep_k6 ({ref["E_corr_sep_k6"]:.3f})')
        ax.axhline(ref['C_full'],        color='orange', ls='--', lw=1.2,
                   label=f'C_full ({ref["C_full"]:.3f})')
        ax.set_title(mname); ax.set_xlabel('arm')
        ax.set_ylabel('best L2' if mname == 'standard' else '')
        ax.legend(fontsize=7); ax.grid(axis='y', alpha=0.3)
    plt.suptitle('Best L2 by arm  (dashed = previous experiment references)')
    plt.tight_layout()
    plt.savefig(f'{OUT}/best_l2_by_arm.png', dpi=130, bbox_inches='tight')
    plt.close()
    print(f'Saved {OUT}/best_l2_by_arm.png')

    # ── Plot 2: convergence curves ────────────────────────────────────────────
    arm_styles = {
        'E_sep_k14':        ('--', '#888888'),
        'E_full_k14':       ('-',  '#1565C0'),
        'E_full_k14_lam28': ('-',  '#00897B'),
    }
    q_max  = plain['queries_phase3'].max()
    q_grid = np.linspace(0, q_max, 60).astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, (mname, _, _) in zip(axes, MODEL_SPECS):
        for a, (ls, color) in arm_styles.items():
            sub = df[(df.model == mname) & (df.arm == a)]
            curves = []
            for _, row in sub.iterrows():
                if len(row.queries_history) < 2:
                    continue
                curves.append(np.interp(q_grid,
                                        row.queries_history, row.l2_history))
            if not curves:
                continue
            mat  = np.stack(curves)
            mean = mat.mean(0); std = mat.std(0)
            ax.plot(q_grid, mean, label=a, color=color, ls=ls, lw=1.6)
            ax.fill_between(q_grid, mean - std, mean + std,
                            color=color, alpha=0.12)
        ref = PREV_REF[mname]
        ax.axhline(ref['E_corr_sep_k6'], color='green',  ls=':',  lw=1.0,
                   label=f'E_corr_sep_k6 ({ref["E_corr_sep_k6"]:.3f})')
        ax.axhline(ref['C_full'],        color='orange', ls=':',  lw=1.0,
                   label=f'C_full ({ref["C_full"]:.3f})')
        ax.set_title(mname); ax.set_xlabel('Phase 3 queries')
        if mname == 'standard':
            ax.set_ylabel('L2 to x_orig')
        ax.legend(fontsize=7); ax.grid(alpha=0.25)
    plt.suptitle('Convergence by arm  (Phase 3 queries, mean ± std)')
    plt.tight_layout()
    plt.savefig(f'{OUT}/convergence_by_arm.png', dpi=130, bbox_inches='tight')
    plt.close()
    print(f'Saved {OUT}/convergence_by_arm.png')

    # ── Visual progressions ───────────────────────────────────────────────────
    for img_idx in VISUAL_IMG_INDICES:
        vis_rows = []
        for mname, _, _ in MODEL_SPECS:
            for arm_name in arm_order:
                key = (img_idx, mname, arm_name)
                if key in vis_data:
                    vis_rows.append(vis_data[key])
        if not vis_rows:
            continue
        out_path = f'{OUT}/visual_img{img_idx}.png'
        save_visual(img_idx, images[img_idx], vis_rows, out_path)
        print(f'Saved {out_path}')

    print('\nDone.')


if __name__ == '__main__':
    main()
