#!/usr/bin/env python3
"""
exp_corruption_subspace.py — Stage 3: corruption-space ES vs pixel-space ES.

Hypothesis: sep-CMA-ES in a subspace spanned by corruption directions
outperforms pixel-space sep-CMA-ES when Phase 1 starts from a corruption
boundary, because the corruption directions ARE aligned with the adversarial
manifold while random pixel-space Gaussian noise is not.

Phase 1: ALL corruption types from INIT_ZOO (10 total).  Each corruption that
finds an adversarial example contributes:
  - a candidate boundary point (for best-L2 selection)
  - a direction vector  d_j = x_boundary_j.flat - x_orig.flat
    (for building the corruption basis)

Phase 3 variants:
  C_full  — Phase 1 (all 10) + pixel-space sep-CMA-ES, lam=28, xi=1.0, tau=3
  E_rand  — Phase 1 (all 10) + random   k-dim subspace ES   (control)
  E_corr  — Phase 1 (all 10) + corruption k-dim subspace ES  (main hypothesis)

Reference (from exp_corruption_phase1 outputs, no rerun):
  C_ref   — Phase 1 (jpeg+blur+fractal) + pixel-space ES     → prev best_l2

Models : Standard  +  Wang2023Better_WRN-28-10 (robust)
N_IMG  : 200 jointly-classified images (20/class)
Q_TOTAL: 1500 per run  (Phase 1 queries count against the budget)

Visual snapshots per run (5 images × 3 arms × 2 models):
  7-frame grid identical to exp_corruption_phase1:
  orig | pre-BS init | boundary | Phase3 Q/3 | 2Q/3 | end | best

Usage:
    python exp_corruption_subspace.py            # full run (~1.5h)
    python exp_corruption_subspace.py --mock     # N=8, Q=200
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
    sep_cmaes_weights, update_diagonal_covariance,
    mean_shift_direction, TAU, BS_STEPS,
)
from phase1_zoo import INIT_ZOO          # all 10 corruption types
from attacks.utils.subspace import corruption_basis, random_basis
from exp_subspace_perf_visual import run_subspace  # reuse (x_b_override patched)


MODEL_SPECS = [
    ('standard', 'Standard',                 'Linf'),
    ('robust',   'Wang2023Better_WRN-28-10', 'Linf'),
]

# Phase 3 config shared by all three arms
LAM_PIXEL  = 28    # pixel-space  (matches exp_corruption_phase1 arm C)
LAM_SUB    = None  # subspace: computed from k inside run_subspace
XI_SCALE   = 1.0   # xi_step_scale for pixel-space arm  (baseline config)
TAU_ARM    = 3


# ── Phase 1: run ALL corruptions, collect direction vectors ───────────────────

def run_full_phase1(oracle_fn, x_orig, y_true, seed):
    """
    Runs every corruption in INIT_ZOO.  Returns:
        best_init       (C,H,W) float32 — adversarial image before BS (winner)
        best_boundary   (C,H,W) float32 — boundary point (winner)
        best_l2         float
        phase1_queries  int
        winning_name    str
        per_corr        dict  name -> l2 or None
        direction_vecs  dict  name -> (n,) float64   d_j = x_bnd_j - x_orig
    """
    rng          = np.random.default_rng(seed)
    x_orig_flat  = x_orig.flatten().astype(np.float64)
    queries      = [0]

    def query(img):
        queries[0] += 1
        return oracle_fn(img)

    best_l2     = float('inf')
    best_bnd    = None
    best_init   = None
    best_name   = None
    per_corr    = {}
    dir_vecs    = {}

    for name, fn in INIT_ZOO.items():
        x_adv = fn(query, x_orig, y_true, rng)
        if x_adv is None:
            per_corr[name] = None
            continue
        x_bnd = binary_search(query, x_adv, x_orig, y_true)
        l2    = float(np.linalg.norm(x_bnd.flatten().astype(np.float64)
                                     - x_orig_flat))
        per_corr[name]  = l2
        dir_vecs[name]  = x_bnd.flatten().astype(np.float64) - x_orig_flat
        if l2 < best_l2:
            best_l2   = l2
            best_bnd  = x_bnd
            best_init = x_adv
            best_name = name

    if best_bnd is None:
        return None, None, None, queries[0], None, per_corr, dir_vecs
    return best_init, best_bnd, best_l2, queries[0], best_name, per_corr, dir_vecs


# ── Phase 3: pixel-space sep-CMA-ES (reused from exp_corruption_phase1) ───────

def run_pixel_phase3(oracle_fn, x_orig, y_true, x_boundary, max_queries, seed,
                     snapshot_qs=None):
    """
    Pixel-space sep-CMA-ES starting from x_boundary (lam=28, xi=1.0, tau=3).
    Identical to exp_corruption_phase1's run_phase3 with baseline config.
    """
    rng   = np.random.default_rng(seed)
    shape = x_orig.shape
    n     = x_orig.size
    x_orig_flat = x_orig.flatten().astype(np.float64)

    queries = [0]
    def query(img):
        queries[0] += 1
        return oracle_fn(img)

    m  = x_boundary.flatten().astype(np.float64)
    D  = np.ones(n, dtype=np.float64)
    lam = LAM_PIXEL
    weights, mueff = sep_cmaes_weights(lam)

    c1  = 2.0 / ((n + 1.3) ** 2 + mueff)
    cmu = min(1.0 - c1,
              2.0 * (mueff - 2.0 + 1.0 / mueff) / ((n + 2.0) ** 2 + mueff))
    cmu = cmu * (n + 2.0) / 3.0

    init_dist = float(np.linalg.norm(m - x_orig_flat))
    best_l2   = init_dist
    x_best    = x_boundary.copy()

    l2_history      = [init_dist]
    queries_history = [0]
    snapshots  = [] if snapshot_qs is not None else None
    snap_next  = 0

    t = 1
    while queries[0] < max_queries:
        dist_to_orig = float(np.linalg.norm(m - x_orig_flat))
        xi      = dist_to_orig / np.sqrt(t)
        xi_step = xi * XI_SCALE

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
                zs, xs, labels, l2s = (zs[:lam_eff], xs[:lam_eff],
                                        labels[:lam_eff], l2s[:lam_eff])
                break

        is_adv  = labels != y_true
        fitness = np.array([objective(l2, adv) for l2, adv in zip(l2s, is_adv)])

        w_eff = weights[:len(fitness)] / weights[:len(fitness)].sum()
        v = mean_shift_direction(zs, fitness, is_adv, w_eff)
        D = update_diagonal_covariance(D, zs, fitness, w_eff, cmu)

        m_shifted = np.clip(m + xi_step * v, 0.0, 1.0)
        while query(m_shifted.reshape(shape).astype(np.float32)) == y_true:
            xi_step  /= 2.0
            m_shifted = np.clip(m + xi_step * v, 0.0, 1.0)
            if queries[0] >= max_queries:
                break

        m_new      = binary_search(query,
                                   m_shifted.reshape(shape).astype(np.float32),
                                   x_orig, y_true)
        m_new_flat = m_new.flatten().astype(np.float64)
        new_dist   = float(np.linalg.norm(m_new_flat - x_orig_flat))

        n_back = 0
        while new_dist > dist_to_orig and n_back < TAU_ARM and queries[0] < max_queries:
            xi_step  /= 2.0
            cand      = np.clip(m + xi_step * v, 0.0, 1.0).reshape(shape).astype(np.float32)
            m_new     = binary_search(query, cand, x_orig, y_true)
            m_new_flat = m_new.flatten().astype(np.float64)
            new_dist  = float(np.linalg.norm(m_new_flat - x_orig_flat))
            n_back   += 1

        m      = m_new_flat
        new_l2 = float(np.linalg.norm(m - x_orig_flat))
        l2_history.append(new_l2)
        queries_history.append(queries[0])

        if new_l2 < best_l2:
            best_l2 = new_l2
            x_best  = m.reshape(shape).astype(np.float32).copy()

        if snapshots is not None:
            while snap_next < len(snapshot_qs) and queries[0] >= snapshot_qs[snap_next]:
                snapshots.append((queries[0], new_l2,
                                  m.reshape(shape).astype(np.float32).copy()))
                snap_next += 1
        t += 1

    if snapshots is not None:
        while snap_next < len(snapshot_qs):
            snapshots.append((queries[0], l2_history[-1],
                              m.reshape(shape).astype(np.float32).copy()))
            snap_next += 1

    return dict(best_l2=best_l2, final_l2=l2_history[-1], x_best=x_best,
                l2_history=l2_history, queries_history=queries_history,
                n_gens=t - 1, snapshots=snapshots)


# ── Visual output (same structure as exp_corruption_phase1) ───────────────────

def save_visual(img_idx, x_orig, vis_rows, out_path):
    n_rows = len(vis_rows)
    fig, axes = plt.subplots(n_rows, 7, figsize=(15.4, 2.0 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]

    col_titles = ['orig', 'pre-BS init', 'boundary\n(Phase1 end)',
                  'Phase3 Q/3', 'Phase3 2Q/3', 'Phase3 end', 'best']
    for ci, ct in enumerate(col_titles):
        axes[0, ci].set_title(ct, fontsize=8)

    def show(ax, img, label=None):
        ax.imshow(np.clip(img.transpose(1, 2, 0), 0, 1), interpolation='nearest')
        if label:
            ax.set_xlabel(label, fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])

    for ri, row in enumerate(vis_rows):
        axes[ri, 0].set_ylabel(f"{row['model_label']}\n{row['arm_label']}",
                               fontsize=8, labelpad=2)
        show(axes[ri, 0], x_orig)
        show(axes[ri, 1], row['x_init'])
        show(axes[ri, 2], row['x_boundary'], f"L2={row['phase1_l2']:.2f}")
        for ci, (q, l2, img) in enumerate(row['snapshots'], start=3):
            show(axes[ri, ci], img, f"q={q}\nL2={l2:.2f}")
        show(axes[ri, 6], row['x_best'], f"best={row['best_l2']:.2f}")

    fig.suptitle(f'image {img_idx}', fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close()


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
                       'outputs', f'exp_corruption_subspace_{TAG}')
    os.makedirs(OUT, exist_ok=True)

    # ── Models ────────────────────────────────────────────────────────────────
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

    # ── Images: jointly classified ─────────────────────────────────────────
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

    # ── Experiment loop ───────────────────────────────────────────────────────
    rows     = []
    vis_data = {}
    t0       = time.time()

    for img_idx in range(len(images)):
        x_orig   = images[img_idx]
        y_true   = int(labels[img_idx])
        seed_base = img_idx * 1000
        want_vis  = img_idx in VISUAL_IMG_INDICES

        for mname, _, _ in MODEL_SPECS:
            oracle_fn = oracles[mname]

            # ── Phase 1: all corruptions, collect direction vectors ──────────
            (x_init, x_bnd, pl2, pq,
             win_name, per_corr, dir_vecs) = run_full_phase1(
                oracle_fn, x_orig, y_true, seed=seed_base)

            if x_bnd is None:
                continue   # all corruptions failed (extremely rare)

            phase1_l2_uniform_ref = None  # loaded from prev results if needed

            # Build bases for subspace arms
            dvecs_list = list(dir_vecs.values())   # (n,) vectors per corruption
            k_corr = len(dvecs_list)
            try:
                basis_corr = corruption_basis(dvecs_list)
            except ValueError:
                basis_corr = None

            basis_rand = random_basis(x_orig.shape, k=k_corr,
                                      seed=seed_base) if k_corr > 0 else None

            q_phase3 = max(10, Q_TOTAL - pq)
            snap_qs  = ([int(q_phase3 * f) for f in [1/3, 2/3, 1.0]]
                        if want_vis else None)

            # ── Arm C_full: pixel-space ES ───────────────────────────────────
            res_pixel = run_pixel_phase3(
                oracle_fn, x_orig, y_true, x_bnd, q_phase3,
                seed=seed_base + 1, snapshot_qs=snap_qs)

            # ── Arm E_rand: random subspace ES ───────────────────────────────
            res_rand = None
            if basis_rand is not None:
                res_rand = run_subspace(
                    oracle_fn, x_orig, y_true, basis_rand, q_phase3,
                    seed=seed_base + 2, x_b_override=x_bnd,
                    snapshot_fracs=([1/3, 2/3, 1.0] if want_vis else None))

            # ── Arm E_corr: corruption subspace ES ───────────────────────────
            res_corr = None
            if basis_corr is not None:
                res_corr = run_subspace(
                    oracle_fn, x_orig, y_true, basis_corr, q_phase3,
                    seed=seed_base + 3, x_b_override=x_bnd,
                    snapshot_fracs=([1/3, 2/3, 1.0] if want_vis else None))

            # ── Build rows ───────────────────────────────────────────────────
            arm_results = [
                ('C_full', 'pixel',       res_pixel),
                ('E_rand', 'random_sub',  res_rand),
                ('E_corr', 'corrupt_sub', res_corr),
            ]
            for arm_name, phase3_type, res in arm_results:
                if res is None:
                    continue
                ir_p3 = (pl2 - res['best_l2']) / pl2 if pl2 > 0 else 0.0
                row = dict(
                    model=mname, arm=arm_name,
                    phase3_type=phase3_type,
                    image_idx=img_idx, y_true=y_true,
                    k_corr=k_corr,
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
                    # for subspace arms x_best is the last snapshot image
                    x_best_vis = (res['x_best'] if 'x_best' in res
                                  else (snaps[-1][2] if snaps else x_bnd))
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

    # ── Save ──────────────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    df.to_parquet(f'{OUT}/results.parquet', index=False)
    print(f'Saved {OUT}/results.parquet')
    df.drop(columns=['l2_history', 'queries_history']).to_csv(
        f'{OUT}/summary.csv', index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    print('\n=== Summary by (model, arm) ===')
    plain = df.drop(columns=['l2_history', 'queries_history'])
    summary = plain.groupby(['model', 'arm']).agg(
        n              = ('best_l2', 'count'),
        mean_phase1_l2 = ('phase1_l2', 'mean'),
        mean_best_l2   = ('best_l2', 'mean'),
        median_best_l2 = ('best_l2', 'median'),
        mean_IR_phase3 = ('IR_phase3', 'mean'),
        mean_n_gens    = ('n_gens', 'mean'),
        mean_k_corr    = ('k_corr', 'mean'),
    ).round(4)
    print(summary.to_string())

    # ── Plot 1: best_l2 by arm ────────────────────────────────────────────────
    arm_order  = ['C_full', 'E_rand', 'E_corr']
    arm_labels = {'C_full': 'C_full\npixel\nspace',
                  'E_rand': 'E_rand\nrandom\nsubspace',
                  'E_corr': 'E_corr\ncorrupt\nsubspace'}
    model_colors = {'standard': '#1976D2', 'robust': '#D32F2F'}

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    for ax, (mname, _, _) in zip(axes, MODEL_SPECS):
        sub  = plain[plain.model == mname]
        data = [sub[sub.arm == a]['best_l2'].values for a in arm_order]
        bp   = ax.boxplot(data, labels=[arm_labels[a] for a in arm_order],
                          patch_artist=True,
                          medianprops=dict(color='white', lw=2))
        for patch in bp['boxes']:
            patch.set_facecolor(model_colors[mname]); patch.set_alpha(0.7)
        # Reference line from prev experiment (arm C_ref)
        ref_l2 = {'standard': 1.876, 'robust': 5.030}
        ax.axhline(ref_l2[mname], color='orange', ls='--', lw=1.2,
                   label=f'C_ref prev ({ref_l2[mname]:.3f})')
        ax.set_title(mname); ax.set_xlabel('arm')
        ax.set_ylabel('best L2' if mname == 'standard' else '')
        ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)
    plt.suptitle('Best L2 by arm  (dashed = prev pixel-space baseline)')
    plt.tight_layout()
    plt.savefig(f'{OUT}/best_l2_by_arm.png', dpi=130, bbox_inches='tight')
    plt.close()
    print(f'Saved {OUT}/best_l2_by_arm.png')

    # ── Plot 2: convergence curves ────────────────────────────────────────────
    arm_styles = {'C_full': ('-',  '#555555'),
                  'E_rand': ('--', '#888888'),
                  'E_corr': ('-',  '#0D47A1')}
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
        ax.set_title(mname); ax.set_xlabel('Phase 3 queries')
        if mname == 'standard': ax.set_ylabel('L2 to x_orig')
        ax.legend(fontsize=8); ax.grid(alpha=0.25)
    plt.suptitle('Convergence by arm  (Phase 3 queries, mean ± std)')
    plt.tight_layout()
    plt.savefig(f'{OUT}/convergence_by_arm.png', dpi=130, bbox_inches='tight')
    plt.close()
    print(f'Saved {OUT}/convergence_by_arm.png')

    # ── Plot 3: k_corr distribution ───────────────────────────────────────────
    sub_c = plain[(plain.arm == 'E_corr')]
    if len(sub_c) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        for ax, (mname, _, _) in zip(axes, MODEL_SPECS):
            vc = sub_c[sub_c.model == mname]['k_corr'].value_counts().sort_index()
            ax.bar(vc.index.astype(str), vc.values, color=model_colors[mname],
                   alpha=0.8)
            ax.set_title(mname); ax.set_xlabel('k (# corruption directions found)')
            ax.set_ylabel('# images')
        plt.suptitle('Corruption basis dimension per image')
        plt.tight_layout()
        plt.savefig(f'{OUT}/k_corr_distribution.png', dpi=130, bbox_inches='tight')
        plt.close()
        print(f'Saved {OUT}/k_corr_distribution.png')

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
