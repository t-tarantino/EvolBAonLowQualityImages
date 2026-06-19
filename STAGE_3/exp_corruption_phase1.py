#!/usr/bin/env python3
"""
exp_corruption_phase1.py — Stage 3: 2×2 factorial experiment.

  Phase 1:  {uniform_random_init  |  corruption_init (jpeg + blur + fractal_random)}
  Phase 3:  {baseline (lam=28, xi=1.0, tau=3)  |  tuned (lam=14, xi=0.75, tau=3)}

  Arms:
    A — uniform   + baseline
    B — uniform   + tuned
    C — corruption + baseline
    D — corruption + tuned

Models: Standard  +  Wang2023Better_WRN-28-10 (robust)
N_IMG : 200 jointly-classified images (20/class)
Q_TOTAL: 1500 per run (Phase 1 + Phase 3 combined)

Visual snapshots per run (5 chosen images × 4 arms × 2 models):
  frame 0  x_orig               — clean reference
  frame 1  x_init               — pre-binary-search adversarial (random noise or corrupted)
  frame 2  x_boundary           — Phase 1 output, start of Phase 3
  frame 3  Phase 3 snap @ Q/3
  frame 4  Phase 3 snap @ 2Q/3
  frame 5  Phase 3 snap @ Q     — end of Phase 3 budget
  frame 6  x_best               — running-minimum-L2 adversarial found during Phase 3

Usage:
    python exp_corruption_phase1.py            # full run (~3-3.5h)
    python exp_corruption_phase1.py --mock     # N=8, Q=200
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
from phase1_zoo import jpeg_init, blur_init, fractal_random_init


# ── Arm definitions ───────────────────────────────────────────────────────────
#   (arm_name, phase1_type, lam, xi_step_scale, tau)
ARMS = [
    ('A', 'uniform',    28, 1.00, 3),
    ('B', 'uniform',    14, 0.75, 3),
    ('C', 'corruption', 28, 1.00, 3),
    ('D', 'corruption', 14, 0.75, 3),
]

CORRUPTION_FNS = [
    ('jpeg',    jpeg_init),
    ('blur',    blur_init),
    ('fractal', fractal_random_init),
]

MODEL_SPECS = [
    ('standard', 'Standard',                  'Linf'),
    ('robust',   'Wang2023Better_WRN-28-10',  'Linf'),
]


# ── Phase 1: uniform ─────────────────────────────────────────────────────────

def run_uniform_phase1(oracle_fn, x_orig, y_true, seed):
    """Returns (x_init, x_boundary, phase1_l2, queries_used) or (None,None,None,q)."""
    rng = np.random.default_rng(seed)
    queries = [0]
    def query(img):
        queries[0] += 1
        return oracle_fn(img)

    x_init = uniform_random_init(query, x_orig.shape, y_true, rng)
    if x_init is None:
        return None, None, None, queries[0]

    x_bnd = binary_search(query, x_init, x_orig, y_true)
    phase1_l2 = float(np.linalg.norm(x_bnd.flatten().astype(np.float64)
                                     - x_orig.flatten().astype(np.float64)))
    return x_init, x_bnd, phase1_l2, queries[0]


# ── Phase 1: corruption ───────────────────────────────────────────────────────

def run_corruption_phase1(oracle_fn, x_orig, y_true, seed):
    """
    Tries jpeg, blur, fractal_random; picks boundary with minimum L2.
    Returns (x_init_best, x_boundary_best, phase1_l2, queries_used,
             winning_name, per_corr_l2s) or (None,None,None,q,None,{}).
    """
    rng = np.random.default_rng(seed)
    queries = [0]
    def query(img):
        queries[0] += 1
        return oracle_fn(img)

    x_orig_flat = x_orig.flatten().astype(np.float64)
    best_l2     = float('inf')
    best_bnd    = None
    best_init   = None
    best_name   = None
    per_corr    = {}

    for name, fn in CORRUPTION_FNS:
        x_adv = fn(query, x_orig, y_true, rng)
        if x_adv is None:
            per_corr[name] = None
            continue
        x_bnd = binary_search(query, x_adv, x_orig, y_true)
        l2    = float(np.linalg.norm(x_bnd.flatten().astype(np.float64) - x_orig_flat))
        per_corr[name] = l2
        if l2 < best_l2:
            best_l2   = l2
            best_bnd  = x_bnd
            best_init = x_adv
            best_name = name

    if best_bnd is None:
        return None, None, None, queries[0], None, per_corr
    return best_init, best_bnd, best_l2, queries[0], best_name, per_corr


# ── Phase 3: sep-CMA-ES ───────────────────────────────────────────────────────

def run_phase3(oracle_fn, x_orig, y_true, x_boundary, max_queries, seed,
               lam_override, xi_step_scale, tau, snapshot_qs=None):
    """
    Sep-CMA-ES in pixel space, starting from x_boundary (already adversarial).

    xi_step_scale decouples the exploration spread (xi * D * zs) from the
    exploitation step (xi * xi_step_scale * v) — Study 3/8 finding.

    snapshot_qs: list of Phase-3-internal query counts at which to capture the
    current boundary image (for the 7-frame visual).

    Returns dict with best_l2, final_l2, x_best, l2_history, queries_history,
    n_gens, snapshots (list of (q, l2, img_chw) or None).
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
    lam = lam_override
    mu  = lam
    weights, mueff = sep_cmaes_weights(mu)

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
        xi           = dist_to_orig / np.sqrt(t)
        xi_step      = xi * xi_step_scale

        # ── sample offspring ─────────────────────────────────────────────────
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

        # ── xi-shrink ────────────────────────────────────────────────────────
        m_shifted = np.clip(m + xi_step * v, 0.0, 1.0)
        while query(m_shifted.reshape(shape).astype(np.float32)) == y_true:
            xi_step   /= 2.0
            m_shifted  = np.clip(m + xi_step * v, 0.0, 1.0)
            if queries[0] >= max_queries:
                break

        m_new      = binary_search(query, m_shifted.reshape(shape).astype(np.float32),
                                   x_orig, y_true)
        m_new_flat = m_new.flatten().astype(np.float64)

        # ── backtrack ────────────────────────────────────────────────────────
        new_dist = float(np.linalg.norm(m_new_flat - x_orig_flat))
        n_back   = 0
        while new_dist > dist_to_orig and n_back < tau and queries[0] < max_queries:
            xi_step   /= 2.0
            cand       = np.clip(m + xi_step * v, 0.0, 1.0).reshape(shape).astype(np.float32)
            m_new      = binary_search(query, cand, x_orig, y_true)
            m_new_flat = m_new.flatten().astype(np.float64)
            new_dist   = float(np.linalg.norm(m_new_flat - x_orig_flat))
            n_back    += 1

        m      = m_new_flat
        new_l2 = float(np.linalg.norm(m - x_orig_flat))
        l2_history.append(new_l2)
        queries_history.append(queries[0])

        if new_l2 < best_l2:
            best_l2 = new_l2
            x_best  = m.reshape(shape).astype(np.float32).copy()

        # ── snapshots ────────────────────────────────────────────────────────
        if snapshots is not None:
            while snap_next < len(snapshot_qs) and queries[0] >= snapshot_qs[snap_next]:
                snapshots.append((queries[0], new_l2,
                                  m.reshape(shape).astype(np.float32).copy()))
                snap_next += 1

        t += 1

    # fill any missed snapshot targets with final state
    if snapshots is not None:
        while snap_next < len(snapshot_qs):
            snapshots.append((queries[0], l2_history[-1],
                              m.reshape(shape).astype(np.float32).copy()))
            snap_next += 1

    return dict(
        best_l2=best_l2,
        final_l2=l2_history[-1],
        x_best=x_best,
        l2_history=l2_history,
        queries_history=queries_history,
        n_gens=t - 1,
        snapshots=snapshots,
    )


# ── Visual output ─────────────────────────────────────────────────────────────

def save_visual(img_idx, x_orig, vis_rows, out_path):
    """
    vis_rows: list of dicts, one per (model, arm), each with keys:
        model_label, arm_label, x_init, x_boundary,
        snapshots [(q,l2,img)×3], x_best, best_l2
    Saves an (n_rows × 7) grid: orig | init | boundary | snap1 | snap2 | snap3 | best
    """
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
        show(axes[ri, 2], row['x_boundary'],
             f"L2={row['phase1_l2']:.2f}")
        for ci, (q, l2, img) in enumerate(row['snapshots'], start=3):
            show(axes[ri, ci], img, f"q={q}\nL2={l2:.2f}")
        show(axes[ri, 6], row['x_best'],
             f"best L2={row['best_l2']:.2f}")

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

    MOCK     = args.mock
    N_IMG    = 8    if MOCK else 200
    Q_TOTAL  = 200  if MOCK else 1500
    VISUAL_IMG_INDICES = [0, 1] if MOCK else [0, 40, 80, 120, 160]
    TAG      = 'mock' if MOCK else f'q{Q_TOTAL}_n{N_IMG}'

    OUT = os.path.join(os.path.dirname(__file__), 'outputs', f'exp_corruption_phase1_{TAG}')
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

    # ── Images: jointly classified (both models correct) ─────────────────────
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
            images.append(x)
            labels.append(label)
            counts[label] += 1
        if sum(counts) >= N_IMG:
            break

    images = np.stack(images)[:N_IMG]
    labels = np.array(labels)[:N_IMG]
    print(f'Images: {len(images)}  |  label dist: {counts}')

    # ── Experiment loop ───────────────────────────────────────────────────────
    rows     = []
    vis_data = {}   # (img_idx) -> dict per (model_label, arm_label)
    t0       = time.time()

    for img_idx in range(len(images)):
        x_orig = images[img_idx]
        y_true = int(labels[img_idx])
        seed_base = img_idx * 1000
        want_vis  = img_idx in VISUAL_IMG_INDICES

        for mname, _, _ in MODEL_SPECS:
            oracle_fn = oracles[mname]

            # ── Phase 1 (run once per model per image, shared across arms) ───
            x_init_u, x_bnd_u, pl2_u, pq_u = run_uniform_phase1(
                oracle_fn, x_orig, y_true, seed=seed_base)

            x_init_c, x_bnd_c, pl2_c, pq_c, win_c, per_c = run_corruption_phase1(
                oracle_fn, x_orig, y_true, seed=seed_base)

            phase1_l2_uniform = pl2_u  # reference for IR_total

            # ── Phase 3 arms ─────────────────────────────────────────────────
            for arm_idx, (arm_name, p1_type, lam, xi_ss, tau) in enumerate(ARMS):
                if p1_type == 'uniform':
                    x_init = x_init_u
                    x_bnd  = x_bnd_u
                    pl2    = pl2_u
                    pq     = pq_u
                else:
                    x_init = x_init_c
                    x_bnd  = x_bnd_c
                    pl2    = pl2_c
                    pq     = pq_c

                if x_bnd is None:
                    continue

                q_phase3 = max(10, Q_TOTAL - pq)

                # snapshot at Phase-3-internal queries 1/3, 2/3, end
                snap_qs = ([int(q_phase3 * f) for f in [1/3, 2/3, 1.0]]
                           if want_vis else None)

                res = run_phase3(
                    oracle_fn, x_orig, y_true, x_bnd, q_phase3,
                    seed=seed_base + arm_idx + 1,
                    lam_override=lam, xi_step_scale=xi_ss, tau=tau,
                    snapshot_qs=snap_qs,
                )

                ir_phase3 = (pl2 - res['best_l2']) / pl2 if pl2 > 0 else 0.0
                ir_total  = ((phase1_l2_uniform - res['best_l2']) / phase1_l2_uniform
                             if phase1_l2_uniform and phase1_l2_uniform > 0 else None)

                row = dict(
                    model          = mname,
                    arm            = arm_name,
                    phase1_type    = p1_type,
                    phase3_cfg     = f'lam{lam}_xi{xi_ss}_tau{tau}',
                    image_idx      = img_idx,
                    y_true         = y_true,
                    phase1_l2_uniform = phase1_l2_uniform,
                    phase1_l2      = pl2,
                    phase1_queries = pq,
                    winning_corruption = win_c if p1_type == 'corruption' else None,
                    jpeg_l2        = per_c.get('jpeg')    if p1_type == 'corruption' else None,
                    blur_l2        = per_c.get('blur')    if p1_type == 'corruption' else None,
                    fractal_l2     = per_c.get('fractal') if p1_type == 'corruption' else None,
                    best_l2        = res['best_l2'],
                    final_l2       = res['final_l2'],
                    IR_phase3      = ir_phase3,
                    IR_total       = ir_total,
                    n_gens         = res['n_gens'],
                    queries_phase3 = res['queries_history'][-1] if res['queries_history'] else 0,
                    l2_history     = res['l2_history'],
                    queries_history= res['queries_history'],
                )
                rows.append(row)

                if want_vis:
                    key = (img_idx, mname, arm_name)
                    vis_data[key] = dict(
                        model_label  = mname,
                        arm_label    = arm_name,
                        phase1_l2    = pl2,
                        x_init       = x_init,
                        x_boundary   = x_bnd,
                        snapshots    = res['snapshots'],   # list of 3 (q,l2,img)
                        x_best       = res['x_best'],
                        best_l2      = res['best_l2'],
                    )

        elapsed = time.time() - t0
        if (img_idx + 1) % 10 == 0 or img_idx < 3:
            print(f'  img {img_idx+1:3d}/{len(images)}  ({elapsed:.0f}s elapsed)')

    print(f'\nTotal time: {time.time()-t0:.1f}s  |  {len(rows)} rows')

    # ── Save results ──────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    df.to_parquet(f'{OUT}/results.parquet', index=False)
    print(f'Saved {OUT}/results.parquet')

    df_plain = df.drop(columns=['l2_history', 'queries_history'])
    df_plain.to_csv(f'{OUT}/summary.csv', index=False)

    # ── Summary table ─────────────────────────────────────────────────────────
    print('\n=== Summary by (model, arm) ===')
    summary = df_plain.groupby(['model', 'arm']).agg(
        n               = ('best_l2', 'count'),
        mean_phase1_l2  = ('phase1_l2', 'mean'),
        mean_best_l2    = ('best_l2', 'mean'),
        median_best_l2  = ('best_l2', 'median'),
        mean_IR_phase3  = ('IR_phase3', 'mean'),
        mean_IR_total   = ('IR_total', 'mean'),
        mean_n_gens     = ('n_gens', 'mean'),
    ).round(4)
    print(summary.to_string())

    # ── Plot 1: best_l2 distribution per arm ─────────────────────────────────
    arm_order = ['A', 'B', 'C', 'D']
    arm_labels = {
        'A': 'A\nuniform\nbaseline',
        'B': 'B\nuniform\ntuned',
        'C': 'C\ncorrupt\nbaseline',
        'D': 'D\ncorrupt\ntuned',
    }
    model_colors = {'standard': '#1976D2', 'robust': '#D32F2F'}

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    for ax, (mname, _, _) in zip(axes, MODEL_SPECS):
        sub = df_plain[df_plain.model == mname]
        data   = [sub[sub.arm == a]['best_l2'].values for a in arm_order]
        bp = ax.boxplot(data, labels=[arm_labels[a] for a in arm_order],
                        patch_artist=True, medianprops=dict(color='white', lw=2))
        for patch in bp['boxes']:
            patch.set_facecolor(model_colors[mname])
            patch.set_alpha(0.7)
        ax.set_title(mname)
        ax.set_ylabel('best L2' if mname == 'standard' else '')
        ax.set_xlabel('arm')
        ax.grid(axis='y', alpha=0.3)
    plt.suptitle('Best L2 distribution per arm')
    plt.tight_layout()
    plt.savefig(f'{OUT}/best_l2_by_arm.png', dpi=130, bbox_inches='tight')
    plt.close()
    print(f'Saved {OUT}/best_l2_by_arm.png')

    # ── Plot 2: convergence curves per arm ────────────────────────────────────
    arm_colors = {'A': '#555555', 'B': '#888888', 'C': '#1565C0', 'D': '#0D47A1'}
    arm_styles = {'A': '--', 'B': ':', 'C': '-', 'D': '-.'}

    q_max  = df_plain['queries_phase3'].max()
    q_grid = np.linspace(0, q_max, 60).astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, (mname, _, _) in zip(axes, MODEL_SPECS):
        for a in arm_order:
            sub = df[(df.model == mname) & (df.arm == a)]
            curves = []
            for _, row in sub.iterrows():
                if len(row.queries_history) < 2:
                    continue
                curves.append(np.interp(q_grid, row.queries_history, row.l2_history))
            if not curves:
                continue
            mat  = np.stack(curves)
            mean = mat.mean(0)
            std  = mat.std(0)
            ax.plot(q_grid, mean, label=f'arm {a}',
                    color=arm_colors[a], ls=arm_styles[a], lw=1.6)
            ax.fill_between(q_grid, mean - std, mean + std,
                            color=arm_colors[a], alpha=0.12)
        ax.set_title(mname)
        ax.set_xlabel('Phase 3 queries')
        if mname == 'standard':
            ax.set_ylabel('L2 to x_orig')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    plt.suptitle('Convergence by arm (Phase 3 queries; mean ± std)')
    plt.tight_layout()
    plt.savefig(f'{OUT}/convergence_by_arm.png', dpi=130, bbox_inches='tight')
    plt.close()
    print(f'Saved {OUT}/convergence_by_arm.png')

    # ── Plot 3: Phase 1 L2 distribution ──────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, (mname, _, _) in zip(axes, MODEL_SPECS):
        # one row per image: uniform vs corruption phase1_l2
        sub_u = df_plain[(df_plain.model == mname) & (df_plain.arm == 'A')]
        sub_c = df_plain[(df_plain.model == mname) & (df_plain.arm == 'C')]
        ax.hist(sub_u['phase1_l2'].values, bins=30, alpha=0.6, label='uniform',
                color='#888888', density=True)
        ax.hist(sub_c['phase1_l2'].values, bins=30, alpha=0.6, label='corruption',
                color='#1565C0', density=True)
        ax.set_title(mname)
        ax.set_xlabel('Phase 1 boundary L2')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    plt.suptitle('Phase 1 boundary L2: uniform vs corruption init')
    plt.tight_layout()
    plt.savefig(f'{OUT}/phase1_l2_distribution.png', dpi=130, bbox_inches='tight')
    plt.close()
    print(f'Saved {OUT}/phase1_l2_distribution.png')

    # ── Plot 4: winning corruption ────────────────────────────────────────────
    sub_c = df_plain[(df_plain.phase1_type == 'corruption') &
                     (df_plain.arm == 'C') &
                     (df_plain.winning_corruption.notna())]
    if len(sub_c) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        for ax, (mname, _, _) in zip(axes, MODEL_SPECS):
            vc = sub_c[sub_c.model == mname]['winning_corruption'].value_counts()
            ax.bar(vc.index, vc.values, color=['#E65100', '#1565C0', '#2E7D32'])
            ax.set_title(mname)
            ax.set_xlabel('corruption')
            ax.set_ylabel('# images it won')
        plt.suptitle('Winning corruption (best Phase 1 L2) per image')
        plt.tight_layout()
        plt.savefig(f'{OUT}/winning_corruption.png', dpi=130, bbox_inches='tight')
        plt.close()
        print(f'Saved {OUT}/winning_corruption.png')

    # ── Visual progression figures ────────────────────────────────────────────
    for img_idx in VISUAL_IMG_INDICES:
        x_orig_vis = images[img_idx]
        vis_rows   = []
        for mname, _, _ in MODEL_SPECS:
            for arm_name, _, _, _, _ in ARMS:
                key = (img_idx, mname, arm_name)
                if key not in vis_data:
                    continue
                vis_rows.append(vis_data[key])
        if not vis_rows:
            continue
        out_path = f'{OUT}/visual_img{img_idx}.png'
        save_visual(img_idx, x_orig_vis, vis_rows, out_path)
        print(f'Saved {out_path}')

    print('\nDone.')


if __name__ == '__main__':
    main()
