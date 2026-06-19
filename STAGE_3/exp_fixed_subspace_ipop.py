#!/usr/bin/env python3
"""
exp_fixed_subspace_ipop.py — Stage 3: fixed Phase 1, sep-CMA-ES population study.

Fixes from exp_full_cmaes_subspace:
  - Phase 1 runs only 3 corruptions (jpeg, blur, fractal_random) for boundary
    finding (~141 queries). All 11 directions come from DIRECTION_ZOO at zero
    oracle cost.
  - sep-CMA-ES only (full CMA-ES dropped: consistently loses to sep for
    QR-orthogonalized basis regardless of lam).

Phase 3 basis: 14 directions = 11 corruption (boundary dir if Phase 1 succeeded,
DIRECTION_ZOO otherwise) + 3 DCT bands. QR-orthogonalized -> k≤14.

Arms (all sep-CMA-ES, k≤14):
  E_lam11   lam=11 (auto for k=14), fixed     — baseline
  E_lam28   lam=28, fixed                     — reference
  E_IPOP    lam starts at 11; doubles (up to
            lam_max=88) when best_l2 stagnates — adaptive
            for W=5 consecutive generations

IPOP rule: after each generation, if running best_l2 did not improve for W
consecutive generations, set lam = min(lam*2, lam_max) and recompute CMA-ES
weight/learning-rate params. D (diagonal covariance), t, theta_m unchanged.

Usage:
    python exp_fixed_subspace_ipop.py            # full run (~1.5h)
    python exp_fixed_subspace_ipop.py --mock     # N=8, Q=200
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
from phase1_zoo import INIT_ZOO, DIRECTION_ZOO
from attacks.utils.subspace import corruption_basis

MODEL_SPECS = [
    ('standard', 'Standard',                 'Linf'),
    ('robust',   'Wang2023Better_WRN-28-10', 'Linf'),
]

PHASE1_CORR_TYPES = ['jpeg', 'blur', 'fractal_random']

IPOP_W       = 5    # stagnation window (generations without improvement)
IPOP_LAM_MAX = 88   # cap at 3 doublings from lam=11: 11->22->44->88


# ── DCT band direction vector ──────────────────────────────────────────────────

def dct_band_vector(shape_chw, fy, fx):
    C, H, W = shape_chw
    ys = np.arange(H, dtype=np.float64)
    xs = np.arange(W, dtype=np.float64)
    v2d = (np.cos(np.pi * (2 * ys[:, None] + 1) * fy / (2 * H)) *
           np.cos(np.pi * (2 * xs[None, :] + 1) * fx / (2 * W)))
    v = np.tile(v2d.flatten(), C).astype(np.float64)
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-12 else v


# ── Phase 1 (fixed): 3 corruption types for boundary, DIRECTION_ZOO for all 11 ─

def run_fixed_phase1(oracle_fn, x_orig, y_true, seed):
    """
    Runs binary-search boundary finding for jpeg, blur, fractal_random only
    (~141 queries total). All 11 corruption directions are collected from
    DIRECTION_ZOO at zero oracle cost; boundary directions replace DIRECTION_ZOO
    for the 3 types where Phase 1 succeeded. 3 DCT band vectors are appended.
    Returns up to 14 direction vectors.
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

    for name in PHASE1_CORR_TYPES:
        x_adv = INIT_ZOO[name](query, x_orig, y_true, rng)
        if x_adv is None:
            per_corr[name] = None
        else:
            x_bnd = binary_search(query, x_adv, x_orig, y_true)
            l2    = float(np.linalg.norm(x_bnd.flatten().astype(np.float64) - x_orig_flat))
            per_corr[name] = l2
            dir_vecs[name] = x_bnd.flatten().astype(np.float64) - x_orig_flat
            if l2 < best_l2:
                best_l2   = l2
                best_bnd  = x_bnd
                best_init = x_adv
                best_name = name

    # All 11 corruption directions from DIRECTION_ZOO (0 oracle queries).
    # Boundary directions from Phase 1 take priority for the 3 types we ran.
    for name, fn in DIRECTION_ZOO.items():
        if name not in dir_vecs:
            d = fn(x_orig).flatten().astype(np.float64) - x_orig_flat
            if np.linalg.norm(d) > 1e-8:
                dir_vecs[name] = d

    # 3 fixed DCT band vectors
    for band_name, fy, fx in [('dct_low', 1, 0), ('dct_mid', 8, 8), ('dct_high', 16, 16)]:
        dir_vecs[band_name] = dct_band_vector(x_orig.shape, fy, fx)

    if best_bnd is None:
        return None, None, None, queries[0], None, per_corr, dir_vecs

    return best_init, best_bnd, best_l2, queries[0], best_name, per_corr, dir_vecs


# ── Shared CMA-ES parameter computation ───────────────────────────────────────

def _sep_params(k, lam):
    weights, mueff = sep_cmaes_weights(lam)
    c1  = 2.0 / ((k + 1.3) ** 2 + mueff)
    cmu = min(1.0 - c1,
              2.0 * (mueff - 2.0 + 1.0 / mueff) / ((k + 2.0) ** 2 + mueff))
    cmu = cmu * (k + 2.0) / 3.0
    return weights, c1, cmu


# ── sep-CMA-ES with fixed lam ──────────────────────────────────────────────────

def run_sep_cmaes(oracle_fn, x_orig, y_true, basis, max_queries, seed,
                  lam_override=None, x_b_override=None, snapshot_fracs=None):
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
    D       = np.ones(k,  dtype=np.float64)

    lam               = lam_override if lam_override is not None else 4 + int(3 * np.log(k))
    weights, c1, cmu  = _sep_params(k, lam)

    def theta_bs(theta_adv):
        lo, hi = theta_orig.copy(), theta_adv.copy()
        for _ in range(BS_STEPS):
            mid = 0.5 * (lo + hi)
            img = to_pixel(mid).reshape(shape).astype(np.float32)
            if query(img) != y_true:
                hi = mid
            else:
                lo = mid
        return hi

    l2_history      = [init_dist]
    queries_history = [0]

    snapshots    = None
    snap_targets = []
    snap_next    = 0
    if snapshot_fracs is not None:
        snapshots    = []
        snap_targets = [int(round(f * max_queries)) for f in snapshot_fracs]
        if snap_targets[0] <= 0:
            snapshots.append((0, init_dist,
                              to_pixel(theta_m).reshape(shape).astype(np.float32)))
            snap_next = 1

    t = 1
    while queries[0] < max_queries:
        dist_to_orig = float(np.linalg.norm(to_pixel(theta_m) - x_orig_flat))
        xi           = dist_to_orig / np.sqrt(t)

        zs         = rng.standard_normal((lam, k))
        theta_cand = theta_m + xi * D * zs

        labels = np.empty(lam, dtype=np.int64)
        l2s    = np.empty(lam, dtype=np.float64)
        for i in range(lam):
            x_cand    = to_pixel(theta_cand[i]).reshape(shape).astype(np.float32)
            labels[i] = query(x_cand)
            l2s[i]    = np.linalg.norm(x_cand.flatten().astype(np.float64) - x_orig_flat)
            if queries[0] >= max_queries:
                lam_eff = i + 1
                zs, theta_cand = zs[:lam_eff], theta_cand[:lam_eff]
                labels, l2s    = labels[:lam_eff], l2s[:lam_eff]
                break

        is_adv  = labels != y_true
        fitness = np.array([objective(l2, adv) for l2, adv in zip(l2s, is_adv)])
        w_eff   = weights[:len(fitness)] / weights[:len(fitness)].sum()

        v = mean_shift_direction(zs, fitness, is_adv, w_eff)
        D = update_diagonal_covariance(D, zs, fitness, w_eff, cmu)

        theta_shifted = theta_m + xi * v
        while query(to_pixel(theta_shifted).reshape(shape).astype(np.float32)) == y_true:
            xi           /= 2.0
            theta_shifted = theta_m + xi * v
            if queries[0] >= max_queries:
                break

        theta_new = theta_bs(theta_shifted)
        new_dist  = float(np.linalg.norm(to_pixel(theta_new) - x_orig_flat))

        for _ in range(TAU):
            if new_dist <= dist_to_orig or queries[0] >= max_queries:
                break
            xi           /= 2.0
            theta_shifted = theta_m + xi * v
            theta_new     = theta_bs(theta_shifted)
            new_dist      = float(np.linalg.norm(to_pixel(theta_new) - x_orig_flat))

        theta_m = theta_new
        l2_history.append(new_dist)
        queries_history.append(queries[0])

        if snapshots is not None:
            while snap_next < len(snap_targets) and queries[0] >= snap_targets[snap_next]:
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


# ── sep-CMA-ES with IPOP ───────────────────────────────────────────────────────

def run_sep_cmaes_ipop(oracle_fn, x_orig, y_true, basis, max_queries, seed,
                       lam_init=None, lam_max=IPOP_LAM_MAX, W=IPOP_W,
                       x_b_override=None, snapshot_fracs=None):
    """
    sep-CMA-ES with IPOP population doubling.

    When best_l2 (running minimum) does not improve for W consecutive
    generations, lam is doubled up to lam_max. D, t, theta_m are preserved —
    this is a population boost, not a restart.

    Extra returned keys (sanity-check metrics):
        lam_history       list[int]  — lam used in each generation
        n_doublings       int        — total number of doublings
        final_lam         int        — lam at end of run
        mean_lam          float      — mean lam weighted by generation count
        doubling_gens     list[int]  — generation index at each doubling
        doubling_queries  list[int]  — cumulative queries at each doubling
        doubling_best_l2  list[float]— best_l2 at each doubling event
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
    D       = np.ones(k,  dtype=np.float64)

    lam              = lam_init if lam_init is not None else 4 + int(3 * np.log(k))
    weights, c1, cmu = _sep_params(k, lam)

    def theta_bs(theta_adv):
        lo, hi = theta_orig.copy(), theta_adv.copy()
        for _ in range(BS_STEPS):
            mid = 0.5 * (lo + hi)
            img = to_pixel(mid).reshape(shape).astype(np.float32)
            if query(img) != y_true:
                hi = mid
            else:
                lo = mid
        return hi

    l2_history      = [init_dist]
    queries_history = [0]
    lam_history     = []

    # IPOP state
    best_so_far      = init_dist
    stagnation_count = 0
    n_doublings      = 0
    doubling_gens     = []
    doubling_queries  = []
    doubling_best_l2  = []

    snapshots    = None
    snap_targets = []
    snap_next    = 0
    if snapshot_fracs is not None:
        snapshots    = []
        snap_targets = [int(round(f * max_queries)) for f in snapshot_fracs]
        if snap_targets[0] <= 0:
            snapshots.append((0, init_dist,
                              to_pixel(theta_m).reshape(shape).astype(np.float32)))
            snap_next = 1

    t = 1
    while queries[0] < max_queries:
        dist_to_orig = float(np.linalg.norm(to_pixel(theta_m) - x_orig_flat))
        xi           = dist_to_orig / np.sqrt(t)

        lam_history.append(lam)

        zs         = rng.standard_normal((lam, k))
        theta_cand = theta_m + xi * D * zs

        labels = np.empty(lam, dtype=np.int64)
        l2s    = np.empty(lam, dtype=np.float64)
        for i in range(lam):
            x_cand    = to_pixel(theta_cand[i]).reshape(shape).astype(np.float32)
            labels[i] = query(x_cand)
            l2s[i]    = np.linalg.norm(x_cand.flatten().astype(np.float64) - x_orig_flat)
            if queries[0] >= max_queries:
                lam_eff = i + 1
                zs, theta_cand = zs[:lam_eff], theta_cand[:lam_eff]
                labels, l2s    = labels[:lam_eff], l2s[:lam_eff]
                break

        is_adv  = labels != y_true
        fitness = np.array([objective(l2, adv) for l2, adv in zip(l2s, is_adv)])
        w_eff   = weights[:len(fitness)] / weights[:len(fitness)].sum()

        v = mean_shift_direction(zs, fitness, is_adv, w_eff)
        D = update_diagonal_covariance(D, zs, fitness, w_eff, cmu)

        theta_shifted = theta_m + xi * v
        while query(to_pixel(theta_shifted).reshape(shape).astype(np.float32)) == y_true:
            xi           /= 2.0
            theta_shifted = theta_m + xi * v
            if queries[0] >= max_queries:
                break

        theta_new = theta_bs(theta_shifted)
        new_dist  = float(np.linalg.norm(to_pixel(theta_new) - x_orig_flat))

        for _ in range(TAU):
            if new_dist <= dist_to_orig or queries[0] >= max_queries:
                break
            xi           /= 2.0
            theta_shifted = theta_m + xi * v
            theta_new     = theta_bs(theta_shifted)
            new_dist      = float(np.linalg.norm(to_pixel(theta_new) - x_orig_flat))

        theta_m = theta_new
        l2_history.append(new_dist)
        queries_history.append(queries[0])

        if snapshots is not None:
            while snap_next < len(snap_targets) and queries[0] >= snap_targets[snap_next]:
                snapshots.append((queries[0], new_dist,
                                  to_pixel(theta_m).reshape(shape).astype(np.float32)))
                snap_next += 1

        # IPOP stagnation check on running best
        if new_dist < best_so_far - 1e-8:
            best_so_far      = new_dist
            stagnation_count = 0
        else:
            stagnation_count += 1

        if stagnation_count >= W and lam < lam_max:
            new_lam = min(lam * 2, lam_max)
            doubling_gens.append(t)
            doubling_queries.append(queries[0])
            doubling_best_l2.append(best_so_far)
            lam              = new_lam
            weights, c1, cmu = _sep_params(k, lam)
            stagnation_count = 0
            n_doublings      += 1

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
        # IPOP diagnostics
        lam_history=lam_history,
        n_doublings=n_doublings,
        final_lam=lam,
        mean_lam=float(np.mean(lam_history)) if lam_history else float(lam),
        first_doubling_gen=(doubling_gens[0]    if doubling_gens else -1),
        first_doubling_queries=(doubling_queries[0] if doubling_queries else -1),
        doubling_gens=doubling_gens,
        doubling_queries=doubling_queries,
        doubling_best_l2=doubling_best_l2,
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
    TAG     = 'mock' if MOCK else f'q{Q_TOTAL}_n{N_IMG}'
    VISUAL_IMG_INDICES = [0, 1] if MOCK else [0, 40, 80, 120, 160]
    SNAP_FRACS = [1/3, 2/3, 1.0]

    OUT = os.path.join(os.path.dirname(__file__), 'outputs',
                       f'exp_fixed_subspace_ipop_{TAG}')
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
    images, labels_list = [], []
    counts = [0] * 10
    for img_pil, label in ds:
        if counts[label] >= per_class:
            continue
        x = np.array(img_pil, dtype=np.float32).transpose(2, 0, 1) / 255.0
        if oracles['standard'](x) == label and oracles['robust'](x) == label:
            images.append(x); labels_list.append(label); counts[label] += 1
        if sum(counts) >= N_IMG:
            break
    images = np.stack(images)[:N_IMG]
    labels = np.array(labels_list)[:N_IMG]
    print(f'Images: {len(images)}  |  label dist: {counts}')

    rows = []
    t0   = time.time()

    for img_idx in range(len(images)):
        x_orig    = images[img_idx]
        y_true    = int(labels[img_idx])
        seed_base = img_idx * 1000
        want_vis  = img_idx in VISUAL_IMG_INDICES

        for mname, _, _ in MODEL_SPECS:
            oracle_fn = oracles[mname]

            (x_init, x_bnd, pl2, pq,
             win_name, per_corr, dir_vecs) = run_fixed_phase1(
                oracle_fn, x_orig, y_true, seed=seed_base)

            if x_bnd is None:
                continue

            dvecs_list = list(dir_vecs.values())
            k_total    = len(dvecs_list)
            try:
                basis = corruption_basis(dvecs_list)
            except ValueError:
                continue

            q_phase3 = max(10, Q_TOTAL - pq)

            res_lam11 = run_sep_cmaes(
                oracle_fn, x_orig, y_true, basis, q_phase3,
                seed=seed_base + 1, lam_override=11,
                x_b_override=x_bnd,
                snapshot_fracs=SNAP_FRACS if want_vis else None)

            res_lam28 = run_sep_cmaes(
                oracle_fn, x_orig, y_true, basis, q_phase3,
                seed=seed_base + 2, lam_override=28,
                x_b_override=x_bnd,
                snapshot_fracs=SNAP_FRACS if want_vis else None)

            res_ipop = run_sep_cmaes_ipop(
                oracle_fn, x_orig, y_true, basis, q_phase3,
                seed=seed_base + 3,
                lam_init=11, lam_max=IPOP_LAM_MAX, W=IPOP_W,
                x_b_override=x_bnd,
                snapshot_fracs=SNAP_FRACS if want_vis else None)

            for arm_name, res, arm_lam in [
                ('E_lam11', res_lam11, 11),
                ('E_lam28', res_lam28, 28),
                ('E_IPOP',  res_ipop,  11),
            ]:
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
                    final_l2=res['final_dist'],
                    IR_phase3=ir_p3,
                    n_gens=res['n_gens'],
                    queries_phase3=res['queries_used'],
                    l2_history=res['l2_history'],
                    queries_history=res['queries_history'],
                    # IPOP diagnostics (0/-1 for fixed-lam arms)
                    n_doublings=res.get('n_doublings', 0),
                    final_lam=res.get('final_lam', arm_lam),
                    mean_lam=res.get('mean_lam', float(arm_lam)),
                    first_doubling_gen=res.get('first_doubling_gen', -1),
                    first_doubling_queries=res.get('first_doubling_queries', -1),
                    lam_history=res.get('lam_history', []),
                    doubling_gens=res.get('doubling_gens', []),
                    doubling_queries=res.get('doubling_queries', []),
                    doubling_best_l2=res.get('doubling_best_l2', []),
                )
                rows.append(row)

        elapsed = time.time() - t0
        if (img_idx + 1) % 10 == 0 or img_idx < 3:
            print(f'  img {img_idx+1:3d}/{len(images)}  ({elapsed:.0f}s elapsed)')

    print(f'\nTotal time: {time.time()-t0:.1f}s  |  {len(rows)} rows')

    df = pd.DataFrame(rows)
    df.to_parquet(f'{OUT}/results.parquet', index=False)
    print(f'Saved {OUT}/results.parquet')

    plain = df.drop(columns=['l2_history', 'queries_history',
                              'lam_history', 'doubling_gens',
                              'doubling_queries', 'doubling_best_l2'])

    # ── Performance summary ───────────────────────────────────────────────────
    print('\n=== Summary by (model, arm) ===')
    summary = plain.groupby(['model', 'arm']).agg(
        n              = ('best_l2',     'count'),
        mean_phase1_l2 = ('phase1_l2',  'mean'),
        mean_best_l2   = ('best_l2',    'mean'),
        median_best_l2 = ('best_l2',    'median'),
        mean_IR_phase3 = ('IR_phase3',  'mean'),
        mean_n_gens    = ('n_gens',     'mean'),
        mean_k_total   = ('k_total',    'mean'),
    ).round(4)
    print(summary.to_string())

    # ── IPOP diagnostics ──────────────────────────────────────────────────────
    print('\n=== IPOP diagnostics (E_IPOP arm only) ===')
    ipop = plain[plain.arm == 'E_IPOP']
    ipop_diag = ipop.groupby('model').agg(
        mean_n_doublings       = ('n_doublings',       'mean'),
        frac_triggered         = ('n_doublings',       lambda x: (x > 0).mean()),
        mean_final_lam         = ('final_lam',         'mean'),
        mean_mean_lam          = ('mean_lam',          'mean'),
        mean_first_doubling_gen= ('first_doubling_gen',
                                  lambda x: x[x > 0].mean() if (x > 0).any() else float('nan')),
    ).round(4)
    print(ipop_diag.to_string())

    arm_order  = ['E_lam11', 'E_lam28', 'E_IPOP']
    arm_labels = ['lam=11\n(fixed)', 'lam=28\n(fixed)', 'IPOP\n(11→88)']
    colors     = {'standard': '#1976D2', 'robust': '#D32F2F'}

    # ── Plot 1: best_l2 boxplot ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=False)
    for ax, (mname, _, _) in zip(axes, MODEL_SPECS):
        sub  = plain[plain.model == mname]
        data = [sub[sub.arm == a]['best_l2'].values for a in arm_order]
        bp   = ax.boxplot(data, labels=arm_labels, patch_artist=True,
                          medianprops=dict(color='white', lw=2))
        for patch in bp['boxes']:
            patch.set_facecolor(colors[mname]); patch.set_alpha(0.7)
        ax.set_title(mname)
        ax.set_ylabel('best L2')
        ax.grid(axis='y', alpha=0.3)
    plt.suptitle('Best L2 by arm  (fixed Phase 1, k=14, sep-CMA-ES)')
    plt.tight_layout()
    plt.savefig(f'{OUT}/best_l2_by_arm.png', dpi=130, bbox_inches='tight')
    plt.close()
    print(f'Saved {OUT}/best_l2_by_arm.png')

    # ── Plot 2: convergence curves (mean L2 vs queries) ───────────────────────
    q_max  = Q_TOTAL
    q_grid = np.linspace(0, q_max, 120)
    arm_styles = {
        'E_lam11': ('--', '#888888'),
        'E_lam28': ('-.',  '#F57C00'),
        'E_IPOP':  ('-',  '#1565C0'),
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
    for ax, (mname, _, _) in zip(axes, MODEL_SPECS):
        for arm_name in arm_order:
            sub    = df[(df.model == mname) & (df.arm == arm_name)]
            curves = []
            for _, row in sub.iterrows():
                qs = [0] + list(row.queries_history)
                l2 = [row.phase1_l2] + list(row.l2_history)
                curves.append(np.interp(q_grid, qs, l2))
            if not curves:
                continue
            arr  = np.stack(curves)
            mean = arr.mean(axis=0)
            std  = arr.std(axis=0)
            ls, c = arm_styles[arm_name]
            label = dict(zip(arm_order, arm_labels))[arm_name].replace('\n', ' ')
            ax.plot(q_grid, mean, ls=ls, color=c, label=label)
            ax.fill_between(q_grid, mean - std, mean + std, color=c, alpha=0.12)
        ax.set_title(mname)
        ax.set_xlabel('total queries (Phase 1 + Phase 3)')
        ax.set_ylabel('mean L2')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    plt.suptitle('Convergence by arm')
    plt.tight_layout()
    plt.savefig(f'{OUT}/convergence_by_arm.png', dpi=130, bbox_inches='tight')
    plt.close()
    print(f'Saved {OUT}/convergence_by_arm.png')

    # ── Plot 3: IPOP diagnostics ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # 3a: histogram of n_doublings
    ax = axes[0]
    for mname, color in [('standard', '#1976D2'), ('robust', '#D32F2F')]:
        sub = plain[(plain.model == mname) & (plain.arm == 'E_IPOP')]
        counts_arr = sub['n_doublings'].value_counts().sort_index()
        ax.bar(counts_arr.index + (0.2 if mname == 'robust' else -0.2),
               counts_arr.values / len(sub),
               width=0.35, color=color, alpha=0.7, label=mname)
    ax.set_xlabel('n_doublings')
    ax.set_ylabel('fraction of images')
    ax.set_title('How often IPOP triggers')
    ax.legend()
    ax.set_xticks([0, 1, 2, 3])

    # 3b: mean lam over query fraction (using lam_history + queries_history)
    ax = axes[1]
    q_frac = np.linspace(0, 1, 80)
    for mname, color in [('standard', '#1976D2'), ('robust', '#D32F2F')]:
        sub = df[(df.model == mname) & (df.arm == 'E_IPOP')]
        lam_curves = []
        for _, row in sub.iterrows():
            lh = row.lam_history
            qh = row.queries_history[1:]  # one entry per gen (skip q=0 init)
            if len(lh) == 0 or len(qh) == 0:
                continue
            q_total_run = row.queries_phase3
            if q_total_run <= 0:
                continue
            fracs = np.array(qh) / q_total_run
            lam_curves.append(np.interp(q_frac, fracs, lh,
                                         left=lh[0], right=lh[-1]))
        if lam_curves:
            arr  = np.stack(lam_curves)
            mean = arr.mean(axis=0)
            std  = arr.std(axis=0)
            ax.plot(q_frac, mean, color=color, label=mname)
            ax.fill_between(q_frac, mean - std, mean + std, color=color, alpha=0.15)
    ax.set_xlabel('query fraction (Phase 3)')
    ax.set_ylabel('lam')
    ax.set_title('Mean lam over query budget (E_IPOP)')
    ax.legend()
    ax.grid(alpha=0.25)

    # 3c: fraction of images where IPOP triggered at all
    ax = axes[2]
    frac_triggered = {mname: plain[(plain.model == mname) & (plain.arm == 'E_IPOP')]['n_doublings'].gt(0).mean()
                      for mname, _, _ in MODEL_SPECS}
    ax.bar(list(frac_triggered.keys()), list(frac_triggered.values()),
           color=['#1976D2', '#D32F2F'], alpha=0.75)
    ax.set_ylim(0, 1)
    ax.set_ylabel('fraction of images')
    ax.set_title('Fraction where IPOP triggered')
    ax.grid(axis='y', alpha=0.3)

    plt.suptitle('IPOP diagnostics (E_IPOP arm)')
    plt.tight_layout()
    plt.savefig(f'{OUT}/ipop_diagnostics.png', dpi=130, bbox_inches='tight')
    plt.close()
    print(f'Saved {OUT}/ipop_diagnostics.png')

    print('\nDone.')


if __name__ == '__main__':
    main()
