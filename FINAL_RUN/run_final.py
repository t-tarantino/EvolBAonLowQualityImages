#!/usr/bin/env python3
"""
run_final.py — FINAL_RUN: the best STAGE_3 configuration (corruption-direction
subspace, sep-CMA-ES, fixed lambda=56), run at N=500, Q=2000, both models.

See ALGORITHM.md for the full description. This script only runs the attack and
saves raw results; all plotting lives in make_final_plots.py so that re-styling
a plot never requires rerunning the attack.

Usage:
    python run_final.py            # full run (N=500, Q=2000, ~2h on GPU)
    python run_final.py --mock     # smoke test (N=8, Q=200)
"""
import os, sys, time, warnings, argparse, json
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'STAGE_1'))

from evolba_baseline import (
    objective, binary_search, sep_cmaes_weights,
    update_diagonal_covariance, mean_shift_direction, TAU, BS_STEPS,
)
from phase1_zoo import INIT_ZOO, DIRECTION_ZOO
from attacks.utils.subspace import corruption_basis
from attacks.utils.metrics import all_metrics

MODEL_SPECS = [
    ('standard', 'Standard',                 'Linf'),
    ('robust',   'Wang2023Better_WRN-28-10', 'Linf'),
]
PHASE1_CORR_TYPES = ['jpeg', 'blur', 'fractal_random']
LAM = 56
SNAP_FRACS = [0.0, 0.25, 0.5, 0.75, 1.0]   # 5 Phase-3 fractions -> 7 cols with orig+boundary


# ── DCT band direction vector (identical to STAGE_3) ───────────────────────────

def dct_band_vector(shape_chw, fy, fx):
    C, H, W = shape_chw
    ys = np.arange(H, dtype=np.float64)
    xs = np.arange(W, dtype=np.float64)
    v2d = (np.cos(np.pi * (2 * ys[:, None] + 1) * fy / (2 * H)) *
           np.cos(np.pi * (2 * xs[None, :] + 1) * fx / (2 * W)))
    v = np.tile(v2d.flatten(), C).astype(np.float64)
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-12 else v


# ── Phase 1: fixed corruption set + zero-cost DIRECTION_ZOO + DCT bands ───────

def run_fixed_phase1(oracle_fn, x_orig, y_true, seed):
    rng         = np.random.default_rng(seed)
    x_orig_flat = x_orig.flatten().astype(np.float64)
    queries     = [0]

    def query(img):
        queries[0] += 1
        return oracle_fn(img)

    best_l2, best_bnd, best_name = float('inf'), None, None
    dir_vecs = {}

    for name in PHASE1_CORR_TYPES:
        x_adv = INIT_ZOO[name](query, x_orig, y_true, rng)
        if x_adv is not None:
            x_bnd = binary_search(query, x_adv, x_orig, y_true)
            l2    = float(np.linalg.norm(x_bnd.flatten().astype(np.float64) - x_orig_flat))
            dir_vecs[name] = x_bnd.flatten().astype(np.float64) - x_orig_flat
            if l2 < best_l2:
                best_l2, best_bnd, best_name = l2, x_bnd, name

    for name, fn in DIRECTION_ZOO.items():
        if name not in dir_vecs:
            d = fn(x_orig).flatten().astype(np.float64) - x_orig_flat
            if np.linalg.norm(d) > 1e-8:
                dir_vecs[name] = d

    for band_name, fy, fx in [('dct_low', 1, 0), ('dct_mid', 8, 8), ('dct_high', 16, 16)]:
        dir_vecs[band_name] = dct_band_vector(x_orig.shape, fy, fx)

    if best_bnd is None:
        return None, None, None, queries[0], dir_vecs
    return best_bnd, best_l2, best_name, queries[0], dir_vecs


def _sep_params(k, lam):
    weights, mueff = sep_cmaes_weights(lam)
    c1  = 2.0 / ((k + 1.3) ** 2 + mueff)
    cmu = min(1.0 - c1,
              2.0 * (mueff - 2.0 + 1.0 / mueff) / ((k + 2.0) ** 2 + mueff))
    cmu = cmu * (k + 2.0) / 3.0
    return weights, c1, cmu


# ── Phase 3: sep-CMA-ES in the k-dim subspace, fixed lambda=56 ────────────────

def run_sep_cmaes_lam56(oracle_fn, x_orig, y_true, basis, x_b, max_queries, seed,
                         snapshot_fracs=None):
    """Same algorithm as STAGE_3/exp_fixed_subspace_ipop2.py's run_sep_cmaes(lam=56),
    instrumented with a per-call-site query breakdown (offspring eval / xi-shrink /
    theta_bs) so the cost structure can be inspected without rerunning."""
    rng         = np.random.default_rng(seed)
    shape       = x_orig.shape
    k           = basis.shape[0]
    x_orig_flat = x_orig.flatten().astype(np.float64)

    queries = [0]
    qcount  = dict(offspring_eval=0, xi_shrink=0, theta_bs=0)

    def query(img, tag):
        queries[0] += 1
        qcount[tag] += 1
        return oracle_fn(img)

    x_b_flat   = x_b.flatten().astype(np.float64)
    init_dist  = float(np.linalg.norm(x_b_flat - x_orig_flat))
    theta_orig = basis @ (x_orig_flat - x_b_flat)

    def to_pixel(theta):
        return np.clip(x_b_flat + basis.T @ theta, 0.0, 1.0)

    theta_m = np.zeros(k, dtype=np.float64)
    D       = np.ones(k,  dtype=np.float64)
    weights, c1, cmu = _sep_params(k, LAM)

    def theta_bs(theta_adv):
        lo, hi = theta_orig.copy(), theta_adv.copy()
        for _ in range(BS_STEPS):
            mid = 0.5 * (lo + hi)
            img = to_pixel(mid).reshape(shape).astype(np.float32)
            if query(img, 'theta_bs') != y_true:
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

        zs         = rng.standard_normal((LAM, k))
        theta_cand = theta_m + xi * D * zs

        labels = np.empty(LAM, dtype=np.int64)
        l2s    = np.empty(LAM, dtype=np.float64)
        lam_eff = LAM
        for i in range(LAM):
            x_cand    = to_pixel(theta_cand[i]).reshape(shape).astype(np.float32)
            labels[i] = query(x_cand, 'offspring_eval')
            l2s[i]    = np.linalg.norm(x_cand.flatten().astype(np.float64) - x_orig_flat)
            if queries[0] >= max_queries:
                lam_eff = i + 1
                break
        zs, theta_cand = zs[:lam_eff], theta_cand[:lam_eff]
        labels, l2s    = labels[:lam_eff], l2s[:lam_eff]

        is_adv  = labels != y_true
        fitness = np.array([objective(l2, adv) for l2, adv in zip(l2s, is_adv)])
        w_eff   = weights[:len(fitness)] / weights[:len(fitness)].sum()
        v       = mean_shift_direction(zs, fitness, is_adv, w_eff)
        D       = update_diagonal_covariance(D, zs, fitness, w_eff, cmu)

        theta_shifted = theta_m + xi * v
        while query(to_pixel(theta_shifted).reshape(shape).astype(np.float32), 'xi_shrink') == y_true:
            xi /= 2.0; theta_shifted = theta_m + xi * v
            if queries[0] >= max_queries:
                break

        theta_new = theta_bs(theta_shifted)
        new_dist  = float(np.linalg.norm(to_pixel(theta_new) - x_orig_flat))
        for _ in range(TAU):
            if new_dist <= dist_to_orig or queries[0] >= max_queries:
                break
            xi /= 2.0; theta_shifted = theta_m + xi * v
            theta_new = theta_bs(theta_shifted)
            new_dist  = float(np.linalg.norm(to_pixel(theta_new) - x_orig_flat))

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

    final_img = to_pixel(theta_m).reshape(shape).astype(np.float32)
    return dict(
        init_dist=init_dist, final_dist=l2_history[-1], final_img=final_img,
        best_l2=float(min(l2_history)),
        n_gens=len(l2_history) - 1, queries_used=queries[0],
        l2_history=l2_history, queries_history=queries_history,
        snapshots=snapshots, qcount=qcount,
    )


# ── Main ────────────────────────────────────────────────────────────────────────

def select_images(oracles, n_img, seed=0):
    import torchvision
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
    return images, labels, counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mock', action='store_true')
    args = parser.parse_args()

    MOCK    = args.mock
    N_IMG   = 8   if MOCK else 500
    Q_TOTAL = 200 if MOCK else 2000
    TAG     = 'mock' if MOCK else f'q{Q_TOTAL}_n{N_IMG}'
    N_SNAPSHOT_IMAGES = 2 if MOCK else 10

    OUT = os.path.join(os.path.dirname(__file__), 'outputs', f'final_{TAG}')
    os.makedirs(OUT, exist_ok=True)
    log_path = f'{OUT}/run.log'
    log_f = open(log_path, 'a')

    def log(msg):
        line = f'[{time.strftime("%H:%M:%S")}] {msg}'
        print(line)
        log_f.write(line + '\n'); log_f.flush()

    log(f'=== FINAL_RUN: N_IMG={N_IMG} Q_TOTAL={Q_TOTAL} lambda={LAM} mock={MOCK} ===')

    from robustbench.utils import load_model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log(f'Device: {device}')

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
        log(f'Loaded {mname}: {arch}')

    images, labels, counts = select_images(oracles, N_IMG)
    log(f'Images: {len(images)}  |  label dist: {counts}')

    snapshot_indices = sorted(set(
        int(round(x)) for x in np.linspace(0, len(images) - 1, N_SNAPSHOT_IMAGES)
    ))
    log(f'Snapshot images (checkpointed): {snapshot_indices}')

    rows = []
    snap_records = []   # list of dict: image_idx, model, checkpoint, queries, l2, ssim, label
    snap_arrays  = {}    # key -> (C,H,W) float32 array
    t0 = time.time()

    for img_idx in range(len(images)):
        x_orig   = images[img_idx]
        y_true   = int(labels[img_idx])
        want_vis = img_idx in snapshot_indices
        img_t0   = time.time()

        for mname, _, _ in MODEL_SPECS:
            oracle_fn = oracles[mname]
            seed_base = img_idx * 1000
            t_start   = time.time()

            try:
                x_bnd, pl2, win_name, pq, dir_vecs = run_fixed_phase1(
                    oracle_fn, x_orig, y_true, seed=seed_base)
                if x_bnd is None:
                    log(f'  img {img_idx} [{mname}]: Phase 1 failed to find a boundary, skipping')
                    continue

                phase1_metrics = all_metrics(x_bnd, x_orig)

                dvecs_list = list(dir_vecs.values())
                k_total    = len(dvecs_list)
                try:
                    basis = corruption_basis(dvecs_list)
                except ValueError:
                    log(f'  img {img_idx} [{mname}]: no valid basis vectors, skipping')
                    continue

                q_phase3 = max(10, Q_TOTAL - pq)
                res = run_sep_cmaes_lam56(
                    oracle_fn, x_orig, y_true, basis, x_bnd, q_phase3,
                    seed=seed_base + 1,
                    snapshot_fracs=SNAP_FRACS if want_vis else None)

                final_label = oracle_fn(res['final_img'])
                final_metrics = all_metrics(res['final_img'], x_orig)
                runtime_sec = time.time() - t_start

                ir_p3 = (pl2 - res['best_l2']) / pl2 if pl2 > 0 else 0.0
                rows.append(dict(
                    model=mname, image_idx=img_idx, y_true=y_true, k_total=k_total,
                    winning_corruption=win_name,
                    phase1_l2=pl2, phase1_queries=pq,
                    phase1_ssim=phase1_metrics['ssim'], phase1_mse=phase1_metrics['mse'],
                    phase1_linf=phase1_metrics['linf'],
                    best_l2=res['best_l2'], final_l2=res['final_dist'],
                    final_label=final_label,
                    final_ssim=final_metrics['ssim'], final_mse=final_metrics['mse'],
                    final_linf=final_metrics['linf'],
                    IR_phase3=ir_p3, n_gens=res['n_gens'], queries_phase3=res['queries_used'],
                    q_offspring_eval=res['qcount']['offspring_eval'],
                    q_xi_shrink=res['qcount']['xi_shrink'],
                    q_theta_bs=res['qcount']['theta_bs'],
                    runtime_sec=runtime_sec,
                    l2_history=res['l2_history'], queries_history=res['queries_history'],
                ))

                if want_vis and res.get('snapshots'):
                    checkpoint_names = ['p3_0', 'p3_25', 'p3_50', 'p3_75', 'p3_100']
                    # Original (no query needed, selection already guarantees correct classification)
                    snap_records.append(dict(image_idx=img_idx, model=mname, checkpoint='original',
                                              queries=0, l2=0.0, ssim=1.0, label=y_true, true_label=y_true))
                    snap_arrays[f'{img_idx}_{mname}_original'] = x_orig

                    bnd_metrics = phase1_metrics
                    bnd_label = oracle_fn(x_bnd)
                    snap_records.append(dict(image_idx=img_idx, model=mname, checkpoint='phase1_boundary',
                                              queries=pq, l2=pl2, ssim=bnd_metrics['ssim'],
                                              label=bnd_label, true_label=y_true))
                    snap_arrays[f'{img_idx}_{mname}_phase1_boundary'] = x_bnd

                    for cp_name, (q, l2v, img_arr) in zip(checkpoint_names, res['snapshots']):
                        lbl = oracle_fn(img_arr)
                        sm  = all_metrics(img_arr, x_orig)
                        snap_records.append(dict(image_idx=img_idx, model=mname, checkpoint=cp_name,
                                                  queries=pq + q, l2=l2v, ssim=sm['ssim'],
                                                  label=lbl, true_label=y_true))
                        snap_arrays[f'{img_idx}_{mname}_{cp_name}'] = img_arr

            except Exception as e:
                log(f'  img {img_idx} [{mname}]: UNEXPECTED ERROR ({type(e).__name__}: {e}), skipping')
                continue

        if (img_idx + 1) % 25 == 0:
            pd.DataFrame(rows).to_parquet(f'{OUT}/results.parquet.tmp', index=False)
            if snap_records:
                pd.DataFrame(snap_records).to_csv(f'{OUT}/snapshots_meta.csv.tmp', index=False)
                np.savez_compressed(f'{OUT}/snapshots.npz.tmp', **snap_arrays)
            log(f'  checkpoint saved at img {img_idx+1} ({len(rows)} rows so far)')

        elapsed = time.time() - t0
        if (img_idx + 1) % 5 == 0 or img_idx < 3:
            per_img = elapsed / (img_idx + 1)
            eta = per_img * (len(images) - img_idx - 1)
            log(f'  img {img_idx+1:4d}/{len(images)}  ({elapsed:.0f}s elapsed, '
                f'{per_img:.1f}s/img, ETA {eta/60:.1f} min)')

    total_time = time.time() - t0
    log(f'Total time: {total_time:.1f}s  ({total_time/60:.1f} min)  |  {len(rows)} rows')

    df = pd.DataFrame(rows)
    df.to_parquet(f'{OUT}/results.parquet', index=False)
    log(f'Saved {OUT}/results.parquet')

    plain = df.drop(columns=['l2_history', 'queries_history'])
    plain.to_csv(f'{OUT}/results_flat.csv', index=False)

    summary = plain.groupby('model').agg(
        n=('best_l2', 'count'),
        mean_phase1_l2=('phase1_l2', 'mean'), median_phase1_l2=('phase1_l2', 'median'),
        mean_best_l2=('best_l2', 'mean'), median_best_l2=('best_l2', 'median'), std_best_l2=('best_l2', 'std'),
        mean_IR_phase3=('IR_phase3', 'mean'), median_IR_phase3=('IR_phase3', 'median'),
        mean_n_gens=('n_gens', 'mean'),
        mean_final_ssim=('final_ssim', 'mean'),
        mean_runtime_sec=('runtime_sec', 'mean'),
    ).round(4)
    summary.to_csv(f'{OUT}/summary.csv')
    log('\n=== Summary ===\n' + summary.to_string())

    if snap_records:
        snap_df = pd.DataFrame(snap_records)
        snap_df.to_csv(f'{OUT}/snapshots_meta.csv', index=False)
        np.savez_compressed(f'{OUT}/snapshots.npz', **snap_arrays)
        log(f'Saved {OUT}/snapshots_meta.csv and snapshots.npz ({len(snap_arrays)} images)')

    with open(f'{OUT}/run_config.json', 'w') as f:
        json.dump(dict(N_IMG=N_IMG, Q_TOTAL=Q_TOTAL, LAM=LAM, mock=MOCK,
                        snapshot_indices=snapshot_indices, total_time_sec=total_time,
                        device=device), f, indent=2)

    log('Done.')
    log_f.close()


if __name__ == '__main__':
    main()
