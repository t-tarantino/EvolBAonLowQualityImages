"""
evolba_tuned.py — EvolBA with five tunable parameters exposed.

Changes from evolba_baseline.py:

  xi_correction  decouple exploration scale from directed-step scale.
                 The baseline uses xi for both offspring sampling (scale xi·√n)
                 and the mean-shift step (scale xi), a ~55x mismatch for CIFAR-10.
                 When True, the directed step uses xi_step = xi/√n instead.

  bs_steps       binary-search precision. Baseline hardcodes 26 steps, which
                 exceeds float32 precision (saturates at ~23). 15 is sufficient.

  tau            max backtracks per generation. Baseline = 3. After fixing the
                 xi mismatch the directed step rarely overshoots, so 1 suffices.

  lam_override   population size. Baseline = 4+floor(3·ln n) = 28 for CIFAR-10.
                 Fewer offspring → more generations in same budget (noisier direction).
                 More offspring → better direction estimate (fewer generations).

  cmu_scale      covariance learning rate multiplier. Baseline = 1.0 (paper tuned
                 for VGG19 at 224×224, n≈150k). CIFAR-10 has n=3072; needs retuning.

All other algorithm logic is identical to evolba_baseline.py.
Import the shared building blocks from there — do not duplicate them.
"""

from __future__ import annotations
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import config
from evolba_baseline import (
    objective,
    binary_search,
    binary_search_adaptive,
    generate_fractal_image,
    uniform_random_init,
    fractal_init,
    jump_operator,
    sep_cmaes_weights,
    mean_shift_direction,
    update_diagonal_covariance,
    JUMP_EVERY,
)


def evolba_tuned(
    oracle_fn,
    x_orig: np.ndarray,
    y_true: int,
    max_queries: int = config.MAX_Q_CIFAR10,
    *,
    # ── tunable parameters ─────────────────────────────────────────────────────
    xi_correction: bool        = True,   # fix exploration/step scale mismatch
    xi_step_scale: float | None = None,  # if set, xi_step = xi * xi_step_scale
                                          # (overrides xi_correction; 1.0 == old
                                          # xi_correction=False, 1/sqrt(n) == old True)
    bs_steps:      int         = 15,     # binary-search steps per call
    bs_adaptive:   bool        = False,  # if True, use binary_search_adaptive(n_steps=bs_cap)
                                          # instead of fixed binary_search(bs_steps) -- stops
                                          # early once the bisection saturates float32
                                          # precision (lossless, see binary_search_adaptive)
    bs_cap:        int         = 26,     # step cap when bs_adaptive=True
    collect_bs_pairs: list | None = None,  # if a list, every binary_search call appends
                                            # (x_adv, x_orig, y_true) to it (offline validation)
    collect_gen_info: list | None = None,  # if a list, every generation appends a dict
                                            # {dist_to_orig, l2_pre_backtrack,
                                            #  l2_post_backtrack, backtracks}
                                            # (regression/recovery analysis)
    tau:           int         = 1,      # max backtracks per generation
    lam_override:  int | None  = None,   # None → auto (28 for CIFAR-10)
    cmu_scale:     float       = 0.1,    # covariance lr multiplier
    # ── unchanged flags ────────────────────────────────────────────────────────
    use_fractal_init: bool            = False,
    use_jump:         bool            = False,
    x_fractal:        np.ndarray | None = None,
    init_fn                           = None,
    mean_shift_fn                     = mean_shift_direction,
    seed: int                         = config.SEED,
) -> dict:
    """
    Run tuned EvolBA on one image.

    Returns
    -------
    dict with keys:
        success          bool
        queries          int   — total oracle calls used
        best_l2          float — L2 of final boundary point
        init_l2          float — L2 of Phase-1 boundary point
        trajectory       list of (query_count: int, l2: float) — one per generation
    """
    rng         = np.random.default_rng(seed)
    shape       = x_orig.shape
    n           = x_orig.size
    sqrt_n      = float(np.sqrt(n))
    x_orig_flat = x_orig.flatten().astype(np.float64)

    queries = [0]
    def query(img: np.ndarray) -> int:
        queries[0] += 1
        return oracle_fn(img)

    bs_queries_actual = [0]
    bs_calls          = [0]
    def do_binary_search(x_adv_img, x_orig_img, y_true_):
        if collect_bs_pairs is not None:
            collect_bs_pairs.append((x_adv_img.copy(), x_orig_img.copy(), int(y_true_)))
        if bs_adaptive:
            hi, n_actual = binary_search_adaptive(
                query, x_adv_img, x_orig_img, y_true_, n_steps=bs_cap)
            bs_queries_actual[0] += n_actual
        else:
            hi = binary_search(query, x_adv_img, x_orig_img, y_true_, n_steps=bs_steps)
            bs_queries_actual[0] += bs_steps
        bs_calls[0] += 1
        return hi

    if x_fractal is None and (use_fractal_init or use_jump):
        x_fractal = generate_fractal_image(shape, seed)

    # ── Phase 1: initial adversarial example ───────────────────────────────────
    if init_fn is not None:
        x0 = init_fn(query, shape, y_true, rng)
    elif use_fractal_init:
        x0 = fractal_init(query, x_orig, y_true, x_fractal)
        if x0 is None:
            x0 = uniform_random_init(query, shape, y_true, rng)
    else:
        x0 = uniform_random_init(query, shape, y_true, rng)

    if x0 is None:
        return {'success': False, 'queries': queries[0],
                'best_l2': float('nan'), 'final_l2': float('nan'),
                'init_l2': float('nan'), 'trajectory': [],
                'shrink_iters': 0, 'backtracks': 0, 'n_generations': 0,
                'bs_calls': 0, 'bs_queries_actual': 0}

    # Project initial adversarial onto boundary
    x_bnd0  = do_binary_search(x0, x_orig, y_true)
    init_l2 = float(np.linalg.norm(x_bnd0.flatten() - x_orig_flat))

    # ── Phase 3: Sep-CMA-ES ────────────────────────────────────────────────────
    m   = x_bnd0.flatten().astype(np.float64)
    D   = np.ones(n, dtype=np.float64)          # isotropic start
    lam = lam_override if lam_override is not None else (4 + int(3 * np.log(n)))
    mu  = lam
    weights, mueff = sep_cmaes_weights(mu)

    c1  = 2.0 / ((n + 1.3) ** 2 + mueff)
    cmu_raw = min(
        1.0 - c1,
        2.0 * (mueff - 2.0 + 1.0 / mueff) / ((n + 2.0) ** 2 + mueff),
    )
    cmu = cmu_raw * (n + 2.0) / 3.0 * cmu_scale

    trajectory: list[tuple[int, float]] = []
    jumped = False
    t = 1
    n_shrink_total    = 0   # total ξ-shrink halvings across all generations
    n_backtrack_total = 0   # total backtracks across all generations

    while queries[0] < max_queries:
        dist_to_orig = float(np.linalg.norm(m - x_orig_flat))

        # Eq. 4 — exploration scale (offspring sampling, unchanged)
        xi = dist_to_orig / np.sqrt(t)

        # Directed-step scale — decoupled when xi_correction=True.
        # Without correction: xi_step == xi → same 55x mismatch as baseline.
        # With correction: xi_step = xi/√n → step magnitude ≈ per-pixel distance,
        # matching the scale of the actual boundary-following move.
        if xi_step_scale is not None:
            xi_step = xi * xi_step_scale
        else:
            xi_step = xi / sqrt_n if xi_correction else xi

        # ── Sample λ offspring and evaluate ────────────────────────────────────
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
                zs, xs, labels, l2s = (
                    zs[:lam_eff], xs[:lam_eff], labels[:lam_eff], l2s[:lam_eff]
                )
                break

        is_adv  = labels != y_true
        fitness = np.array([objective(l2, adv) for l2, adv in zip(l2s, is_adv)])

        # ── Update mean-shift direction and diagonal covariance ────────────────
        w_eff = weights[:len(fitness)] / weights[:len(fitness)].sum()
        v = mean_shift_fn(zs, fitness, is_adv, w_eff)
        D = update_diagonal_covariance(D, zs, fitness, w_eff, cmu)

        # ── ξ-shrink: halve xi_step until directed move stays adversarial ──────
        m_shifted = np.clip(m + xi_step * v, 0.0, 1.0)
        while query(m_shifted.reshape(shape).astype(np.float32)) == y_true:
            xi_step /= 2.0
            n_shrink_total += 1
            m_shifted = np.clip(m + xi_step * v, 0.0, 1.0)
            if queries[0] >= max_queries:
                break

        # ── Binary search to boundary ──────────────────────────────────────────
        m_new = do_binary_search(
            m_shifted.reshape(shape).astype(np.float32), x_orig, y_true,
        ).flatten().astype(np.float64)
        l2_pre_backtrack = float(np.linalg.norm(m_new - x_orig_flat))

        # ── Backtrack if the move regressed ────────────────────────────────────
        backtracks = 0
        while (
            np.linalg.norm(m_new - x_orig_flat) > dist_to_orig
            and backtracks < tau
            and queries[0] < max_queries
        ):
            xi_step /= 2.0
            cand  = np.clip(m + xi_step * v, 0.0, 1.0).reshape(shape).astype(np.float32)
            m_new = do_binary_search(cand, x_orig, y_true).flatten().astype(np.float64)
            backtracks += 1

        l2_post_backtrack = float(np.linalg.norm(m_new - x_orig_flat))
        if collect_gen_info is not None:
            collect_gen_info.append(dict(
                dist_to_orig=dist_to_orig,
                l2_pre_backtrack=l2_pre_backtrack,
                l2_post_backtrack=l2_post_backtrack,
                backtracks=backtracks,
            ))

        n_backtrack_total += backtracks
        m = m_new
        trajectory.append((queries[0], float(np.linalg.norm(m - x_orig_flat))))

        # ── Optional jump ──────────────────────────────────────────────────────
        if use_jump and not jumped and queries[0] >= JUMP_EVERY:
            m_img = jump_operator(m.reshape(shape).astype(np.float32), x_fractal)
            m     = m_img.flatten().astype(np.float64)
            jumped = True

        t += 1

    final_l2 = float(np.linalg.norm(m - x_orig_flat))
    best_l2  = min([init_l2] + [l2 for _, l2 in trajectory])

    return {
        'success'      : True,
        'queries'      : queries[0],
        'best_l2'      : best_l2,
        'final_l2'     : final_l2,
        'init_l2'      : init_l2,
        'trajectory'   : trajectory,
        'shrink_iters' : n_shrink_total,
        'backtracks'   : n_backtrack_total,
        'n_generations': len(trajectory),
        'bs_calls'         : bs_calls[0],
        'bs_queries_actual': bs_queries_actual[0],
    }
