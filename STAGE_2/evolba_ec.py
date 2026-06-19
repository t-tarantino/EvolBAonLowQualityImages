"""
evolba_ec.py -- Stage 2: Evolutionary-Computation variants of evolba_tuned's
Phase 3 (Sep-CMA-ES boundary-following).

Two new variants:

  evolba_vkd(..., vk_rank=k)
      Generalises the diagonal covariance D to  C = D (I + V V^T) D,  where
      V is an n x k matrix (k small, <= 3). k=0 reduces EXACTLY to
      evolba_tuned's sep-CMA-ES (same RNG draws, same D update, same
      directed-step / backtrack logic -- only the offspring-sampling
      transform changes, and it is the identity at k=0).

      V tracks the top-k directions of "excess" weighted variance among the
      current generation's offspring (i.e. directions where the population
      spread out more than an isotropic sampler would, beyond what the
      diagonal D already explains), blended across generations with its own
      learning rate `cv` (much faster than D's `cmu`, since V has only k
      "parameters" instead of n).

  evolba_one_plus_one(...)
      A genuinely different generation loop: 1 query/generation,
      accept-if-improves-and-adversarial, 1/5-success-rule step size
      (Igel/Suttorp/Hansen 2006 constants), rank-1 diagonal covariance
      update on success only. Elitist by construction (m's L2 is
      monotonically non-increasing), so best_l2 == final_l2 always.

Both share Phase 1 (initial adversarial point) and the binary-search /
objective building blocks with evolba_baseline.py.
"""

from __future__ import annotations
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'STAGE_1'))

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

# Cap on V's per-generation blending rate (see module docstring). V has only
# `vk_rank` "parameters" (vs n for D), so it can adapt far faster than D's
# cmu (~1e-3 for n=3072) without becoming unstable -- but cv=1 would mean
# "V = this generation's noisy estimate, no smoothing", so we cap it.
CV_CAP = 0.3


def _vkd_sample(zs: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Apply (I + V V^T)^{1/2} to each row of zs. V: (n, k). k=0 -> identity."""
    k = V.shape[1]
    if k == 0:
        return zs
    U, S, _ = np.linalg.svd(V, full_matrices=False)   # U: (n,k), S: (k,)
    factor  = np.sqrt(1.0 + S ** 2) - 1.0              # (k,)
    coeffs  = zs @ U                                   # (lam, k)
    return zs + (coeffs * factor) @ U.T


def _vkd_update(V: np.ndarray, zs_ranked: np.ndarray, w_eff: np.ndarray,
                cv: float) -> np.ndarray:
    """Blend V toward this generation's top-k directions of excess weighted
    variance. zs_ranked: zs[order] (offspring ranked best-first), already
    weighted by w_eff (i.e. pass zs_ranked = zs[order]).

    Self-normalising baseline: Y = zs_ranked * w_eff has shape (mu, n) with
    mu << n, so Y^T Y has rank <= mu in an n-dimensional space -- ALL mu of
    its nonzero eigenvalues are inflated by ~n/mu relative to the "no
    structure" population value 1/mueff, purely as a rank-deficiency
    artifact. Comparing each top eigenvalue to a fixed constant (e.g. 1)
    would flag this artifact as "excess" every generation. Instead compare
    each top eigenvalue to the MEAN of this same Y's mu eigenvalues -- the
    n/mu inflation cancels, and only genuine top-heaviness (relative to the
    other mu-1 directions estimated from the same sample) counts as excess.
    """
    k = V.shape[1]
    if k == 0 or cv == 0.0:
        return V
    Y = zs_ranked * w_eff[:, None]                     # (mu, n)
    k_avail = min(k, Y.shape[0])
    _, S_y, Vt_y = np.linalg.svd(Y, full_matrices=False)
    eig = S_y ** 2
    baseline = eig.mean()
    if baseline <= 0.0:
        return V
    V_cand = np.zeros_like(V)
    for i in range(k_avail):
        direction = Vt_y[i, :]
        if np.dot(direction, V[:, i]) < 0:
            direction = -direction
        excess = max(0.0, eig[i] - baseline)
        V_cand[:, i] = np.sqrt(excess / baseline) * direction
    return (1.0 - cv) * V + cv * V_cand


def evolba_vkd(
    oracle_fn,
    x_orig: np.ndarray,
    y_true: int,
    max_queries: int = config.MAX_Q_CIFAR10,
    *,
    vk_rank:       int         = 0,      # 0 = sep-CMA-ES (exact match to evolba_tuned)
    xi_step_scale: float       = 1.0,
    bs_steps:      int         = 15,
    bs_adaptive:   bool        = False,
    bs_cap:        int         = 26,
    collect_bs_pairs: list | None = None,
    collect_gen_info: list | None = None,
    tau:           int         = 1,
    lam_override:  int | None  = None,
    cmu_scale:     float       = 1.0,
    use_fractal_init: bool            = False,
    use_jump:         bool            = False,
    x_fractal:        np.ndarray | None = None,
    init_fn                           = None,
    mean_shift_fn                     = mean_shift_direction,
    seed: int                         = config.SEED,
) -> dict:
    """Sep-CMA-ES (vk_rank=0) / VkD-CMA (vk_rank>=1) boundary-following. See
    module docstring. vk_rank=0 is bit-for-bit identical to evolba_tuned()
    given the same other params (xi_correction's role is taken over by
    xi_step_scale, which evolba_tuned also supports)."""
    rng         = np.random.default_rng(seed)
    shape       = x_orig.shape
    n           = x_orig.size
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
                'bs_calls': 0, 'bs_queries_actual': 0, 'v_norms': []}

    x_bnd0  = do_binary_search(x0, x_orig, y_true)
    init_l2 = float(np.linalg.norm(x_bnd0.flatten() - x_orig_flat))

    # ── Phase 3: VkD-CMA boundary-following ────────────────────────────────────
    m   = x_bnd0.flatten().astype(np.float64)
    D   = np.ones(n, dtype=np.float64)
    V   = np.zeros((n, vk_rank), dtype=np.float64)
    lam = lam_override if lam_override is not None else (4 + int(3 * np.log(n)))
    mu  = lam
    weights, mueff = sep_cmaes_weights(mu)

    c1  = 2.0 / ((n + 1.3) ** 2 + mueff)
    cmu_raw = min(
        1.0 - c1,
        2.0 * (mueff - 2.0 + 1.0 / mueff) / ((n + 2.0) ** 2 + mueff),
    )
    cmu = cmu_raw * (n + 2.0) / 3.0 * cmu_scale

    cv = 0.0
    if vk_rank > 0:
        cv = min(CV_CAP, 2.0 * (mueff - 2.0 + 1.0 / mueff)
                 / ((vk_rank + 2.0) ** 2 + mueff)) * cmu_scale

    trajectory: list[tuple[int, float]] = []
    v_norms:    list[list[float]] = []
    jumped = False
    t = 1
    n_shrink_total    = 0
    n_backtrack_total = 0

    while queries[0] < max_queries:
        dist_to_orig = float(np.linalg.norm(m - x_orig_flat))

        xi      = dist_to_orig / np.sqrt(t)
        xi_step = xi * xi_step_scale

        # ── Sample lambda offspring (VkD transform applied before D scaling) ──
        zs = rng.standard_normal((lam, n))
        ys = _vkd_sample(zs, V)
        xs = np.clip(m + xi * D * ys, 0.0, 1.0)

        labels = np.empty(lam, dtype=np.int64)
        l2s    = np.empty(lam, dtype=np.float64)
        for k_ in range(lam):
            x_cand    = xs[k_].reshape(shape).astype(np.float32)
            labels[k_] = query(x_cand)
            l2s[k_]    = np.linalg.norm(xs[k_] - x_orig_flat)
            if queries[0] >= max_queries:
                lam_eff = k_ + 1
                zs, xs, labels, l2s = (
                    zs[:lam_eff], xs[:lam_eff], labels[:lam_eff], l2s[:lam_eff]
                )
                break

        is_adv  = labels != y_true
        fitness = np.array([objective(l2, adv) for l2, adv in zip(l2s, is_adv)])

        # ── Update mean-shift direction, diagonal D, and V ──────────────────────
        w_eff = weights[:len(fitness)] / weights[:len(fitness)].sum()
        v = mean_shift_fn(zs, fitness, is_adv, w_eff)
        D = update_diagonal_covariance(D, zs, fitness, w_eff, cmu)
        if vk_rank > 0:
            order = np.argsort(fitness)
            V = _vkd_update(V, zs[order], w_eff, cv)
        v_norms.append([float(np.linalg.norm(V[:, i])) for i in range(vk_rank)])

        # ── xi-shrink: halve xi_step until directed move stays adversarial ──────
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

        n_backtrack_total += backtracks
        m = m_new
        trajectory.append((queries[0], float(np.linalg.norm(m - x_orig_flat))))

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
        'v_norms'          : v_norms,
    }


def evolba_one_plus_one(
    oracle_fn,
    x_orig: np.ndarray,
    y_true: int,
    max_queries: int = config.MAX_Q_CIFAR10,
    *,
    bs_steps:      int         = 15,
    bs_adaptive:   bool        = True,
    bs_cap:        int         = 26,
    collect_bs_pairs: list | None = None,
    collect_gen_info: list | None = None,
    use_fractal_init: bool            = False,
    use_jump:         bool            = False,
    x_fractal:        np.ndarray | None = None,
    init_fn                           = None,
    seed: int                         = config.SEED,
) -> dict:
    """(1+1)-CMA-ES boundary-following.

    Each generation: sample ONE offspring, query once. If it is adversarial
    AND closer to x_orig than the current point, accept it (project to the
    boundary via binary search, costing extra queries), update the diagonal
    covariance with a rank-1 term, and grow sigma. Otherwise reject (zero
    extra queries) and shrink sigma. Constants (p_target, c_p, d, c_cov) are
    the standard (1+1)-CMA-ES values from Igel/Suttorp/Hansen (2006).

    Elitist by construction: m's L2 is monotonically non-increasing, so
    best_l2 == final_l2 always.
    """
    rng         = np.random.default_rng(seed)
    shape       = x_orig.shape
    n           = x_orig.size
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
                'bs_calls': 0, 'bs_queries_actual': 0, 'v_norms': [],
                'n_successes': 0, 'sigma_trajectory': [], 'p_succ_trajectory': []}

    x_bnd0  = do_binary_search(x0, x_orig, y_true)
    init_l2 = float(np.linalg.norm(x_bnd0.flatten() - x_orig_flat))

    # ── Phase 3: (1+1)-CMA-ES ────────────────────────────────────────────────────
    m     = x_bnd0.flatten().astype(np.float64)
    D     = np.ones(n, dtype=np.float64)
    # sigma0 = init_l2/sqrt(n) would make E[||step||] ~ init_l2, i.e.
    # E[l2_cand^2] = dist^2 + sigma^2*n = 2*dist^2 -- a "typical" step ALWAYS
    # increases L2 in expectation (isotropic noise adds variance regardless
    # of direction), so l2_cand < dist_to_orig is essentially unreachable.
    # sigma0 = init_l2/n makes E[l2_cand^2] ~ dist^2*(1+1/n) -- a small,
    # roughly-neutral step where improvement is achievable.
    sigma = init_l2 / n

    # Igel/Suttorp/Hansen (2006) constants for (1+1)-CMA-ES
    p_target = 2.0 / 11.0
    c_p      = 1.0 / 12.0
    d_damp   = 1.0 + n / 2.0
    c_cov    = 2.0 / (n ** 2 + 6.0)
    p_succ   = p_target

    trajectory: list[tuple[int, float]] = []
    sigma_trajectory: list[float] = []
    p_succ_trajectory: list[float] = []
    jumped = False
    n_successes = 0

    while queries[0] < max_queries:
        dist_to_orig = float(np.linalg.norm(m - x_orig_flat))

        z      = rng.standard_normal(n)
        x_cand = np.clip(m + sigma * D * z, 0.0, 1.0)
        label  = query(x_cand.reshape(shape).astype(np.float32))
        l2_cand = float(np.linalg.norm(x_cand - x_orig_flat))
        success = (label != y_true) and (l2_cand < dist_to_orig)

        if success:
            n_successes += 1
            m_new = do_binary_search(
                x_cand.reshape(shape).astype(np.float32), x_orig, y_true,
            ).flatten().astype(np.float64)
            # Elitist guard: binary_search always returns an adversarial
            # point on the segment from x_orig towards x_cand, so it is
            # guaranteed to be at least as close as x_cand was; m only moves
            # if this is also no farther than the current m.
            if np.linalg.norm(m_new - x_orig_flat) <= dist_to_orig:
                m = m_new
            D2 = (1.0 - c_cov) * D ** 2 + c_cov * z ** 2
            D  = np.sqrt(np.clip(D2, 1e-20, 1e10))

        p_succ = (1.0 - c_p) * p_succ + c_p * (1.0 if success else 0.0)
        sigma  = sigma * np.exp((1.0 / d_damp) * (p_succ - p_target) / (1.0 - p_target))

        trajectory.append((queries[0], float(np.linalg.norm(m - x_orig_flat))))
        sigma_trajectory.append(float(sigma))
        p_succ_trajectory.append(float(p_succ))

        if use_jump and not jumped and queries[0] >= JUMP_EVERY:
            m_img = jump_operator(m.reshape(shape).astype(np.float32), x_fractal)
            m     = m_img.flatten().astype(np.float64)
            jumped = True

    final_l2 = float(np.linalg.norm(m - x_orig_flat))
    best_l2  = min([init_l2] + [l2 for _, l2 in trajectory])

    return {
        'success'      : True,
        'queries'      : queries[0],
        'best_l2'      : best_l2,
        'final_l2'     : final_l2,
        'init_l2'      : init_l2,
        'trajectory'   : trajectory,
        'shrink_iters' : 0,
        'backtracks'   : 0,
        'n_generations': len(trajectory),
        'bs_calls'         : bs_calls[0],
        'bs_queries_actual': bs_queries_actual[0],
        'v_norms'          : [],
        'n_successes'      : n_successes,
        'sigma_trajectory' : sigma_trajectory,
        'p_succ_trajectory': p_succ_trajectory,
    }
