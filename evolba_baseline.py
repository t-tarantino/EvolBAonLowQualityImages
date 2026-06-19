"""
evolba_baseline.py — faithful implementation of EvolBA (Tajima & Ono, 2024,
arXiv:2407.02248), built for experimentation.

Why this file exists
--------------------
`evolba.py` is a *pragmatic* three-phase pipeline (noise init -> binary search
-> off-the-shelf CMA-ES) that only loosely resembles the paper. This file is
the opposite: it follows Algorithms 1-4 and Eqs. 1-10 of the paper as closely
as practical, so that experiments against it actually test the paper's ideas
— and so that *variants* of those ideas are a one-line change away.

Design philosophy: every conceptual ingredient of the paper is its own small,
pure-ish function (init, boundary search, mean-shift direction, step size,
covariance update, jump). `evolba_baseline()` just wires them together in the
order Algorithm 1 specifies. To try a variant:
  - swap one function for your own  (e.g. pass a different `mean_shift_fn`)
  - flip one of the `use_*` flags    (e.g. `use_jump=True` -> "EvolBA+J")
  - tweak one constant at the top of the file
No need to touch the orchestration loop itself.

Paper roadmap (so you can jump to the source of any piece below):
  Eq. 1-3        objective / penalty                  -> objective()
  Eq. 5-8, Alg.3 fractal initial-solution generation  -> fractal_init()
  Alg. 2         binary search to the boundary        -> binary_search()
  Eq. 9-10       custom mean-shift direction v^(t)    -> mean_shift_direction()
  Eq. 4          custom step size xi^(t)              -> inside evolba_baseline()
  Alg. 4         jump operator                        -> jump_operator()
  Alg. 1         main loop                            -> evolba_baseline()

The oracle convention matches `evolba.py::make_oracle`: a callable
`oracle_fn(image_chw: np.ndarray[float32, in [0,1]]) -> int` returning only
the predicted Top-1 label (hard-label black-box / HL-BB condition).
"""

from __future__ import annotations
import numpy as np

import config


# ─────────────────────────────────────────────────────────────────────────────
#  Constants taken directly from the paper's experimental setup (§IV-A).
#  Centralised here (not in config.py) so the whole algorithm stays in one
#  file you can read, tweak and fork without hunting across modules.
# ─────────────────────────────────────────────────────────────────────────────
C_PEN        = 1000.0   # penalty for non-adversarial candidates       (eq. 3)
BS_STEPS     = 26       # binary-search bisection steps                (§IV-A, "as in HSJA")
                        # Why 26: each bisection halves the interval, so 26 steps locates the
                        # boundary to within 2^-26 ≈ 1.5e-8 — just below float32's mantissa
                        # precision (2^-23 ≈ 1.2e-7). Further iterations move pixels by amounts
                        # that get rounded away in float32 storage, so 26 is the principled
                        # stopping point. Value adopted directly from HSJA (Chen et al., 2020).
TAU          = 3        # max step-size backtracks per generation      (Alg. 1, footnote)
INIT_CUTOFF  = 25       # DFT cutoff radius r for fractal init         (§IV-A)
JUMP_CUTOFF  = 50       # DFT cutoff radius r for the jump operator    (§IV-A)
JUMP_EVERY   = 1000     # queries elapsed before the (single) jump     (§IV-A, Pre-Exp 4)


# ─────────────────────────────────────────────────────────────────────────────
#  Objective function  (Eq. 1-3)
#
#  f(x) = ||x - x_orig|| + f_p(x),   f_p = c_pen if not adversarial else 0
#
#  Intuition: rank candidates by perturbation size, but make "not adversarial
#  at all" dominate any realistic distance — so the search never prefers a
#  tiny, useless perturbation over a slightly larger but *working* one.
# ─────────────────────────────────────────────────────────────────────────────

def objective(l2: float, is_adversarial: bool, c_pen: float = C_PEN) -> float:
    return l2 if is_adversarial else l2 + c_pen


# ─────────────────────────────────────────────────────────────────────────────
#  Frequency-domain blending  (Eq. 5-8 / Alg. 3 lines 2-5, reused by Alg. 4)
#
#  Both the fractal initialiser and the jump operator do the *same* trick:
#  keep an image's low frequencies (its coarse structure) but splice in
#  another image's high frequencies (its fine texture/edges). Idea 2 in the
#  paper is "CNNs are sensitive to high-frequency content", so grafting a
#  fractal's high-frequency texture onto the clean image is a cheap way to
#  manufacture something that looks similar but reads very differently to
#  the network.
# ─────────────────────────────────────────────────────────────────────────────

def _radial_lowpass_mask(h: int, w: int, cutoff: float) -> np.ndarray:
    """Disk of radius `cutoff` around the zero-frequency (DC) component."""
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= cutoff ** 2


def blend_frequencies(x_base: np.ndarray, x_donor: np.ndarray, cutoff: float) -> np.ndarray:
    """
    f_blend = lp(DFT(x_base); r) + hp(DFT(x_donor); r),  x_blend = IDFT(f_blend)

    Per channel: keep x_base inside the cutoff disk, take x_donor outside it.
    `cutoff` is the frequency-space radius r — large r keeps more of x_base,
    small r replaces more of it with x_donor's texture.
    """
    out = np.empty_like(x_base)
    h, w = x_base.shape[1:]
    lp_mask = _radial_lowpass_mask(h, w, cutoff)
    for c in range(x_base.shape[0]):
        f_base  = np.fft.fftshift(np.fft.fft2(x_base[c]))
        f_donor = np.fft.fftshift(np.fft.fft2(x_donor[c]))
        f_mixed = np.where(lp_mask, f_base, f_donor)
        out[c]  = np.real(np.fft.ifft2(np.fft.ifftshift(f_mixed)))
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def generate_fractal_image(shape_chw: tuple, seed: int, beta: float = 2.0) -> np.ndarray:
    """
    Stand-in for a FractalDB-1k sample (the paper uses a FIXED natural-fractal
    image — #5 or #7 of Fig. 4 — which isn't bundled with this repo). We
    synthesise "1/f^beta" coloured noise: inverse-FFT of a power-law spectrum
    with random phases. Such noise is *scale-invariant* (statistically self-
    similar at every zoom level) just like real fractals, so it keeps the
    high-pass-texture trick meaningful. Pass your own `x_fractal=` to
    `evolba_baseline` for a closer match to the paper (e.g. a real FractalDB
    image, or the corruption-family fractal already used in this repo).
    """
    rng = np.random.default_rng(seed)
    c, h, w = shape_chw
    fy, fx = np.fft.fftfreq(h)[:, None], np.fft.fftfreq(w)[None, :]
    radius = np.sqrt(fy ** 2 + fx ** 2)
    radius[0, 0] = radius[radius > 0].min()              # avoid 1/0 at DC
    amplitude = 1.0 / radius ** (beta / 2.0)

    img = np.empty(shape_chw, dtype=np.float32)
    for ch in range(c):
        phase    = rng.uniform(0.0, 2 * np.pi, size=(h, w))
        spectrum = amplitude * np.exp(1j * phase)
        plane    = np.real(np.fft.ifft2(spectrum))
        plane    = (plane - plane.min()) / (plane.max() - plane.min() + 1e-12)
        img[ch]  = plane
    return img


# ─────────────────────────────────────────────────────────────────────────────
#  Initial-solution generation
#
#  Two interchangeable strategies — pass whichever via `init_fn`:
#    * uniform_random_init  : Algorithm 1, line 2 (the algorithm's literal
#                             baseline — "EvolBA" in the paper's ablations)
#    * fractal_init         : Algorithm 3 / Idea 2  ("EvolBA+I..." variants)
# ─────────────────────────────────────────────────────────────────────────────

def uniform_random_init(query, shape, y_true, rng, max_attempts: int = 1000):
    """Alg. 1 line 2: keep drawing uniform-random images until one fools the
    model. Crude, slow to converge, but needs no domain knowledge whatsoever
    — the algorithm's true "cold start"."""
    for _ in range(max_attempts):
        x = rng.uniform(0.0, 1.0, size=shape).astype(np.float32)
        if query(x) != y_true:
            return x
    return None


def fractal_init(query, x_orig, y_true, x_fractal, r0: int = INIT_CUTOFF, r_min: int = 0):
    """
    Algorithm 3: repeatedly blend (lower the cutoff r each round) until the
    low-freq(x_orig) + high-freq(x_fractal) composite crosses the boundary.

    Intuition: start by keeping nearly all of x_orig (large r -> big disk of
    preserved low frequencies) and progressively graft in more of the
    fractal's high-frequency texture (shrinking r) until the classifier
    flips. Because we begin close to x_orig, the very first AE candidate is
    already close to the original image — a much better launchpad than
    uniform noise.
    """
    for r in range(r0, r_min - 1, -1):
        x_cand = blend_frequencies(x_orig, x_fractal, r)
        if query(x_cand) != y_true:
            return x_cand
    return None   # never crossed the boundary at any cutoff -> caller should fall back


# ─────────────────────────────────────────────────────────────────────────────
#  Binary search to the decision boundary  (Algorithm 2)
#
#  Standard bisection on the segment [x_orig, x_adv]. Used twice per
#  generation in the main loop: once to land the very first candidate on the
#  boundary, and once *every generation* to pull the mean back onto it
#  (the "Boundary Attack" half of "Sep-CMA-ES x BA").
# ─────────────────────────────────────────────────────────────────────────────

def binary_search(query, x_adv, x_orig, y_true, n_steps: int = BS_STEPS):
    """
    Invariant: `lo` stays correctly classified, `hi` stays adversarial.
    Returns `hi` — the adversarial point closest to x_orig found on the
    segment, within roughly ||x_adv - x_orig|| / 2**n_steps.
    """
    lo, hi = x_orig.copy(), x_adv.copy()
    for _ in range(n_steps):
        mid = np.clip(0.5 * (lo + hi), 0.0, 1.0).astype(np.float32)
        if query(mid) != y_true:
            hi = mid
        else:
            lo = mid
    return hi


def binary_search_adaptive(query, x_adv, x_orig, y_true, n_steps: int = BS_STEPS):
    """
    Same as binary_search, but stops early once `mid` rounds (in float32) to
    exactly `lo` or `hi` -- at that point the bracket can no longer shrink,
    so by the loop invariant `query(mid)` is guaranteed to equal `query(lo)`
    or `query(hi)` and the step would be a no-op. Skipping these saturated
    steps is lossless: `hi` is identical to what binary_search(n_steps)
    would return, just computed with <= n_steps queries.

    Returns (hi, n_actual) where n_actual <= n_steps is the number of
    queries actually spent.
    """
    lo, hi = x_orig.copy(), x_adv.copy()
    n_actual = 0
    for _ in range(n_steps):
        mid = np.clip(0.5 * (lo + hi), 0.0, 1.0).astype(np.float32)
        if np.array_equal(mid, lo) or np.array_equal(mid, hi):
            break
        if query(mid) != y_true:
            hi = mid
        else:
            lo = mid
        n_actual += 1
    return hi, n_actual


# ─────────────────────────────────────────────────────────────────────────────
#  Jump operator  (Algorithm 4 / Idea 3 — "EvolBA+J..." variants)
#
#  CMA-ES is a fundamentally local optimiser; left alone it can settle into
#  a mediocre region of the boundary. The jump operator gives the search a
#  one-off kick by grafting fractal high-frequency texture onto the *current
#  mean* — exactly the same blend used at init, just applied mid-search and
#  with a coarser cutoff (less disruptive: r=50 vs r=25).
# ─────────────────────────────────────────────────────────────────────────────

def jump_operator(m_img, x_fractal, cutoff: int = JUMP_CUTOFF):
    return blend_frequencies(m_img, x_fractal, cutoff)


# ─────────────────────────────────────────────────────────────────────────────
#  Sep-CMA-ES building blocks, customised per §III-F / Eq. 9-10
#
#  Standard Sep-CMA-ES recombines only the mu best offspring and moves the
#  mean toward them. EvolBA changes the policy because, under HL-BB, even
#  "failed" (non-adversarial) offspring carry information: they mark
#  directions that cross back into the clean region. So EvolBA
#    (a) uses ALL lambda offspring (mu = lambda), and
#    (b) NEGATES the sampled directions of non-adversarial ones,
#  turning "this direction failed" into "move away from this direction".
# ─────────────────────────────────────────────────────────────────────────────

def sep_cmaes_weights(mu: int) -> tuple[np.ndarray, float]:
    """Standard log-decreasing recombination weights w_1 > ... > w_mu,
    normalised to sum to 1, plus the resulting "effective sample size" mueff
    (used to scale the covariance learning rate)."""
    w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1, dtype=np.float64))
    w /= w.sum()
    mueff = 1.0 / np.sum(w ** 2)
    return w, mueff


def mean_shift_direction(zs: np.ndarray, fitness: np.ndarray, is_adv: np.ndarray,
                         weights: np.ndarray) -> np.ndarray:
    """
    Eq. 9-10:  v^(t) = [ sum_{i<=l} w_i z_{i:mu} - sum_{j>l} w_j z_{(mu-j+1):mu} ]
                       / || ... ||

    Read at the level of intuition rather than index gymnastics: rank ALL mu
    offspring by fitness (adversarial ones always rank above non-adversarial
    ones, because of the penalty), assign the usual decreasing weights by that
    *global* rank, but flip the sign of any non-adversarial offspring's
    direction. The result is a single unit vector that says "go this way,
    and also, don't go that other way".
    """
    order = np.argsort(fitness)                         # best (lowest f) first
    signs = np.where(is_adv[order], 1.0, -1.0)
    v     = (weights * signs)[:, None] * zs[order]
    v     = v.sum(axis=0)
    return v / (np.linalg.norm(v) + 1e-12)


def update_diagonal_covariance(D: np.ndarray, zs: np.ndarray, fitness: np.ndarray,
                               weights: np.ndarray, cmu: float) -> np.ndarray:
    """
    Rank-mu diagonal-covariance update (the dominant Sep-CMA-ES term once
    mu = lambda, as EvolBA sets it). We deliberately drop the rank-one /
    evolution-path machinery: that path is driven by CSA's accumulated
    sigma-history, and EvolBA — as explained in the report — replaces CSA
    outright with the deterministic geometric schedule xi^(t) (Eq. 4), so
    there is no natural evolution path left to accumulate. This is an
    explicit simplification; swap in your own update_*_fn to explore richer
    covariance schemes.

    Note z, not +-z: squaring erases the sign flip from mean_shift_direction,
    so non-adversarial offspring still teach the optimiser about *scale*
    (which axes are too wide / too narrow) even though their *direction* was
    discounted for the mean update.
    """
    order = np.argsort(fitness)
    z_w2  = (weights[:, None] * (zs[order] * D) ** 2).sum(axis=0)
    D2    = (1.0 - cmu) * D ** 2 + cmu * z_w2
    return np.sqrt(np.clip(D2, 1e-20, 1e10))


# ─────────────────────────────────────────────────────────────────────────────
#  Main loop  (Algorithm 1)
# ─────────────────────────────────────────────────────────────────────────────

def evolba_baseline(
    oracle_fn,
    x_orig: np.ndarray,
    y_true: int,
    max_queries: int = config.MAX_Q_CIFAR10,
    *,
    use_fractal_init: bool = False,    # False -> Alg.1 line 2 ("EvolBA")
    use_jump:         bool = False,    # True  -> Algorithm 4   ("...+J...")
    x_fractal:        np.ndarray | None = None,
    init_fn = None,                    # override: custom (query, shape, y, rng) -> x0
    mean_shift_fn = mean_shift_direction,
    cmu_scale: float = 1.0,            # paper's Pre-Exp 2 found x0.1 best for VGG19/224px;
                                       # CIFAR-10 is far lower-dimensional, so re-tune here
    seed: int = config.SEED,
) -> dict:
    """
    Run EvolBA on a single image, following Algorithm 1 line-for-line:

      1-5   build an initial AE candidate, project it onto the boundary,
            seed the Sep-CMA-ES mean and covariance there
      6-9   sample lambda offspring from N(m, xi^2 * D^2), evaluate them,
            derive the mean-shift direction v and update the covariance D
      10-14 pick a step size xi (Eq. 4), shrinking it until m + xi*v is
            still adversarial
      15-18 move the mean along the boundary by xi*v, then pull it back onto
            the boundary toward x_orig via binary search
      19-22 if that move ended up *farther* from x_orig than before, halve xi
            and retry — up to TAU times (the "backtracking" safeguard)

    Returns a dict with the final adversarial image, its L2 distance, the
    query count, and a per-generation L2 history for convergence plots —
    mirroring the dict shape returned by evolba() in evolba.py.
    """
    rng    = np.random.default_rng(seed)
    shape  = x_orig.shape
    n      = x_orig.size
    x_orig_flat = x_orig.flatten().astype(np.float64)

    # Every call to the model is metered here — this single counter is the
    # one source of truth for the query budget across every sub-routine.
    queries = [0]
    def query(img):
        queries[0] += 1
        return oracle_fn(img)

    # A fixed "donor" image for the frequency-blending tricks (init + jump).
    # The paper uses one fixed FractalDB-1k image throughout; we do the same
    # (generate once, reuse for both operators) for consistency.
    if x_fractal is None and (use_fractal_init or use_jump):
        x_fractal = generate_fractal_image(shape, seed)

    # ── lines 1-4: build x^(0), then project it onto the boundary ───────────
    if init_fn is not None:
        x0 = init_fn(query, shape, y_true, rng)
    elif use_fractal_init:
        x0 = fractal_init(query, x_orig, y_true, x_fractal)
        if x0 is None:                                   # never crossed -> fall back
            x0 = uniform_random_init(query, shape, y_true, rng)
    else:
        x0 = uniform_random_init(query, shape, y_true, rng)

    if x0 is None:
        return {'success': False, 'queries': queries[0]}

    x_tilde0 = binary_search(query, x0, x_orig, y_true)

    # ── line 5: initialise Sep-CMA-ES state at the boundary point ───────────
    m   = x_tilde0.flatten().astype(np.float64)
    D   = np.ones(n, dtype=np.float64)              # diagonal covariance, isotropic start
    lam = 4 + int(3 * np.log(n))                    # Hansen's CMA-ES default: λ = 4 + ⌊3 ln N⌋
                                                    # (Hansen et al., 2003 / ref [18] in paper)
                                                    # Derived from convergence theory: population
                                                    # scales logarithmically with dimension to
                                                    # balance exploration vs. query cost. Not a
                                                    # heuristic — recomputed from image shape each
                                                    # run, so it adapts automatically to the input
                                                    # resolution (e.g. N=3072 → λ=28 for CIFAR-10
                                                    # 32×32; N=150528 → λ=39 for VGG19 224×224).
    mu  = lam                                       # EvolBA: mu = lambda (use ALL offspring)
    weights, mueff = sep_cmaes_weights(mu)

    # Sep-CMA-ES rank-mu learning rate, Hansen's recommended formula scaled
    # by (N+2)/3 — the correction the paper applies "for the application to
    # Sep-CMA-ES" (compensating the diagonal-only parameterisation).
    c1  = 2.0 / ((n + 1.3) ** 2 + mueff)
    cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((n + 2.0) ** 2 + mueff))
    cmu = cmu * (n + 2.0) / 3.0 * cmu_scale

    gen_l2_history = []
    jumped = False
    t = 1   # NB: Eq. 4 has a 1/sqrt(t) singularity at t=0, so the geometric
            # schedule is taken to start at t=1 (m^(1) = m^(0) = boundary point)

    # ── lines 6-24: the generational loop ────────────────────────────────────
    while queries[0] < max_queries:
        dist_to_orig = float(np.linalg.norm(m - x_orig_flat))

        # Eq. 4 — the *only* step-size signal in EvolBA (no CSA at all): it
        # shrinks deterministically as the mean nears x_orig. It plays a dual
        # role here — the spread sigma of the sampling distribution below,
        # and (further down) the magnitude of the boundary-following move.
        xi = dist_to_orig / np.sqrt(t)

        # ── lines 7-8: sample & evaluate lambda offspring ───────────────────
        zs = rng.standard_normal((lam, n))
        xs = np.clip(m + xi * D * zs, 0.0, 1.0)

        labels  = np.empty(lam, dtype=np.int64)
        l2s     = np.empty(lam, dtype=np.float64)
        for k in range(lam):
            x_cand    = xs[k].reshape(shape).astype(np.float32)
            labels[k] = query(x_cand)
            l2s[k]    = np.linalg.norm(xs[k] - x_orig_flat)
            if queries[0] >= max_queries:
                lam_eff = k + 1               # ran out of budget mid-batch
                zs, xs, labels, l2s = zs[:lam_eff], xs[:lam_eff], labels[:lam_eff], l2s[:lam_eff]
                break

        is_adv  = labels != y_true
        fitness = np.array([objective(l2, adv) for l2, adv in zip(l2s, is_adv)])

        # ── line 9: update Sep-CMA-ES parameters — direction v and covariance D
        w_eff = weights[:len(fitness)] / weights[:len(fitness)].sum()
        v = mean_shift_fn(zs, fitness, is_adv, w_eff)
        D = update_diagonal_covariance(D, zs, fitness, w_eff, cmu)

        # ── lines 10-14: shrink xi until m + xi*v actually crosses the boundary
        m_shifted = np.clip(m + xi * v, 0.0, 1.0)
        while query(m_shifted.reshape(shape).astype(np.float32)) == y_true:
            xi /= 2.0
            m_shifted = np.clip(m + xi * v, 0.0, 1.0)
            if queries[0] >= max_queries:
                break

        # ── lines 15-18: move along the boundary, then pull back toward x_orig
        m_new = binary_search(query, m_shifted.reshape(shape).astype(np.float32),
                              x_orig, y_true)
        m_new = m_new.flatten().astype(np.float64)

        # ── lines 19-22: backtrack (halve xi, retry) if that move regressed —
        # i.e. landed *farther* from x_orig than m was. Capped at TAU tries so
        # a stubborn generation can't stall the whole search.
        backtracks = 0
        while (np.linalg.norm(m_new - x_orig_flat) > dist_to_orig
               and backtracks < TAU and queries[0] < max_queries):
            xi /= 2.0
            cand  = np.clip(m + xi * v, 0.0, 1.0).reshape(shape).astype(np.float32)
            m_new = binary_search(query, cand, x_orig, y_true).flatten().astype(np.float64)
            backtracks += 1

        m = m_new
        gen_l2_history.append(float(np.linalg.norm(m - x_orig_flat)))

        # ── Idea 3 / Algorithm 4: one-off jump once the query trigger fires ──
        if use_jump and not jumped and queries[0] >= JUMP_EVERY:
            m_img = jump_operator(m.reshape(shape).astype(np.float32), x_fractal)
            m     = m_img.flatten().astype(np.float64)
            jumped = True

        t += 1

    best_adv = m.reshape(shape).astype(np.float32)
    best_l2  = float(np.linalg.norm(m - x_orig_flat))

    return {
        'success'        : True,
        'queries'        : queries[0],
        'best_adv'       : best_adv,
        'best_l2'        : best_l2,
        'gen_l2_history' : gen_l2_history,
    }
