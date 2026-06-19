"""
phase1_zoo.py — Phase-1 initialisation zoo.

11 strategies in INIT_ZOO (used for Phase 1 boundary search):

  Group B — corruption-based:
      jpeg, blur, brightness, contrast, inversion, hue_shift, posterize,
      sharpen, saturation, gamma

  Group D — frequency-domain:
      fractal_random   paper's Algorithm 3 with a randomly-seeded synthetic
                       fractal (DFT frequency blending).

DIRECTION_ZOO — fixed-severity direction functions for Phase 3 subspace:
    For each corruption type, computes corrupt(x_orig, fixed_severity) so
    callers can derive d_j = corrupt(x_orig) - x_orig without any oracle
    queries.  All 11 INIT_ZOO types are represented.

All INIT_ZOO functions share the signature:
    fn(query, x_orig, y_true, rng) -> np.ndarray (float32, [0,1]) | None

All DIRECTION_ZOO functions share the signature:
    fn(x_orig) -> np.ndarray (float32, [0,1])
"""

from __future__ import annotations
import io
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.fft import dctn, idctn
from skimage.color import rgb2hsv, hsv2rgb
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from evolba_baseline import blend_frequencies, generate_fractal_image


# ── Shared severity-search helper ─────────────────────────────────────────────

def _severity_bs(query, x_orig, y_true, corrupt_fn, s_max: float, n_steps: int = 20):
    """
    Binary search in [0, s_max] for the *minimum* severity s such that
    corrupt_fn(x_orig, s) is adversarial.

    Returns corrupt_fn(x_orig, s_min_adv) or None if s_max doesn't flip.
    Uses exactly (1 + n_steps) oracle queries.
    """
    if query(corrupt_fn(x_orig, s_max)) == y_true:
        return None          # even maximum severity doesn't fool the model
    lo, hi = 0.0, float(s_max)
    for _ in range(n_steps):
        mid = 0.5 * (lo + hi)
        if query(corrupt_fn(x_orig, mid)) != y_true:
            hi = mid
        else:
            lo = mid
    return corrupt_fn(x_orig, hi).astype(np.float32)


# ── Group B: Corruption-based ─────────────────────────────────────────────────

def jpeg_init(query, x_orig, y_true, rng):
    """JPEG compression: severity 0 → quality 100 (near-lossless), severity 1 → quality 1 (maximal).
    Introduces block artefacts; effective when the model is sensitive to high-frequency texture."""
    def corrupt(x, severity):
        quality = max(1, int(100 * (1.0 - severity)))
        hwc = (np.clip(x, 0.0, 1.0) * 255).astype(np.uint8).transpose(1, 2, 0)
        buf = io.BytesIO()
        Image.fromarray(hwc).save(buf, format='JPEG', quality=quality)
        buf.seek(0)
        out = np.array(Image.open(buf)).transpose(2, 0, 1).astype(np.float32) / 255.0
        return out
    return _severity_bs(query, x_orig, y_true, corrupt, s_max=1.0)


def blur_init(query, x_orig, y_true, rng):
    """Gaussian blur, sigma 0 → 20. Destroys fine detail; effective when
    the model relies on high-frequency texture (standard model)."""
    def corrupt(x, sigma):
        out = np.stack([gaussian_filter(x[c].astype(np.float64), sigma)
                        for c in range(x.shape[0])])
        return np.clip(out, 0, 1).astype(np.float32)
    return _severity_bs(query, x_orig, y_true, corrupt, s_max=20.0)


def brightness_init(query, x_orig, y_true, rng):
    """Linear blend toward white (1.0), alpha 0 → 1."""
    def corrupt(x, alpha):
        return np.clip(x * (1 - alpha) + alpha, 0, 1).astype(np.float32)
    return _severity_bs(query, x_orig, y_true, corrupt, s_max=1.0)


def contrast_init(query, x_orig, y_true, rng):
    """Linear blend toward neutral grey (0.5), alpha 0 → 1."""
    def corrupt(x, alpha):
        return np.clip(x * (1 - alpha) + 0.5 * alpha, 0, 1).astype(np.float32)
    return _severity_bs(query, x_orig, y_true, corrupt, s_max=1.0)


def inversion_init(query, x_orig, y_true, rng):
    """Linear blend toward the color negative (1 - x), alpha 0 → 1.
    At alpha=1 the image is fully inverted; boundary typically reachable
    well before that."""
    def corrupt(x, alpha):
        return np.clip(x * (1 - alpha) + (1.0 - x) * alpha, 0, 1).astype(np.float32)
    return _severity_bs(query, x_orig, y_true, corrupt, s_max=1.0)


def hue_shift_init(query, x_orig, y_true, rng):
    """Rotate hue in HSV space, 0 → 180 degrees.  Preserves luminance and
    saturation; changes dominant object colour."""
    def corrupt(x, degrees):
        hwc = x.transpose(1, 2, 0)
        hsv = rgb2hsv(hwc.astype(np.float64))
        hsv[:, :, 0] = (hsv[:, :, 0] + degrees / 360.0) % 1.0
        out = hsv2rgb(hsv).transpose(2, 0, 1)
        return np.clip(out, 0, 1).astype(np.float32)
    return _severity_bs(query, x_orig, y_true, corrupt, s_max=180.0)


def posterize_init(query, x_orig, y_true, rng):
    """Reduce bit depth from 8 → 1 bit per channel (discrete steps).
    At 1 bit each pixel is pure black or white; try coarsest levels first."""
    for bits in range(1, 8):
        levels = 2 ** bits
        # Round each pixel to nearest level
        x_post = np.round(x_orig * (levels - 1)) / (levels - 1)
        x_post = np.clip(x_post, 0, 1).astype(np.float32)
        if query(x_post) != y_true:
            return x_post
    return None


# ── Group C: New corruptions ──────────────────────────────────────────────────

def sharpen_init(query, x_orig, y_true, rng):
    """Unsharp masking: x + alpha*(x - blur(x, sigma=1)), alpha 0 → 10.
    Amplifies high-frequency detail; complements blur."""
    def corrupt(x, alpha):
        blurred = np.stack([gaussian_filter(x[c].astype(np.float64), sigma=1.0)
                            for c in range(x.shape[0])])
        out = x.astype(np.float64) + alpha * (x.astype(np.float64) - blurred)
        return np.clip(out, 0, 1).astype(np.float32)
    return _severity_bs(query, x_orig, y_true, corrupt, s_max=10.0)


def saturation_init(query, x_orig, y_true, rng):
    """Multiply HSV saturation by (1 + factor), factor 0 → 4.
    Pushes colours toward their fully-saturated hue."""
    def corrupt(x, factor):
        hwc = x.transpose(1, 2, 0)
        hsv = rgb2hsv(hwc.astype(np.float64))
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.0 + factor), 0, 1)
        out = hsv2rgb(hsv).transpose(2, 0, 1)
        return np.clip(out, 0, 1).astype(np.float32)
    return _severity_bs(query, x_orig, y_true, corrupt, s_max=4.0)


def gamma_init(query, x_orig, y_true, rng):
    """Gamma correction: x → x^(1+severity), severity 0 → 4 (progressive darkening)."""
    def corrupt(x, severity):
        gamma = 1.0 + severity
        return np.clip(np.power(np.clip(x, 0, 1), gamma), 0, 1).astype(np.float32)
    return _severity_bs(query, x_orig, y_true, corrupt, s_max=4.0)


# ── Group D: Frequency-domain ─────────────────────────────────────────────────

def fractal_random_init(query, x_orig, y_true, rng):
    """
    Paper's Algorithm 3 with a randomly seeded synthetic fractal.

    Procedure (Eq. 5-8):
        f_orig    = DFT(x_orig)
        f_fractal = DFT(x_fractal)
        x_oi      = IDFT( lp(f_orig, r) + hp(f_fractal, r) )

    Starting from r = r_max (only DC preserved from fractal → result ≈ x_orig)
    down to r = 1 (almost entirely fractal texture), we stop and return x_oi at
    the first r where C(x_oi) ≠ C(x_orig).

    For CIFAR-10 (32×32) the effective radius range is ~1–23 (diagonal of the
    frequency grid). Each r value costs exactly 1 oracle query.
    """
    seed = int(rng.integers(0, 2**31))
    x_fractal = generate_fractal_image(x_orig.shape, seed)

    H, W = x_orig.shape[1], x_orig.shape[2]
    # Maximum radius that still excludes any frequency bin:
    # at r > diagonal, the low-pass mask covers the full spectrum → x_oi = x_orig.
    r_max = int(np.ceil(np.sqrt((H // 2) ** 2 + (W // 2) ** 2))) + 1

    for r in range(r_max, 0, -1):
        x_oi = blend_frequencies(x_orig, x_fractal, float(r))
        if query(x_oi) != y_true:
            return x_oi
    return None


def low_freq_rand_init(query, x_orig, y_true, rng):
    """
    Random DCT noise confined to the low-frequency band (ρ < 0.2, where
    ρ = normalised radial DCT frequency in [0, 1]).

    For CIFAR-10 (32×32) this band contains ~72 coefficients per channel
    (roughly the 8-cycle-per-image content); all higher coefficients are
    zeroed.  The result is a smooth, blob-like perturbation direction.

    Procedure:
        1. Sample random DCT coefficients, zero out ρ ≥ 0.2, IDCT → direction d.
        2. Binary search on magnitude s in [0, s_max] until
           C(clip(x_orig + s*d, 0,1)) ≠ C(x_orig).

    Study 0 showed this band needs median L2 ≈ 4.3 for the standard model,
    ≈ 9.3 for the robust model — still much better than uniform random init.
    """
    C, H, W = x_orig.shape
    fy = np.arange(H)[:, None] / H
    fx = np.arange(W)[None, :] / W
    rho = np.sqrt(fy ** 2 + fx ** 2) / np.sqrt(2)
    mask = (rho < 0.2).astype(np.float64)

    direction = np.empty_like(x_orig, dtype=np.float64)
    for c in range(C):
        coeffs = rng.standard_normal((H, W)) * mask
        direction[c] = idctn(coeffs, norm='ortho')
    direction /= (np.linalg.norm(direction) + 1e-12)

    def corrupt(x, magnitude):
        return np.clip(x.astype(np.float64) + magnitude * direction,
                       0, 1).astype(np.float32)

    return _severity_bs(query, x_orig, y_true, corrupt, s_max=25.0)


# ── Registry and multi-init factory ──────────────────────────────────────────

INIT_ZOO: dict = {
    'jpeg':           jpeg_init,
    'blur':           blur_init,
    'brightness':     brightness_init,
    'contrast':       contrast_init,
    'inversion':      inversion_init,
    'hue_shift':      hue_shift_init,
    'posterize':      posterize_init,
    'fractal_random': fractal_random_init,
    'sharpen':        sharpen_init,
    'saturation':     saturation_init,
    'gamma':          gamma_init,
}


# ── Direction functions (fixed-severity, no oracle queries) ───────────────────
# Used by Phase 3 to build the corruption subspace basis.
# Each function: (C,H,W) float32 → (C,H,W) float32 corrupted image.
# Callers compute d_j = direction_fn(x_orig) - x_orig.

def _jpeg_direction(x):
    hwc = (np.clip(x, 0, 1) * 255).astype(np.uint8).transpose(1, 2, 0)
    buf = io.BytesIO()
    Image.fromarray(hwc).save(buf, format='JPEG', quality=30)
    buf.seek(0)
    return np.array(Image.open(buf)).transpose(2, 0, 1).astype(np.float32) / 255.0

def _blur_direction(x):
    out = np.stack([gaussian_filter(x[c].astype(np.float64), sigma=10.0)
                    for c in range(x.shape[0])])
    return np.clip(out, 0, 1).astype(np.float32)

def _brightness_direction(x):
    return np.clip(x * 0.5 + 0.5, 0, 1).astype(np.float32)

def _contrast_direction(x):
    return np.clip(x * 0.5 + 0.25, 0, 1).astype(np.float32)

def _inversion_direction(x):
    return np.clip(x * 0.5 + (1.0 - x) * 0.5, 0, 1).astype(np.float32)

def _hue_shift_direction(x):
    hwc = x.transpose(1, 2, 0)
    hsv = rgb2hsv(hwc.astype(np.float64))
    hsv[:, :, 0] = (hsv[:, :, 0] + 90.0 / 360.0) % 1.0
    out = hsv2rgb(hsv).transpose(2, 0, 1)
    return np.clip(out, 0, 1).astype(np.float32)

def _posterize_direction(x):
    levels = 4  # 2 bits
    x_post = np.round(x * (levels - 1)) / (levels - 1)
    return np.clip(x_post, 0, 1).astype(np.float32)

def _fractal_direction(x):
    x_fractal = generate_fractal_image(x.shape, seed=0)
    H, W = x.shape[1], x.shape[2]
    r_mid = max(1, int(np.ceil(np.sqrt((H // 2) ** 2 + (W // 2) ** 2))) // 2)
    return blend_frequencies(x, x_fractal, float(r_mid))

def _sharpen_direction(x):
    blurred = np.stack([gaussian_filter(x[c].astype(np.float64), sigma=1.0)
                        for c in range(x.shape[0])])
    out = x.astype(np.float64) + 5.0 * (x.astype(np.float64) - blurred)
    return np.clip(out, 0, 1).astype(np.float32)

def _saturation_direction(x):
    hwc = x.transpose(1, 2, 0)
    hsv = rgb2hsv(hwc.astype(np.float64))
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 3.0, 0, 1)
    out = hsv2rgb(hsv).transpose(2, 0, 1)
    return np.clip(out, 0, 1).astype(np.float32)

def _gamma_direction(x):
    return np.clip(np.power(np.clip(x, 0, 1), 3.0), 0, 1).astype(np.float32)


DIRECTION_ZOO: dict = {
    'jpeg':           _jpeg_direction,
    'blur':           _blur_direction,
    'brightness':     _brightness_direction,
    'contrast':       _contrast_direction,
    'inversion':      _inversion_direction,
    'hue_shift':      _hue_shift_direction,
    'posterize':      _posterize_direction,
    'fractal_random': _fractal_direction,
    'sharpen':        _sharpen_direction,
    'saturation':     _saturation_direction,
    'gamma':          _gamma_direction,
}


def make_multi_init(x_orig: np.ndarray, binary_search_fn, bs_steps: int = 15):
    """
    Factory that returns an init_fn compatible with evolba_tuned's interface:
        init_fn(query, shape, y_true, rng) -> np.ndarray | None

    Internally tries every strategy in INIT_ZOO, projects each successful
    adversarial image onto the decision boundary (using binary_search_fn),
    and returns the boundary point with the smallest L2 distance to x_orig.

    Also returns a per-init L2 dict as a second element — callers that want
    the breakdown can unpack it; evolba_tuned ignores extra return values.
    """
    x_orig_flat = x_orig.flatten().astype(np.float64)

    def multi_init(query, shape, y_true, rng):
        best_x   = None
        best_l2  = float('inf')
        per_init = {}

        for name, fn in INIT_ZOO.items():
            x_adv = fn(query, x_orig, y_true, rng)
            if x_adv is None:
                per_init[name] = None
                continue
            x_bnd = binary_search_fn(query, x_adv, x_orig, y_true, n_steps=bs_steps)
            l2 = float(np.linalg.norm(x_bnd.flatten() - x_orig_flat))
            per_init[name] = l2
            if l2 < best_l2:
                best_l2 = l2
                best_x  = x_bnd

        return best_x   # evolba_tuned will run one more binary search on this point,
                        # which is harmless (it's already on the boundary).

    multi_init._per_init_breakdown = None   # populated at call-time if needed
    return multi_init
