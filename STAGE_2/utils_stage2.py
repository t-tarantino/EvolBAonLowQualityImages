"""
Shared utilities for Stage 2 experiments.
All notebooks import from this module.
"""

import io
import warnings
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2471, 0.2435, 0.2616)
LINF_THRESHOLD = 0.10   # hard L-inf constraint on perturbation
SSIM_STOP      = 0.95   # early-stop Phase 3 when SSIM reaches this
H, W, C = 32, 32, 3
N = H * W * C           # pixel-space dimensionality (3072)
N_IMAGES   = 200        # full evaluation set
N_QUICK    = 30         # fast sanity-check set
RANDOM_SEED = 42
DATA_ROOT  = "../data"

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def compute_ssim(a: np.ndarray, b: np.ndarray) -> float:
    """SSIM between two HWC float32 images in [0, 1]."""
    return float(structural_similarity(a, b, data_range=1.0,
                                       channel_axis=2, win_size=7))


def compute_l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a.flatten() - b.flatten()))


def compute_linf(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a.flatten() - b.flatten())))


# ---------------------------------------------------------------------------
# Oracle
# ---------------------------------------------------------------------------

class Oracle:
    """
    Wraps a classifier, counts queries per phase, returns (is_adv, label).
    phase: 'p1' | 'p2' | 'p3'
    """
    def __init__(self, model, true_label: int, device: torch.device):
        self.model = model
        self.true_label = true_label
        self.device = device
        self.total = 0
        self.phase_count: dict = {"p1": 0, "p2": 0, "p3": 0}
        _m = torch.tensor(CIFAR10_MEAN, device=device).view(3, 1, 1)
        _s = torch.tensor(CIFAR10_STD,  device=device).view(3, 1, 1)
        self._mean = _m
        self._std  = _s

    def query(self, img_hwc: np.ndarray, phase: str = "p3"):
        x = torch.tensor(img_hwc.transpose(2, 0, 1),
                         dtype=torch.float32, device=self.device)
        x_norm = (x - self._mean) / self._std
        with torch.no_grad():
            lbl = self.model(x_norm.unsqueeze(0)).argmax(1).item()
        is_adv = (lbl != self.true_label)
        self.total += 1
        self.phase_count[phase] = self.phase_count.get(phase, 0) + 1
        return is_adv, lbl

    def reset_counts(self):
        self.total = 0
        self.phase_count = {"p1": 0, "p2": 0, "p3": 0}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_models(device: torch.device):
    """Load standard and robust WRN-28-10 via RobustBench."""
    from robustbench.utils import load_model as rb_load
    model_std = rb_load(model_name="Standard",
                        dataset="cifar10", threat_model="Linf")
    model_std = model_std.to(device).eval()

    model_rob = rb_load(model_name="Wang2023Better_WRN-28-10",
                        dataset="cifar10", threat_model="Linf")
    model_rob = model_rob.to(device).eval()
    return model_std, model_rob


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _predict(model, img_hwc: np.ndarray, device: torch.device) -> int:
    x = torch.tensor(img_hwc.transpose(2, 0, 1),
                     dtype=torch.float32, device=device)
    m = torch.tensor(CIFAR10_MEAN, device=device).view(3, 1, 1)
    s = torch.tensor(CIFAR10_STD,  device=device).view(3, 1, 1)
    with torch.no_grad():
        return model(((x - m) / s).unsqueeze(0)).argmax(1).item()


def get_jointly_correct(model_std, model_rob, device,
                        n: int = 200, seed: int = 42) -> list:
    """
    Return list of dicts {'idx', 'img' (HWC float32), 'label'}
    for images correctly classified by BOTH models.
    """
    dataset = torchvision.datasets.CIFAR10(
        root=DATA_ROOT, train=False,
        transform=T.ToTensor(), download=True)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(dataset))
    results = []
    for idx in order:
        if len(results) >= n:
            break
        img_t, label = dataset[int(idx)]
        img_hwc = img_t.numpy().transpose(1, 2, 0).astype(np.float32)
        if (_predict(model_std, img_hwc, device) == label and
                _predict(model_rob, img_hwc, device) == label):
            results.append({"idx": int(idx), "img": img_hwc, "label": label})
    return results


# ---------------------------------------------------------------------------
# Corruption family
# ---------------------------------------------------------------------------

def _jpeg(img_hwc: np.ndarray, s: float) -> np.ndarray:
    quality = max(5, int(100 - s * 95))
    pil = Image.fromarray((img_hwc * 255).clip(0, 255).astype(np.uint8))
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return np.array(Image.open(buf)).astype(np.float32) / 255.0


def _brightness(img_hwc: np.ndarray, s: float) -> np.ndarray:
    return clip01(img_hwc * (1.0 - 0.6 * s))


def _contrast(img_hwc: np.ndarray, s: float) -> np.ndarray:
    mean = img_hwc.mean(axis=(0, 1), keepdims=True)
    return clip01(mean + (1.0 - 0.8 * s) * (img_hwc - mean))


def _lf_noise(img_hwc: np.ndarray, s: float, rng: np.random.Generator) -> np.ndarray:
    noise = rng.standard_normal(img_hwc.shape).astype(np.float32)
    smooth = gaussian_filter(noise, sigma=4.0)
    smooth /= smooth.std() + 1e-8
    return clip01(img_hwc + s * 0.15 * smooth)


FAMILY = ["jpeg", "brightness", "contrast", "lf_noise"]


def make_corruption(name: str, img_hwc: np.ndarray, seed: int = 0):
    """Return a callable s -> corrupted_image for the given corruption type."""
    rng = np.random.default_rng(seed)
    fns = {
        "jpeg":       lambda s: _jpeg(img_hwc, s),
        "brightness": lambda s: _brightness(img_hwc, s),
        "contrast":   lambda s: _contrast(img_hwc, s),
        "lf_noise":   lambda s: _lf_noise(img_hwc, s, rng),
    }
    if name not in fns:
        raise ValueError(f"Unknown corruption: {name}")
    return fns[name]


def make_hybrid(img_hwc: np.ndarray, seed: int = 0):
    """Dirichlet-weighted linear combination of all four family members."""
    rng = np.random.default_rng(seed)
    weights = rng.dirichlet(np.ones(4))
    fns = [make_corruption(n, img_hwc, seed=seed + i)
           for i, n in enumerate(FAMILY)]
    deltas = [fn(1.0) - img_hwc for fn in fns]

    def apply(s):
        delta = sum(w * d for w, d in zip(weights, deltas))
        return clip01(img_hwc + s * delta)
    return apply


# ---------------------------------------------------------------------------
# Phase 1 — boundary initialization
# ---------------------------------------------------------------------------

def phase1(oracle: Oracle, img_hwc: np.ndarray,
           seed: int = 0, bs_steps: int = 12,
           exclude: set = None) -> tuple:
    """
    Try each family member (plus hybrid fallback).
    Returns (x_boundary_hwc, winner_name) or (None, None) if all fail.
    exclude: set of family names to skip (used in restart experiments).
    """
    exclude = exclude or set()
    candidates = []
    for i, name in enumerate(FAMILY):
        if name in exclude:
            continue
        fn = make_corruption(name, img_hwc, seed=seed + i)
        is_adv, _ = oracle.query(fn(1.0), phase="p1")
        if not is_adv:
            continue
        lo, hi = 0.0, 1.0
        for _ in range(bs_steps):
            mid = 0.5 * (lo + hi)
            is_adv_mid, _ = oracle.query(fn(mid), phase="p1")
            if is_adv_mid:
                hi = mid
            else:
                lo = mid
        x_bnd = clip01(fn(hi))
        oracle.query(x_bnd, phase="p1")
        candidates.append((compute_ssim(img_hwc, x_bnd), x_bnd, name))

    if not candidates:
        fn = make_hybrid(img_hwc, seed=seed)
        is_adv, _ = oracle.query(fn(1.0), phase="p1")
        if is_adv:
            lo, hi = 0.0, 1.0
            for _ in range(bs_steps):
                mid = 0.5 * (lo + hi)
                is_adv_mid, _ = oracle.query(fn(mid), phase="p1")
                if is_adv_mid:
                    hi = mid
                else:
                    lo = mid
            x_bnd = clip01(fn(hi))
            oracle.query(x_bnd, phase="p1")
            candidates.append((compute_ssim(img_hwc, x_bnd), x_bnd, "hybrid"))

    if not candidates:
        return None, None
    candidates.sort(reverse=True)
    return candidates[0][1], candidates[0][2]


# ---------------------------------------------------------------------------
# Phase 2 — binary search toward original
# ---------------------------------------------------------------------------

def phase2(oracle: Oracle, img_hwc: np.ndarray,
           x_adv: np.ndarray, bs_steps: int = 20) -> np.ndarray:
    """Interpolate x_adv toward img_hwc until boundary; return boundary point."""
    lo, hi = 0.0, 1.0
    for _ in range(bs_steps):
        t = 0.5 * (lo + hi)
        x_mid = clip01((1 - t) * img_hwc + t * x_adv)
        is_adv, _ = oracle.query(x_mid, phase="p2")
        if is_adv:
            hi = t
        else:
            lo = t
    x_bnd = clip01((1 - hi) * img_hwc + hi * x_adv)
    oracle.query(x_bnd, phase="p2")
    return x_bnd


# ---------------------------------------------------------------------------
# Subspace basis construction
# ---------------------------------------------------------------------------

def build_dct_basis(k_dct: int,
                    H: int = 32, W: int = 32, C: int = 3) -> np.ndarray:
    """DC-free 2-D DCT basis vectors, one per channel, sorted by frequency."""
    ys = np.arange(H, dtype=np.float64)
    xs = np.arange(W, dtype=np.float64)
    freq_pairs = sorted(
        [(fy, fx) for fy in range(H) for fx in range(W)
         if not (fy == 0 and fx == 0)],
        key=lambda t: t[0] + t[1],
    )
    vecs = []
    for ch in range(C):
        for fy, fx in freq_pairs:
            if len(vecs) >= k_dct:
                break
            v2d = (np.cos(np.pi * (2 * ys[:, None] + 1) * fy / (2 * H))
                   * np.cos(np.pi * (2 * xs[None, :] + 1) * fx / (2 * W)))
            v = np.zeros((H, W, C), dtype=np.float32)
            v[:, :, ch] = v2d.astype(np.float32)
            v_flat = v.flatten()
            nrm = np.linalg.norm(v_flat)
            if nrm > 1e-12:
                vecs.append(v_flat / nrm)
        if len(vecs) >= k_dct:
            break
    return np.array(vecs[:k_dct], dtype=np.float32)


def build_grid_basis(k_sp: int,
                     H: int = 32, W: int = 32, C: int = 3) -> np.ndarray:
    """Regular grid superpixel basis vectors."""
    k_side = int(np.ceil(np.sqrt(k_sp)))
    h_step = max(1, H // k_side)
    w_step = max(1, W // k_side)
    vecs = []
    for yi in range(0, H, h_step):
        for xi in range(0, W, w_step):
            v = np.zeros((H, W, C), dtype=np.float32)
            v[yi:yi + h_step, xi:xi + w_step, :] = 1.0
            v_flat = v.flatten()
            nrm = np.linalg.norm(v_flat)
            if nrm > 1e-12:
                vecs.append(v_flat / nrm)
    return np.array(vecs[:k_sp], dtype=np.float32)


def build_subspace(k_dct: int = 20, k_sp: int = 20,
                   H: int = 32, W: int = 32, C: int = 3) -> np.ndarray:
    """
    DC-free DCT + grid-superpixel basis (Stage 1 original).
    Excludes DC (fy=fx=0) to prevent large colour-shift artefacts.
    NOTE: brightness/contrast Phase-2 directions project poorly onto this
    basis (theta_m ≈ 0). Use build_subspace_with_dc for Phase 3 experiments.
    """
    B_dct = build_dct_basis(k_dct, H, W, C)
    B_sp  = build_grid_basis(k_sp,  H, W, C)
    B_raw = np.vstack([B_dct, B_sp]).astype(np.float64)
    Q, _ = np.linalg.qr(B_raw.T)
    B = Q.T[: B_raw.shape[0]].astype(np.float32)
    return B


def build_subspace_with_dc(k_dct: int = 20, k_sp: int = 20,
                            H: int = 32, W: int = 32, C: int = 3) -> np.ndarray:
    """
    DC-inclusive DCT + grid-superpixel basis for Phase 3 experiments.
    Includes the DC (fy=fx=0) component per channel so brightness/contrast
    Phase-2 directions project correctly: theta_m = B @ delta is non-zero
    for all corruption types, giving Phase 3 the right starting point.
    """
    ys = np.arange(H, dtype=np.float64)
    xs = np.arange(W, dtype=np.float64)
    freq_pairs = sorted(
        [(fy, fx) for fy in range(H) for fx in range(W)],  # DC included
        key=lambda t: t[0] + t[1],
    )
    vecs = []
    for ch in range(C):
        for fy, fx in freq_pairs:
            if len(vecs) >= k_dct:
                break
            v2d = (np.cos(np.pi * (2 * ys[:, None] + 1) * fy / (2 * H))
                   * np.cos(np.pi * (2 * xs[None, :] + 1) * fx / (2 * W)))
            v = np.zeros((H, W, C), dtype=np.float32)
            v[:, :, ch] = v2d.astype(np.float32)
            v_flat = v.flatten()
            nrm = np.linalg.norm(v_flat)
            if nrm > 1e-12:
                vecs.append(v_flat / nrm)
        if len(vecs) >= k_dct:
            break
    B_dct = np.array(vecs[:k_dct], dtype=np.float32)
    B_sp  = build_grid_basis(k_sp, H, W, C)
    B_raw = np.vstack([B_dct, B_sp]).astype(np.float64)
    Q, _  = np.linalg.qr(B_raw.T)
    return Q.T[:B_raw.shape[0]].astype(np.float32)


# ---------------------------------------------------------------------------
# Phase 3 — sep-CMA-ES in subspace (Stage 1 best configuration)
# ---------------------------------------------------------------------------

def lam_default(k: int) -> int:
    """CMA-ES default population size for k-dimensional problem."""
    return 4 + int(np.floor(3 * np.log(k)))


def sep_cmaes(oracle: Oracle, img_hwc: np.ndarray,
              x_adv_init: np.ndarray, B: np.ndarray,
              lam: int = 10, max_queries: int = 1000,
              ssim_stop: float = SSIM_STOP,
              stag_T: int = None) -> tuple:
    """
    Sep-CMA-ES in the k-dimensional subspace B.

    Returns
    -------
    best_x    : HWC float32 adversarial image
    best_l2   : float
    best_ssim : float
    queries   : int  (queries used by this call)
    history   : list of dicts {gen, queries, best_l2, best_ssim, sigma}
    stagnated : bool (only present when stag_T is not None)
    """
    k = B.shape[0]
    x0 = img_hwc.flatten()
    delta0 = x_adv_init.flatten() - x0
    theta_m = (B @ delta0).astype(np.float32)

    # CMA-ES parameters (k-dimensional search space)
    mu = max(1, lam // 2)
    w_raw = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1, dtype=np.float64))
    w = (w_raw / w_raw.sum()).astype(np.float32)
    mueff = float(1.0 / np.sum(w ** 2))
    cc    = (4 + mueff / k) / (k + 4 + 2 * mueff / k)
    cs    = (mueff + 2) / (k + mueff + 5)
    c1    = 2.0 / ((k + 1.3) ** 2 + mueff)
    cmu   = float(min(1 - c1,
                      2 * (mueff - 2 + 1 / mueff) / ((k + 2) ** 2 + mueff)))
    damps = 1 + 2 * max(0.0, np.sqrt((mueff - 1) / (k + 1)) - 1) + cs
    chiN  = k ** 0.5 * (1 - 1 / (4 * k) + 1 / (21 * k ** 2))

    D  = np.ones(k,  dtype=np.float32)
    pc = np.zeros(k, dtype=np.float32)
    ps = np.zeros(k, dtype=np.float32)
    sigma = float(max(0.1 * np.linalg.norm(delta0) / np.sqrt(k), 1e-5))

    best_x    = x_adv_init.copy()
    best_l2   = compute_l2(img_hwc, best_x)
    best_ssim = compute_ssim(img_hwc, best_x)

    history = []
    gen = 0
    queries = 0
    last_improvement = 0   # generation index of last improvement

    while queries < max_queries:
        batch = min(lam, max_queries - queries)
        if batch <= 0:
            break

        zs = np.random.randn(batch, k).astype(np.float32)
        thetas = theta_m[None, :] + sigma * D[None, :] * zs

        pool = []
        for i in range(batch):
            x_cand = clip01(x0 + B.T @ thetas[i]).reshape(H, W, C)
            is_adv, _ = oracle.query(x_cand, phase="p3")
            queries += 1
            if is_adv:
                l2 = compute_l2(img_hwc, x_cand)
                pool.append((l2, zs[i], thetas[i], x_cand))
                if l2 < best_l2:
                    best_l2 = l2
                    best_x  = x_cand.copy()
                    best_ssim = compute_ssim(img_hwc, best_x)
                    last_improvement = gen

        history.append({"gen": gen, "queries": queries,
                        "best_l2": best_l2, "best_ssim": best_ssim,
                        "sigma": sigma})

        if best_ssim >= ssim_stop:
            break

        if stag_T is not None and (gen - last_improvement) >= stag_T and gen > 0:
            return best_x, best_l2, best_ssim, queries, history, True

        if not pool:
            sigma = min(sigma * 2.0, 10.0)
            gen += 1
            continue

        pool.sort(key=lambda t: t[0])
        sel   = pool[:mu]
        n_sel = len(sel)
        w_s   = w[:n_sel] / w[:n_sel].sum()

        zs_sel = np.array([t[1] for t in sel])
        yw = (w_s[:, None] * zs_sel).sum(0)

        theta_m = theta_m + sigma * D * yw
        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * yw
        gen += 1
        hsig = ((np.linalg.norm(ps) / chiN
                 / np.sqrt(1 - (1 - cs) ** (2 * gen)))
                < (1.4 + 2 / (k + 1)))
        pc = ((1 - cc) * pc
              + hsig * np.sqrt(cc * (2 - cc) * mueff) * D * yw)

        wz2 = (w_s[:, None] * (zs_sel * D[None, :]) ** 2).sum(0)
        D = np.sqrt(np.clip(
            (1 - cmu - c1) * D ** 2
            + c1 * (pc ** 2 + (1 - hsig) * cc * (2 - cc) * D ** 2)
            + cmu * wz2,
            1e-20, 1e10,
        ))
        sigma = float(np.clip(
            sigma * np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1)),
            1e-10, 10.0,
        ))

    if stag_T is not None:
        return best_x, best_l2, best_ssim, queries, history, False
    return best_x, best_l2, best_ssim, queries, history


# ---------------------------------------------------------------------------
# Convenience: run full pipeline (Phase 1 + Phase 2 + Phase 3)
# ---------------------------------------------------------------------------

def run_pipeline(model, img_hwc: np.ndarray, label: int,
                 device: torch.device, B: np.ndarray,
                 phase3_fn=None, seed: int = 0) -> dict:
    """
    Run the full EvolBA pipeline on one image.
    phase3_fn: callable(oracle, img_hwc, x_adv_init, B) -> (best_x, best_l2, best_ssim, q, history)
               defaults to sep_cmaes with Stage 1 best config.
    Returns dict with metrics and query counts.
    """
    if phase3_fn is None:
        phase3_fn = lambda oracle, img, x0, B_: sep_cmaes(
            oracle, img, x0, B_, lam=10, max_queries=1000)

    oracle = Oracle(model, label, device)

    x_bnd, winner = phase1(oracle, img_hwc, seed=seed)
    if x_bnd is None:
        return None

    q_p1 = oracle.phase_count["p1"]
    x_bnd = phase2(oracle, img_hwc, x_bnd)
    q_p2 = oracle.phase_count["p2"]
    l2_p2 = compute_l2(img_hwc, x_bnd)

    result = phase3_fn(oracle, img_hwc, x_bnd, B)
    best_x, best_l2, best_ssim = result[0], result[1], result[2]
    q_p3 = oracle.phase_count["p3"]

    return {
        "winner":    winner,
        "l2_p2":     l2_p2,
        "l2_p3":     best_l2,
        "ssim_p3":   best_ssim,
        "linf_p3":   compute_linf(img_hwc, best_x),
        "improvement": l2_p2 - best_l2,
        "q_p1":      q_p1,
        "q_p2":      q_p2,
        "q_p3":      q_p3,
        "q_total":   oracle.total,
        "best_x":    best_x,
        "history":   result[4] if len(result) > 4 else [],
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_l2_curves(histories_dict: dict, title: str = "",
                   budget_marks: list = None):
    """
    histories_dict: {label: list_of_history_dicts}
    Each history_dict has keys 'queries' and 'best_l2'.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, histories in histories_dict.items():
        # Interpolate to common query grid
        q_max = max(h[-1]["queries"] for h in histories if h)
        q_grid = np.arange(1, q_max + 1)
        curves = []
        for hist in histories:
            if not hist:
                continue
            qs  = np.array([h["queries"]  for h in hist])
            l2s = np.array([h["best_l2"]  for h in hist])
            curves.append(np.interp(q_grid, qs, l2s))
        if not curves:
            continue
        arr = np.array(curves)
        med = np.median(arr, axis=0)
        p25 = np.percentile(arr, 25, axis=0)
        p75 = np.percentile(arr, 75, axis=0)
        ax.plot(q_grid, med, label=label)
        ax.fill_between(q_grid, p25, p75, alpha=0.15)

    if budget_marks:
        for bm in budget_marks:
            ax.axvline(bm, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("Queries (Phase 3)")
    ax.set_ylabel("Best L2")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    return fig


def summary_table(results_dict: dict, budget_marks: list = None) -> None:
    """Print a markdown-style summary table of L2 improvement."""
    header = f"{'Method':<35} {'median L2':>10} {'IQR':>10} {'SSIM':>8} {'q_p3 med':>10}"
    print(header)
    print("-" * len(header))
    for name, recs in results_dict.items():
        recs = [r for r in recs if r is not None]
        if not recs:
            print(f"{name:<35} {'n/a':>10}")
            continue
        l2s   = [r["l2_p3"]   for r in recs]
        ssims = [r["ssim_p3"] for r in recs]
        q3s   = [r["q_p3"]    for r in recs]
        med   = np.median(l2s)
        iqr   = np.percentile(l2s, 75) - np.percentile(l2s, 25)
        print(f"{name:<35} {med:>10.4f} {iqr:>10.4f} "
              f"{np.median(ssims):>8.4f} {np.median(q3s):>10.0f}")
