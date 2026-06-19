"""
subspace.py — structured search subspace construction.

Supports DCT basis, spatial-grid superpixel basis, and their combination.
These bases define lower-dimensional spaces in which Phase 3 optimizers
can work more efficiently than in raw pixel space (BRAINSTORMING §2).

All functions return (k, n) float64 arrays with orthonormal rows,
where n = C*H*W is the flat image dimension.
"""

from __future__ import annotations
import numpy as np


# ── DCT basis ─────────────────────────────────────────────────────────────────

def dct_basis(shape_chw: tuple, k: int) -> np.ndarray:
    """
    Top-k DCT-II basis vectors, sorted by increasing spatial frequency.

    Each basis vector corresponds to one (fy, fx, channel) triplet. Low-frequency
    (fy+fx small) components are selected first; they capture the perceptually
    important structure while remaining imperceptible. Channels are interleaved
    within each frequency (one vector per channel before moving to the next
    frequency) so that any k >= 1 spans all C channels rather than exhausting
    channel 0 alone for k <= H*W.

    Parameters
    ----------
    shape_chw : (C, H, W) image shape
    k         : number of basis vectors to return

    Returns
    -------
    basis : (k, C*H*W) orthonormal float64 array
    """
    C, H, W = shape_chw
    n = C * H * W

    ys = np.arange(H, dtype=np.float64)
    xs = np.arange(W, dtype=np.float64)

    # All (fy, fx) pairs sorted by Manhattan frequency
    freq_pairs = sorted(
        [(fy, fx) for fy in range(H) for fx in range(W)],
        key=lambda t: t[0] + t[1],
    )

    vecs = []
    for fy, fx in freq_pairs:
        if len(vecs) >= k:
            break
        # DCT-II basis function for frequency (fy, fx), shared across channels
        v2d = (
            np.cos(np.pi * (2 * ys[:, None] + 1) * fy / (2 * H))
            * np.cos(np.pi * (2 * xs[None, :] + 1) * fx / (2 * W))
        )
        for c in range(C):
            if len(vecs) >= k:
                break
            v = np.zeros(n, dtype=np.float64)
            v[c * H * W: (c + 1) * H * W] = v2d.flatten()
            norm = np.linalg.norm(v)
            if norm > 1e-12:
                vecs.append(v / norm)

    return np.array(vecs[:k], dtype=np.float64)


# ── Random basis ─────────────────────────────────────────────────────────────

def random_basis(shape_chw: tuple, k: int, seed: int = 0) -> np.ndarray:
    """
    k random orthonormal directions in pixel space (QR of a Gaussian matrix).

    Unstructured control basis: if a structured basis (DCT / superpixel) does
    not outperform this at the same k, the structure isn't adding value.

    Parameters
    ----------
    shape_chw : (C, H, W) image shape
    k         : number of basis vectors to return
    seed      : RNG seed (fixed per k so the same random subspace is reused
                across images/runs for a given k)

    Returns
    -------
    basis : (k, C*H*W) orthonormal float64 array
    """
    C, H, W = shape_chw
    n = C * H * W
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, k))
    Q, _ = np.linalg.qr(A)
    return Q.T


# ── Spatial-grid superpixel basis ─────────────────────────────────────────────

def grid_superpixel_basis(shape_chw: tuple, K: int) -> np.ndarray:
    """
    Spatial-grid superpixel basis: partition the image into a K-cell grid,
    each basis vector uniformly perturbs all pixels in one cell.

    This is the axis-aligned approximation to SLIC superpixels (BRAINSTORMING
    §2.2). Full SLIC (image-adaptive) is the preferred option for production
    but requires CIELAB conversion and iterative k-means; the grid version
    is model-agnostic, free to compute, and sufficient for initial experiments.

    Parameters
    ----------
    shape_chw : (C, H, W)
    K         : target number of cells (actual count may be slightly less)

    Returns
    -------
    basis : (K', C*H*W) orthonormal float64 array, K' ≤ K
    """
    C, H, W = shape_chw
    n = C * H * W

    k_side  = int(np.ceil(np.sqrt(K)))
    h_step  = max(1, H // k_side)
    w_step  = max(1, W // k_side)

    vecs = []
    for yi in range(0, H, h_step):
        for xi in range(0, W, w_step):
            v = np.zeros((C, H, W), dtype=np.float64)
            v[:, yi:yi + h_step, xi:xi + w_step] = 1.0
            flat = v.flatten()
            norm = np.linalg.norm(flat)
            if norm > 1e-12:
                vecs.append(flat / norm)

    basis = np.array(vecs[:K], dtype=np.float64)
    # Re-orthonormalize via QR (grid cells may slightly overlap at boundaries)
    Q, _ = np.linalg.qr(basis.T)
    return Q.T[: len(vecs)]


# ── Combined DCT + superpixel basis ───────────────────────────────────────────

def combined_basis(
    shape_chw: tuple,
    k_dct: int = 100,
    k_sp: int  = 32,
) -> np.ndarray:
    """
    Concatenate DCT and grid-superpixel bases, orthogonalize, and return the
    top-(k_dct + k_sp) combined vectors.

    DCT captures the frequency / imperceptibility prior.
    Superpixels capture the semantic / spatial prior.
    Together they span a richer space than either alone (BRAINSTORMING §2.3).

    Parameters
    ----------
    shape_chw : (C, H, W)
    k_dct     : DCT components
    k_sp      : superpixel components

    Returns
    -------
    basis : (k, C*H*W) orthonormal float64 array, k ≤ k_dct + k_sp
    """
    B_dct = dct_basis(shape_chw, k_dct)
    B_sp  = grid_superpixel_basis(shape_chw, k_sp)
    combined = np.vstack([B_dct, B_sp])
    Q, _ = np.linalg.qr(combined.T)
    k = min(combined.shape[0], Q.shape[1])
    return Q.T[:k]


# ── Corruption basis ─────────────────────────────────────────────────────────

def corruption_basis(direction_vecs: list[np.ndarray]) -> np.ndarray:
    """
    Basis built from Phase-1 corruption direction vectors.

    Each d_j = x_boundary_j.flatten() - x_orig.flatten() points from x_orig
    toward the decision boundary along corruption j's path.  The span of
    these vectors defines the adversarial subspace for that image — unlike DCT,
    the residual floor is near zero because x_b is already one of the d_j.

    Parameters
    ----------
    direction_vecs : list of (n,) float64 arrays, one per corruption type
                     that found an adversarial example in Phase 1.
                     Near-zero vectors (failed / trivial corruptions) are
                     silently dropped.

    Returns
    -------
    basis : (k, n) orthonormal float64 array, k = len(valid directions)
    """
    valid = [d.astype(np.float64) for d in direction_vecs
             if np.linalg.norm(d) > 1e-8]
    if not valid:
        raise ValueError('corruption_basis: no valid direction vectors provided')
    A = np.column_stack(valid)          # (n, k)
    Q, _ = np.linalg.qr(A)
    return Q.T[:len(valid)]             # (k, n) orthonormal rows


# ── Project / reconstruct ─────────────────────────────────────────────────────

def project(x_flat: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Project flat image vector into the subspace: returns (k,) coordinates."""
    return basis @ x_flat


def reconstruct(coords: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Reconstruct flat image from (k,) subspace coordinates."""
    return coords @ basis
