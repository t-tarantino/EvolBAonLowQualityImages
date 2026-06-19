"""
boundary.py — geometric utilities for boundary-following attacks.

All functions operate on flat (n,) numpy arrays unless stated otherwise.
"""

from __future__ import annotations
import numpy as np


def approaching_direction(x_orig: np.ndarray, x_b: np.ndarray) -> np.ndarray:
    """
    Unit vector pointing from x_b toward x_orig (the 'good' direction).

    This is the direction that reduces perturbation magnitude.
    Using û as u_hat throughout Phase 3.
    """
    d = x_orig.flatten() - x_b.flatten()
    norm = np.linalg.norm(d)
    if norm < 1e-12:
        return np.zeros_like(d)
    return d / norm


def half_space_reflect(delta: np.ndarray, u_hat: np.ndarray) -> np.ndarray:
    """
    Reflect delta so it never moves away from the original image.

    For each sampled step delta, compute dot(delta, u_hat):
      - If negative (moving away from x_orig): reflect across the hyperplane
        perpendicular to u_hat.
      - If non-negative: return as-is.

    Reflection is preferred over rejection because it preserves step
    magnitude and avoids biasing the covariance update (BRAINSTORMING §3).

    Parameters
    ----------
    delta : (n,) step vector
    u_hat : (n,) unit vector toward x_orig

    Returns
    -------
    delta_reflected : (n,)
    """
    d = float(np.dot(delta, u_hat))
    if d < 0:
        return delta - 2.0 * d * u_hat
    return delta


def boundary_normal_estimate(
    non_adv_flat: np.ndarray,
    x_b_flat: np.ndarray,
) -> np.ndarray | None:
    """
    Estimate the local decision-boundary normal from non-adversarial candidates.

    Each non-adversarial point lies on the clean side of the boundary; the
    vector (x_i - x_b) approximates a direction pointing toward the clean
    region, i.e. the outward boundary normal (BRAINSTORMING §4).

    Parameters
    ----------
    non_adv_flat : (k, n) array of non-adversarial candidate images
    x_b_flat     : (n,)  current boundary point

    Returns
    -------
    n_hat : (n,) unit normal, or None if the estimate is degenerate
    """
    if non_adv_flat.ndim == 1:
        non_adv_flat = non_adv_flat[np.newaxis, :]
    if len(non_adv_flat) == 0:
        return None

    dirs  = non_adv_flat - x_b_flat[np.newaxis, :]
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    valid = (norms.squeeze(1) > 1e-12)
    if not valid.any():
        return None

    unit_dirs = dirs[valid] / norms[valid]
    n_hat     = unit_dirs.mean(axis=0)
    norm      = np.linalg.norm(n_hat)
    if norm < 1e-12:
        return None
    return n_hat / norm


def init_covariance_biased(
    n: int,
    u_hat: np.ndarray,
    epsilon: float = 0.01,
) -> np.ndarray:
    """
    Build an initial diagonal covariance that suppresses variance in û.

    C₀ = I - (1 - ε) · û·ûᵀ

    This gives ~ε variance in the û direction (toward x_orig, which
    immediately crosses the boundary) and ~1 variance in all perpendicular
    directions (boundary-tangent, where useful search lives).

    Returned as a (n,) diagonal vector (sep-CMA-ES representation).
    (BRAINSTORMING §3)
    """
    diag = np.ones(n, dtype=np.float64)
    diag -= (1.0 - epsilon) * u_hat ** 2
    return np.clip(diag, 1e-10, None)
