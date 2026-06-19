"""
metrics.py — perceptual and pixel-level distance metrics.

All functions accept (C, H, W) float32 numpy arrays in [0, 1].
"""

from __future__ import annotations
import numpy as np
from skimage.metrics import structural_similarity as _ski_ssim


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Structural similarity between two (C, H, W) images."""
    if a.shape[0] == 1:
        return float(_ski_ssim(a.squeeze(0), b.squeeze(0), data_range=1.0))
    return float(_ski_ssim(
        a.transpose(1, 2, 0), b.transpose(1, 2, 0),
        data_range=1.0, channel_axis=2,
    ))


def mse(a: np.ndarray, b: np.ndarray) -> float:
    """Mean squared error."""
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


def linf(a: np.ndarray, b: np.ndarray) -> float:
    """L∞ norm of the perturbation."""
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def l2(a: np.ndarray, b: np.ndarray) -> float:
    """L2 norm of the perturbation."""
    return float(np.linalg.norm(a.astype(np.float64) - b.astype(np.float64)))


def all_metrics(a: np.ndarray, b: np.ndarray) -> dict:
    """Return all four metrics in a single dict."""
    delta = a.astype(np.float64) - b.astype(np.float64)
    return {
        'ssim': ssim(a, b),
        'mse':  float(np.mean(delta ** 2)),
        'linf': float(np.max(np.abs(delta))),
        'l2':   float(np.linalg.norm(delta)),
    }
