"""Test cloud helpers for astel_lod.

Provides factory functions that construct (opacity, log_scales) arrays with a
KNOWN importance ordering so tests can assert on exact top-k selections.

Importance formula (from importance.py):
    importance[i] = opacity[i] * exp(log_scales[i].sum())

Cloud layout used in most tests
--------------------------------
We build a cloud with a small number of "dominant" Gaussians (high opacity,
large scales → high importance) and a larger number of "background" Gaussians
(low opacity, small scales → low importance).  The dominant indices are always
0..n_dominant-1 in the returned arrays, making ground-truth assertions easy.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def dominant_cloud(
    n_dominant: int = 5,
    n_background: int = 20,
    *,
    dominant_opacity: float = 0.9,
    background_opacity: float = 0.05,
    dominant_log_scale: float = 2.0,  # exp(2)*exp(2)*exp(2) ≈ 403
    background_log_scale: float = -3.0,  # exp(-3)^3 ≈ 6e-5
    rng_seed: int = 42,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return (opacity, log_scales) with indices 0..n_dominant-1 strictly dominant.

    All dominant Gaussians have importance >> all background Gaussians, so the
    ground-truth top-k for any k ≤ n_dominant is exactly {0, 1, …, k-1} (in
    ascending order, matching select_lod_indices output).

    For k > n_dominant the remaining selections are arbitrary background ones;
    tests only assert exact top-k content when k ≤ n_dominant.
    """
    rng = np.random.default_rng(rng_seed)
    n = n_dominant + n_background

    opacity = np.empty(n, dtype=np.float64)
    log_scales = np.empty((n, 3), dtype=np.float64)

    # Dominant block: uniformly high opacity, large isotropic scale.
    opacity[:n_dominant] = dominant_opacity
    log_scales[:n_dominant] = dominant_log_scale

    # Background block: low opacity, small isotropic scale with slight jitter
    # so no two background Gaussians are identical (avoids tie-breaking issues).
    opacity[n_dominant:] = background_opacity
    log_scales[n_dominant:] = background_log_scale + rng.uniform(
        -0.1, 0.1, size=(n_background, 3)
    )

    return opacity, log_scales


def single_splat_pair() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Two Gaussians: index 0 is large+opaque, index 1 is tiny+transparent."""
    opacity = np.array([0.95, 0.01], dtype=np.float64)
    log_scales = np.array([[1.0, 1.0, 1.0], [-2.0, -2.0, -2.0]], dtype=np.float64)
    return opacity, log_scales


def monotone_opacity_cloud(
    n: int = 10,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Cloud where opacity increases with index (fixed scales).

    importance[i] strictly increases with i, so the top-1 is index n-1.
    """
    opacity = np.linspace(0.01, 0.99, n, dtype=np.float64)
    log_scales = np.zeros((n, 3), dtype=np.float64)  # footprint = exp(0)^3 = 1
    return opacity, log_scales


def monotone_scale_cloud(
    n: int = 10,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Cloud where log_scale increases with index (fixed opacity).

    projected_footprint increases with index → importance increases with index.
    """
    opacity = np.full(n, 0.5, dtype=np.float64)
    scales_1d = np.linspace(-2.0, 2.0, n, dtype=np.float64)
    log_scales = np.stack([scales_1d, scales_1d, scales_1d], axis=1)
    return opacity, log_scales
