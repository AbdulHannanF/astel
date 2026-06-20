"""Bake a DeformationField into explicit per-frame positions.

Useful for export, preview, or when downstream consumers cannot evaluate LBS
themselves.
"""

from __future__ import annotations

import numpy as np

from .field import DeformationField


def bake_per_frame(
    field: DeformationField,
    base_positions: np.ndarray,
) -> np.ndarray:
    """Return explicit per-frame deformed Gaussian positions.

    Parameters
    ----------
    field:
        A fitted :class:`~astel_dynamics.field.DeformationField`.
    base_positions:
        Rest-pose Gaussian means, shape ``(N, 3)``.

    Returns
    -------
    np.ndarray
        Shape ``(F, N, 3)`` float32 — deformed positions for every frame.
    """
    base = np.asarray(base_positions, dtype=np.float32)
    frames_out = np.empty((field.n_frames, field.n_gaussians, 3), dtype=np.float32)
    for f in range(field.n_frames):
        frames_out[f] = field.apply(base, frame=f)
    return frames_out
