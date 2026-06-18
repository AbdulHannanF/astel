"""Per-splat surfel normals from 2DGS orientation (self-contained for L4).

Mirrors ``astel_solid.surfel_normals`` (L5 uses the same thin-axis normal) so
``astel_appearance`` can decompose appearance without depending on the heavier
``astel_solid`` (scipy / skimage / coacd) stack. Kept deliberately in sync —
both derive the surface normal as the thinnest principal axis of each gaussian.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _quats_to_matrices(quats: NDArray[np.floating]) -> FloatArray:
    """Convert (N,4) wxyz quaternions to (N,3,3) rotation matrices."""
    q = np.asarray(quats, dtype=np.float64)
    n = np.linalg.norm(q, axis=1, keepdims=True)
    n = np.where(n == 0.0, 1.0, n)
    w, x, y, z = (q / n).T
    r = np.empty((q.shape[0], 3, 3), dtype=np.float64)
    r[:, 0, 0] = 1 - 2 * (y * y + z * z)
    r[:, 1, 0] = 2 * (x * y + w * z)
    r[:, 2, 0] = 2 * (x * z - w * y)
    r[:, 0, 1] = 2 * (x * y - w * z)
    r[:, 1, 1] = 1 - 2 * (x * x + z * z)
    r[:, 2, 1] = 2 * (y * z + w * x)
    r[:, 0, 2] = 2 * (x * z + w * y)
    r[:, 1, 2] = 2 * (y * z - w * x)
    r[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return r


def surfel_normals(
    positions: NDArray[np.floating],
    quats: NDArray[np.floating],
    log_scales: NDArray[np.floating],
    *,
    orient_outward: bool = True,
) -> FloatArray:
    """Unit normal per splat = the thinnest principal axis of each gaussian.

    When ``orient_outward`` the normals are flipped to point away from the cloud
    centroid (correct for star-shaped objects; matches ``astel_solid``).
    """
    mats = _quats_to_matrices(quats)
    thin_axis = np.argmin(np.asarray(log_scales, dtype=np.float64), axis=1)
    normals = np.take_along_axis(mats, thin_axis[:, None, None], axis=2)[:, :, 0]

    if orient_outward:
        pos = np.asarray(positions, dtype=np.float64)
        centroid = pos.mean(axis=0)
        outward = pos - centroid
        flip = np.einsum("ij,ij->i", normals, outward) < 0.0
        normals[flip] *= -1.0

    length = np.linalg.norm(normals, axis=1, keepdims=True)
    length = np.where(length == 0.0, 1.0, length)
    normals /= length
    return normals
