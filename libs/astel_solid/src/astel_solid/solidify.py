"""Top-level L5 solidification: oriented splat cloud → solid + mass properties.

Ties together the SDF → isosurface → mass-property stages, and provides
:func:`surfel_normals` to derive per-splat OUTWARD normals from 2DGS quaternions
and log-scales (the producer integration point — our L3 splats carry orientation
+ anisotropic scale, the thin axis being the surfel normal).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .isosurface import TriMesh, extract_isosurface
from .mass import MassProperties, compute_mass_properties
from .sdf import SdfGrid, oriented_point_sdf


@dataclass(frozen=True)
class SolidResult:
    """The derived L5 solid: watertight mesh, mass properties, and source grid."""

    mesh: TriMesh
    mass: MassProperties
    grid: SdfGrid


def _quats_to_matrices(quats: NDArray[np.float32]) -> NDArray[np.float32]:
    """Convert (N,4) wxyz quaternions to (N,3,3) rotation matrices (columns=axes)."""
    q = np.ascontiguousarray(quats, dtype=np.float64)
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
    return r.astype(np.float32)


def surfel_normals(
    positions: NDArray[np.float32],
    quats: NDArray[np.float32],
    log_scales: NDArray[np.float32],
    *,
    orient_outward: bool = True,
) -> NDArray[np.float32]:
    """Per-splat unit normal = the thinnest principal axis of each gaussian.

    For a surfel the smallest-scale axis is the surface normal. We take the
    rotation-matrix column at ``argmin(log_scales)``. A quaternion gives no
    inside/outside sense, so when ``orient_outward`` we flip each normal to point
    away from the cloud centroid — correct for star-shaped objects; non-star-shaped
    geometry needs proper normal-orientation propagation (future work).
    """
    mats = _quats_to_matrices(quats)
    thin_axis = np.argmin(log_scales, axis=1)  # (N,)
    normals = np.take_along_axis(
        mats, thin_axis[:, None, None], axis=2
    )[:, :, 0]  # (N,3): the selected column
    normals = normals.astype(np.float32)

    if orient_outward:
        centroid = positions.mean(axis=0)
        outward = (positions - centroid).astype(np.float32)
        flip = np.einsum("ij,ij->i", normals, outward) < 0.0
        normals[flip] *= -1.0

    length = np.linalg.norm(normals, axis=1, keepdims=True)
    length = np.where(length == 0.0, 1.0, length)
    return (normals / length).astype(np.float32)


def solidify(
    positions: NDArray[np.float32],
    normals: NDArray[np.float32],
    *,
    resolution: int = 64,
    knn: int = 16,
    bandwidth_voxels: float = 2.5,
    padding: float = 0.1,
    density: float = 1.0,
) -> SolidResult:
    """Oriented surface samples → SDF → watertight mesh → mass properties."""
    grid = oriented_point_sdf(
        positions,
        normals,
        resolution=resolution,
        knn=knn,
        bandwidth_voxels=bandwidth_voxels,
        padding=padding,
    )
    mesh = extract_isosurface(grid)
    mass = compute_mass_properties(mesh, density=density)
    return SolidResult(mesh=mesh, mass=mass, grid=grid)
