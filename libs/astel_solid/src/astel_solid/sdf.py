"""Oriented-point signed distance field on a regular voxel grid.

Given surface samples (positions) with OUTWARD unit normals, estimate a signed
distance field whose zero level set is the surface. We use an implicit
moving-least-squares (IMLS / Hoppe tangent-plane) estimator: at a query point
``q`` the signed value is the gaussian-weighted average of the point-to-plane
distances ``<q - p_i, n_i>`` over the ``k`` nearest samples. With outward normals
this is **negative inside, positive outside** — exactly the convention marching
cubes consumes at ``level=0``.

Pure CPU (numpy + scipy KDTree); no torch, no GPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class SdfGrid:
    """A signed distance field sampled on a regular axis-aligned grid.

    ``values`` is ``(nx, ny, nz)``; ``origin`` is the world coordinate of voxel
    ``(0,0,0)``; ``spacing`` is the per-axis voxel size. World coordinate of voxel
    ``(i,j,k)`` is ``origin + (i,j,k) * spacing``.
    """

    values: NDArray[np.float32]
    origin: NDArray[np.float32]
    spacing: NDArray[np.float32]


def _grid_axes(
    lo: NDArray[np.float32], hi: NDArray[np.float32], resolution: int
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """Per-axis grid sample coordinates and spacing for the longest-axis res."""
    extent = hi - lo
    longest = float(extent.max())
    spacing_scalar = longest / float(resolution - 1)
    counts = np.maximum(2, np.ceil(extent / spacing_scalar).astype(int) + 1)
    axes = [
        (lo[d] + np.arange(counts[d], dtype=np.float32) * spacing_scalar).astype(
            np.float32
        )
        for d in range(3)
    ]
    return axes[0], axes[1], axes[2]


def oriented_point_sdf(
    positions: NDArray[np.float32],
    normals: NDArray[np.float32],
    *,
    resolution: int = 64,
    knn: int = 16,
    bandwidth_voxels: float = 2.5,
    padding: float = 0.1,
) -> SdfGrid:
    """Estimate a signed distance grid from oriented surface samples.

    ``resolution`` is the voxel count along the longest bbox axis (other axes get
    proportionally fewer voxels at the same spacing). ``knn`` neighbours and a
    gaussian ``bandwidth_voxels`` (in voxel-spacing units) control smoothness.
    ``padding`` expands the bbox by a fraction of its longest extent so the
    surface sits strictly inside the grid (marching cubes needs the zero crossing
    interior).
    """
    positions = np.ascontiguousarray(positions, dtype=np.float32)
    normals = np.ascontiguousarray(normals, dtype=np.float32)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must be (N, 3)")
    if normals.shape != positions.shape:
        raise ValueError("normals must match positions shape")

    norm_len = np.linalg.norm(normals, axis=1, keepdims=True)
    norm_len = np.where(norm_len == 0.0, 1.0, norm_len)
    unit_normals = (normals / norm_len).astype(np.float32)

    lo = positions.min(axis=0)
    hi = positions.max(axis=0)
    pad = float((hi - lo).max()) * padding
    lo = (lo - pad).astype(np.float32)
    hi = (hi + pad).astype(np.float32)

    xs, ys, zs = _grid_axes(lo, hi, resolution)
    spacing = np.full(3, float(xs[1] - xs[0]), dtype=np.float32)

    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    query = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1).astype(np.float32)

    tree = cKDTree(positions)
    k = int(min(knn, positions.shape[0]))
    dists, idx = tree.query(query, k=k)
    if k == 1:
        dists = dists[:, None]
        idx = idx[:, None]

    h = float(spacing[0]) * bandwidth_voxels
    weights = np.exp(-((dists / h) ** 2)).astype(np.float32)
    weight_sum = weights.sum(axis=1, keepdims=True)
    weight_sum = np.where(weight_sum == 0.0, 1.0, weight_sum)

    # Point-to-plane signed distance for each (query, neighbour) pair.
    neigh_p = positions[idx]  # (M, k, 3)
    neigh_n = unit_normals[idx]  # (M, k, 3)
    delta = query[:, None, :] - neigh_p  # (M, k, 3)
    signed = np.einsum("mkd,mkd->mk", delta, neigh_n)  # (M, k)
    sdf_flat = (weights * signed).sum(axis=1, keepdims=True) / weight_sum
    values = sdf_flat.reshape(gx.shape).astype(np.float32)

    return SdfGrid(values=values, origin=lo, spacing=spacing)
