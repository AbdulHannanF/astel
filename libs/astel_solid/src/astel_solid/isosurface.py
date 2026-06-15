"""Watertight isosurface extraction from a signed distance grid (marching cubes).

Wraps ``skimage.measure.marching_cubes`` at the zero level set and returns world-
space vertices + triangle faces wound so that the surface normals point OUTWARD
(verified by signed volume — flipped if negative). The result is a closed,
manifold triangle mesh used only internally (print path / physics volume /
collision proxy), never shipped as the asset.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from skimage import measure

from .sdf import SdfGrid


@dataclass(frozen=True)
class TriMesh:
    """A triangle mesh: ``vertices`` ``(V,3)`` float32, ``faces`` ``(F,3)`` int."""

    vertices: NDArray[np.float32]
    faces: NDArray[np.int64]

    @property
    def n_vertices(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def n_faces(self) -> int:
        return int(self.faces.shape[0])


def _signed_volume(vertices: NDArray[np.float32], faces: NDArray[np.int64]) -> float:
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    return float(np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum() / 6.0)


def extract_isosurface(grid: SdfGrid, *, level: float = 0.0) -> TriMesh:
    """Marching-cubes the SDF at ``level`` into an outward-wound world-space mesh.

    Raises ``ValueError`` (from skimage) if the level set does not intersect the
    grid (e.g. an all-positive or all-negative field).
    """
    verts_idx, faces, _normals, _values = measure.marching_cubes(  # type: ignore[no-untyped-call]
        grid.values, level=level, spacing=tuple(float(s) for s in grid.spacing)
    )
    vertices = (verts_idx + grid.origin[None, :]).astype(np.float32)
    faces = faces.astype(np.int64)

    # Ensure outward winding (positive enclosed volume).
    if _signed_volume(vertices, faces) < 0.0:
        faces = faces[:, ::-1].copy()

    return TriMesh(vertices=vertices, faces=faces)
