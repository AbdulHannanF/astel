"""SDF sign convention + isosurface extraction on a sampled sphere."""

from __future__ import annotations

import numpy as np
from _shapes import fibonacci_sphere

from astel_solid.isosurface import extract_isosurface
from astel_solid.sdf import oriented_point_sdf


def test_sdf_sign_inside_negative_outside_positive() -> None:
    pts, normals = fibonacci_sphere(3000, radius=1.0)
    grid = oriented_point_sdf(pts, normals, resolution=48)

    # Centre voxel is deep inside -> negative; a corner is outside -> positive.
    nx, ny, nz = grid.values.shape
    center = grid.values[nx // 2, ny // 2, nz // 2]
    corner = grid.values[0, 0, 0]
    assert center < 0.0
    assert corner > 0.0


def test_isosurface_is_outward_wound_and_near_sphere() -> None:
    pts, normals = fibonacci_sphere(4000, radius=1.0)
    grid = oriented_point_sdf(pts, normals, resolution=56)
    mesh = extract_isosurface(grid)

    assert mesh.n_vertices > 100
    assert mesh.n_faces > 100

    # Outward winding => positive signed volume.
    a = mesh.vertices[mesh.faces[:, 0]]
    b = mesh.vertices[mesh.faces[:, 1]]
    c = mesh.vertices[mesh.faces[:, 2]]
    signed_vol = float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)
    assert signed_vol > 0.0

    # Vertices sit near radius 1 (within the grid voxel scale).
    radii = np.linalg.norm(mesh.vertices, axis=1)
    assert abs(float(radii.mean()) - 1.0) < 0.1
