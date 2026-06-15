"""Analytic test shapes with known mass properties."""

from __future__ import annotations

import numpy as np

from astel_solid.isosurface import TriMesh


def unit_cube(
    side: float = 1.0, center: tuple[float, float, float] = (0, 0, 0)
) -> TriMesh:
    """An axis-aligned cube, outward-wound. Volume = side³; centred at ``center``."""
    h = side / 2.0
    c = np.asarray(center, dtype=np.float32)
    corners = np.array(
        [
            [-h, -h, -h], [h, -h, -h], [h, h, -h], [-h, h, -h],
            [-h, -h, h], [h, -h, h], [h, h, h], [-h, h, h],
        ],
        dtype=np.float32,
    ) + c
    faces = np.array(
        [
            [0, 3, 2], [0, 2, 1],   # -z
            [4, 5, 6], [4, 6, 7],   # +z
            [0, 1, 5], [0, 5, 4],   # -y
            [3, 7, 6], [3, 6, 2],   # +y
            [0, 4, 7], [0, 7, 3],   # -x
            [1, 2, 6], [1, 6, 5],   # +x
        ],
        dtype=np.int64,
    )
    return TriMesh(vertices=corners, faces=faces)


def fibonacci_sphere(n: int, radius: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """``n`` near-uniform points on a sphere + outward (radial) unit normals."""
    i = np.arange(n, dtype=np.float64)
    phi = np.pi * (3.0 - np.sqrt(5.0))  # golden angle
    y = 1.0 - 2.0 * (i + 0.5) / n
    r = np.sqrt(np.clip(1.0 - y * y, 0.0, 1.0))
    theta = phi * i
    pts = np.stack([np.cos(theta) * r, y, np.sin(theta) * r], axis=1)
    normals = pts.copy().astype(np.float32)
    return (pts * radius).astype(np.float32), normals
