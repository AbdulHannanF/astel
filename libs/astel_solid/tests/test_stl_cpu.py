"""Binary STL writer: exact byte layout + round-trip header/count."""

from __future__ import annotations

import struct
from pathlib import Path

from _shapes import unit_cube

from astel_solid.stl import write_binary_stl


def test_binary_stl_size_and_count(tmp_path: Path) -> None:
    mesh = unit_cube()
    path = tmp_path / "cube.stl"
    n = write_binary_stl(mesh, path)

    # 80-byte header + 4-byte count + 50 bytes per triangle.
    expected = 84 + 50 * mesh.n_faces
    assert n == expected
    assert path.stat().st_size == expected

    data = path.read_bytes()
    (count,) = struct.unpack_from("<I", data, 80)
    assert count == mesh.n_faces  # 12 for a cube


def test_binary_stl_first_triangle_vertices(tmp_path: Path) -> None:
    mesh = unit_cube(side=2.0)
    path = tmp_path / "cube2.stl"
    write_binary_stl(mesh, path)
    data = path.read_bytes()

    # First triangle record starts at byte 84: normal(3f) then v1(3f).
    nx, ny, nz, v1x, v1y, v1z = struct.unpack_from("<6f", data, 84)
    # Normal is unit length.
    assert abs((nx * nx + ny * ny + nz * nz) ** 0.5 - 1.0) < 1e-5
    # v1 of face 0 is corner 0 = (-1,-1,-1) for a side-2 cube.
    assert (round(v1x, 5), round(v1y, 5), round(v1z, 5)) == (-1.0, -1.0, -1.0)
