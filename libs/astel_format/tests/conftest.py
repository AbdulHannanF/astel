"""Shared fixtures for astel_format tests."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

# Minimal binary-little-endian PLY: the exact 3DGS property layout from
# pipelines/stub/make_sample_splat.py, but tiny (a handful of splats) so
# tests stay fast and self-contained -- we don't depend on that sibling
# package.
_PLY_PROPERTIES: tuple[str, ...] = (
    "x",
    "y",
    "z",
    "f_dc_0",
    "f_dc_1",
    "f_dc_2",
    "opacity",
    "scale_0",
    "scale_1",
    "scale_2",
    "rot_0",
    "rot_1",
    "rot_2",
    "rot_3",
)


def _make_ply_bytes(count: int) -> bytes:
    header_lines = ["ply", "format binary_little_endian 1.0", f"element vertex {count}"]
    header_lines += [f"property float {name}" for name in _PLY_PROPERTIES]
    header_lines.append("end_header")
    header = ("\n".join(header_lines) + "\n").encode("ascii")

    row = struct.pack(
        "<14f",
        0.0,
        0.0,
        0.0,  # xyz
        0.1,
        0.2,
        0.3,  # f_dc
        0.5,  # opacity
        -2.0,
        -2.0,
        -2.0,  # log scales
        1.0,
        0.0,
        0.0,
        0.0,  # quat (identity)
    )
    return header + row * count


@pytest.fixture
def small_ply_path(tmp_path: Path) -> Path:
    """A tiny (8-gaussian) binary PLY in the 3DGS layout."""
    path = tmp_path / "splats.ply"
    path.write_bytes(_make_ply_bytes(8))
    return path


@pytest.fixture
def small_ply_count() -> int:
    return 8
