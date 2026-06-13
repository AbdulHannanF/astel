from __future__ import annotations

from pathlib import Path

import numpy as np

from astel_splat_io.cloud import SplatCloud
from astel_splat_io.ply import PLY_PROPERTIES, cloud_to_ply_bytes, read_ply, write_ply


def test_ply_header_field_order_and_count(small_cloud: SplatCloud) -> None:
    data = cloud_to_ply_bytes(small_cloud)
    assert data.startswith(b"ply\n")

    end_idx = data.find(b"end_header\n")
    header = data[: end_idx + len(b"end_header\n")].decode("ascii")
    lines = header.splitlines()

    assert "format binary_little_endian 1.0" in lines
    assert f"element vertex {small_cloud.count}" in lines

    properties = [line.split()[2] for line in lines if line.startswith("property")]
    assert properties == list(PLY_PROPERTIES)

    payload = data[end_idx + len(b"end_header\n") :]
    assert len(payload) == small_cloud.count * len(PLY_PROPERTIES) * 4


def test_ply_write_read_round_trip(tmp_path: Path, small_cloud: SplatCloud) -> None:
    out = tmp_path / "cloud.ply"
    size = write_ply(small_cloud, out)
    assert out.stat().st_size == size

    loaded = read_ply(out)
    assert loaded.count == small_cloud.count
    np.testing.assert_array_equal(loaded.positions, small_cloud.positions)
    np.testing.assert_array_equal(loaded.colors_dc, small_cloud.colors_dc)
    np.testing.assert_array_equal(loaded.opacity, small_cloud.opacity)
    np.testing.assert_array_equal(loaded.log_scales, small_cloud.log_scales)
    np.testing.assert_array_equal(loaded.quats, small_cloud.quats)


def test_ply_is_deterministic(small_cloud: SplatCloud) -> None:
    assert cloud_to_ply_bytes(small_cloud) == cloud_to_ply_bytes(small_cloud)
