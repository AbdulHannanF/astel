"""Tests for KHR_gaussian_splatting glTF/GLB export + round-trip."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from astel_splat_io.cloud import SplatCloud
from astel_splat_io.gltf import (
    _CHUNK_BIN,
    _CHUNK_JSON,
    _EXT,
    _GLB_MAGIC,
    read_gltf,
    write_gltf,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cloud(n: int = 8, seed: int = 42) -> SplatCloud:
    rng = np.random.default_rng(seed)
    positions = rng.uniform(-1.0, 1.0, (n, 3)).astype(np.float32)
    colors_dc = rng.uniform(-0.5, 0.5, (n, 3)).astype(np.float32)
    opacity = rng.normal(0.0, 1.0, n).astype(np.float32)
    log_scales = rng.uniform(-3.0, -1.0, (n, 3)).astype(np.float32)
    q = rng.normal(size=(n, 4)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return SplatCloud(
        positions=positions,
        colors_dc=colors_dc,
        opacity=opacity,
        log_scales=log_scales,
        quats=q.astype(np.float32),
    )


# ---------------------------------------------------------------------------
# GLB structure tests
# ---------------------------------------------------------------------------


def test_write_gltf_produces_file(tmp_path: Path) -> None:
    cloud = _make_cloud()
    out = tmp_path / "test.glb"
    n_bytes = write_gltf(cloud, out)
    assert out.exists()
    assert n_bytes == out.stat().st_size
    assert n_bytes > 0


def test_glb_magic_and_version(tmp_path: Path) -> None:
    cloud = _make_cloud()
    out = tmp_path / "test.glb"
    write_gltf(cloud, out)
    data = out.read_bytes()
    magic, version, total = struct.unpack_from("<III", data, 0)
    assert magic == _GLB_MAGIC, "GLB magic mismatch"
    assert version == 2
    assert total == len(data)


def test_glb_json_chunk_first(tmp_path: Path) -> None:
    cloud = _make_cloud()
    out = tmp_path / "test.glb"
    write_gltf(cloud, out)
    data = out.read_bytes()
    _, json_type = struct.unpack_from("<II", data, 12)
    assert json_type == _CHUNK_JSON


def test_glb_bin_chunk_second(tmp_path: Path) -> None:
    cloud = _make_cloud()
    out = tmp_path / "test.glb"
    write_gltf(cloud, out)
    data = out.read_bytes()
    json_len, _ = struct.unpack_from("<II", data, 12)
    offset = 12 + 8 + json_len
    _, bin_type = struct.unpack_from("<II", data, offset)
    assert bin_type == _CHUNK_BIN


def test_glb_declares_extension(tmp_path: Path) -> None:
    import json

    cloud = _make_cloud()
    out = tmp_path / "test.glb"
    write_gltf(cloud, out)
    data = out.read_bytes()
    json_len, _ = struct.unpack_from("<II", data, 12)
    gltf = json.loads(data[20 : 20 + json_len])
    assert _EXT in gltf["extensionsUsed"]
    prim = gltf["meshes"][0]["primitives"][0]
    assert _EXT in prim["extensions"]


def test_empty_cloud_raises(tmp_path: Path) -> None:
    empty = SplatCloud(
        positions=np.zeros((0, 3), dtype=np.float32),
        colors_dc=np.zeros((0, 3), dtype=np.float32),
        opacity=np.zeros(0, dtype=np.float32),
        log_scales=np.zeros((0, 3), dtype=np.float32),
        quats=np.zeros((0, 4), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="empty"):
        write_gltf(empty, tmp_path / "empty.glb")


# ---------------------------------------------------------------------------
# Round-trip tests (golden-file behaviour)
# ---------------------------------------------------------------------------


def test_round_trip_positions(tmp_path: Path) -> None:
    cloud = _make_cloud(n=32)
    out = tmp_path / "rt.glb"
    write_gltf(cloud, out)
    back = read_gltf(out)
    np.testing.assert_allclose(back.positions, cloud.positions, atol=1e-5)


def test_round_trip_quats(tmp_path: Path) -> None:
    cloud = _make_cloud(n=32)
    out = tmp_path / "rt.glb"
    write_gltf(cloud, out)
    back = read_gltf(out)
    np.testing.assert_allclose(back.quats, cloud.quats, atol=1e-5)


def test_round_trip_log_scales(tmp_path: Path) -> None:
    cloud = _make_cloud(n=32)
    out = tmp_path / "rt.glb"
    write_gltf(cloud, out)
    back = read_gltf(out)
    np.testing.assert_allclose(back.log_scales, cloud.log_scales, atol=1e-5)


def test_round_trip_opacity(tmp_path: Path) -> None:
    """Opacity passes through sigmoid → logit with < 0.5% relative error."""
    cloud = _make_cloud(n=32)
    out = tmp_path / "rt.glb"
    write_gltf(cloud, out)
    back = read_gltf(out)
    # Convert both to alpha for a numerically fair comparison
    alpha_orig = 1.0 / (1.0 + np.exp(-cloud.opacity))
    alpha_back = 1.0 / (1.0 + np.exp(-back.opacity))
    np.testing.assert_allclose(alpha_back, alpha_orig, atol=1e-5)


def test_round_trip_colors(tmp_path: Path) -> None:
    cloud = _make_cloud(n=32)
    out = tmp_path / "rt.glb"
    write_gltf(cloud, out)
    back = read_gltf(out)
    np.testing.assert_allclose(back.colors_dc, cloud.colors_dc, atol=1e-4)


def test_round_trip_count(tmp_path: Path, small_cloud: SplatCloud) -> None:
    out = tmp_path / "rt_small.glb"
    write_gltf(small_cloud, out)
    back = read_gltf(out)
    assert back.count == small_cloud.count


def test_read_rejects_non_glb(tmp_path: Path) -> None:
    bad = tmp_path / "bad.glb"
    bad.write_bytes(b"notglTF" + b"\x00" * 20)
    with pytest.raises(ValueError):
        read_gltf(bad)


def test_read_rejects_missing_extension(tmp_path: Path) -> None:
    """A GLB whose JSON lacks KHR_gaussian_splatting must raise ValueError."""
    import json as _json

    # Build a minimal GLB, then rebuild it from scratch without the extension.
    cloud = _make_cloud(n=4)
    ref = tmp_path / "ref.glb"
    write_gltf(cloud, ref)
    data = ref.read_bytes()
    json_len, _ = struct.unpack_from("<II", data, 12)
    raw_json = data[20 : 20 + json_len]
    gltf = _json.loads(raw_json)
    # Strip the extension
    gltf["extensionsUsed"] = []
    prim = gltf["meshes"][0]["primitives"][0]
    prim.pop("extensions", None)

    new_json = _pad4(_json.dumps(gltf).encode("utf-8"), 0x20)
    bin_offset = 12 + 8 + json_len  # original BIN offset
    bin_len, _ = struct.unpack_from("<II", data, bin_offset)
    bin_data = data[bin_offset + 8 : bin_offset + 8 + bin_len]
    new_bin = _pad4(bin_data, 0x00)

    from astel_splat_io.gltf import (  # noqa: F401
        _CHUNK_BIN,
        _CHUNK_JSON,
        _GLB_MAGIC,
        _GLB_VERSION,
    )

    total = 12 + 8 + len(new_json) + 8 + len(new_bin)
    header = struct.pack("<III", _GLB_MAGIC, _GLB_VERSION, total)
    json_chunk = struct.pack("<II", len(new_json), _CHUNK_JSON) + new_json
    bin_chunk = struct.pack("<II", len(new_bin), _CHUNK_BIN) + new_bin
    bad = tmp_path / "bad_ext.glb"
    bad.write_bytes(header + json_chunk + bin_chunk)

    with pytest.raises(ValueError, match=_EXT):
        read_gltf(bad)


def _pad4(data: bytes, pad_byte: int) -> bytes:
    rem = len(data) % 4
    if rem:
        data += bytes([pad_byte] * (4 - rem))
    return data
