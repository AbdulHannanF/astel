"""Binary pack/unpack and bake_per_frame tests."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from _motion import static_cloud

from astel_dynamics.baked import bake_per_frame
from astel_dynamics.field import DeformationField
from astel_dynamics.fit import fit_deformation_field
from astel_dynamics.pack import read_deformation_bin, write_deformation_bin


def _small_field(n: int = 20, k: int = 3, f: int = 5) -> DeformationField:
    """Create a DeformationField with deterministic random content."""
    rng = np.random.default_rng(99)
    node_positions = rng.random((k, 3)).astype(np.float32)
    raw_w = rng.random((n, k)).astype(np.float32)
    weights = raw_w / raw_w.sum(axis=1, keepdims=True)
    node_transforms = rng.standard_normal((f, k, 3, 4)).astype(np.float32)
    return DeformationField(
        node_positions=node_positions,
        weights=weights,
        node_transforms=node_transforms,
    )


def test_write_read_lossless_node_positions() -> None:
    field = _small_field()
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir) / "field.bin"
        write_deformation_bin(field, p)
        field2 = read_deformation_bin(p)
    np.testing.assert_array_equal(field.node_positions, field2.node_positions)


def test_write_read_lossless_weights() -> None:
    field = _small_field()
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir) / "field.bin"
        write_deformation_bin(field, p)
        field2 = read_deformation_bin(p)
    np.testing.assert_array_equal(field.weights, field2.weights)


def test_write_read_lossless_transforms() -> None:
    field = _small_field()
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir) / "field.bin"
        write_deformation_bin(field, p)
        field2 = read_deformation_bin(p)
    np.testing.assert_array_equal(field.node_transforms, field2.node_transforms)


def test_shapes_preserved_after_round_trip() -> None:
    n, k, f = 37, 6, 11
    field = _small_field(n=n, k=k, f=f)
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir) / "field.bin"
        write_deformation_bin(field, p)
        field2 = read_deformation_bin(p)
    assert field2.node_positions.shape == (k, 3)
    assert field2.weights.shape == (n, k)
    assert field2.node_transforms.shape == (f, k, 3, 4)


def test_bad_magic_raises() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir) / "bad.bin"
        p.write_bytes(b"BADMAGIC" + b"\x00" * 100)
        with pytest.raises(ValueError, match="magic"):
            read_deformation_bin(p)


def test_truncated_file_raises() -> None:
    """A file whose body is shorter than its header declares must raise."""
    field = _small_field(n=20, k=3, f=5)
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir) / "field.bin"
        write_deformation_bin(field, p)
        full = p.read_bytes()
        p.write_bytes(full[:-16])  # drop the last 4 floats
        with pytest.raises(ValueError, match="size mismatch"):
            read_deformation_bin(p)


def test_malicious_oversized_header_raises_without_huge_alloc() -> None:
    """A tiny file declaring an enormous N/K/F must be rejected immediately.

    Closes the amplification vector: the exact-size check rejects the crafted
    header before any large allocation/slice, so an untrusted .astel cannot
    coerce a huge read from a few bytes.
    """
    import struct

    # magic + header claiming N=K=F=2**20 (~TBs of declared float data) but no body.
    payload = b"ASTLDYN0" + struct.pack("<III", 2**20, 2**20, 2**20)
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir) / "evil.bin"
        p.write_bytes(payload)
        with pytest.raises(ValueError, match="size mismatch"):
            read_deformation_bin(p)


def test_dtype_is_float32_after_round_trip() -> None:
    field = _small_field()
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir) / "field.bin"
        write_deformation_bin(field, p)
        field2 = read_deformation_bin(p)
    assert field2.node_positions.dtype == np.float32
    assert field2.weights.dtype == np.float32
    assert field2.node_transforms.dtype == np.float32


def test_bake_per_frame_shape() -> None:
    """bake_per_frame must return (F, N, 3)."""
    n = 40
    base = static_cloud(n, seed=10)
    field = _small_field(n=n, k=4, f=7)
    baked = bake_per_frame(field, base)
    assert baked.shape == (7, n, 3)
    assert baked.dtype == np.float32


def test_bake_equals_stacked_apply() -> None:
    """bake_per_frame must equal stacking field.apply for each frame."""
    n = 30
    base = static_cloud(n, seed=11)
    field = _small_field(n=n, k=3, f=6)
    baked = bake_per_frame(field, base)

    for f in range(6):
        expected = field.apply(base, frame=f)
        np.testing.assert_array_equal(baked[f], expected)


def test_bake_on_fit_field() -> None:
    """bake_per_frame on a fitted field must match the fitted field's apply."""
    from _motion import rigid_rotation_motion

    base = static_cloud(60, seed=12)
    frames = rigid_rotation_motion(base, n_frames=5, axis=[0, 1, 0], total_angle=1.0)
    field, _ = fit_deformation_field(base, frames, n_nodes=3, seed=0)

    baked = bake_per_frame(field, base)
    assert baked.shape == (5, 60, 3)

    for f in range(5):
        np.testing.assert_array_equal(baked[f], field.apply(base, frame=f))
