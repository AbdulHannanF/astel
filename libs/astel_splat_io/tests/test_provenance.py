from __future__ import annotations

from pathlib import Path

import numpy as np

from astel_splat_io.cloud import SplatCloud
from astel_splat_io.provenance import (
    decode_provenance_u8,
    encode_provenance_u8,
    read_provenance_sidecar,
    write_provenance_sidecar,
)
from astel_splat_io.spz import write_spz


def test_provenance_encode_decode_round_trip() -> None:
    provenance = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
    encoded = encode_provenance_u8(provenance)

    assert encoded.dtype == np.uint8
    assert encoded.tolist() == [0, 64, 128, 191, 255]

    decoded = decode_provenance_u8(encoded)
    np.testing.assert_allclose(decoded, provenance, atol=1.0 / 255.0)


def test_provenance_sidecar_round_trip(tmp_path: Path, small_cloud: SplatCloud) -> None:
    rng = np.random.default_rng(1)
    provenance = rng.uniform(0.0, 1.0, size=small_cloud.count).astype(np.float32)

    spz_path = tmp_path / "asset.spz"
    write_spz(small_cloud, spz_path)

    sidecar_path = tmp_path / "asset.astl.json"
    write_provenance_sidecar(provenance, spz_path, sidecar_path)

    bin_path = tmp_path / "asset.provenance.bin"
    assert bin_path.exists()
    assert bin_path.stat().st_size == small_cloud.count

    loaded = read_provenance_sidecar(sidecar_path)
    np.testing.assert_allclose(loaded, provenance, atol=1.0 / 255.0)


def test_provenance_reorder_alignment(tmp_path: Path, small_cloud: SplatCloud) -> None:
    """Golden test: a geometry permutation must produce the same permutation
    in the provenance buffer (manifest-v0 section 5.4)."""
    rng = np.random.default_rng(2)
    n = small_cloud.count
    provenance = rng.uniform(0.0, 1.0, size=n).astype(np.float32)

    # Simulate an exporter reorder (e.g. SPZ Morton sort): reverse order.
    order = np.arange(n)[::-1].copy()

    reordered_cloud = small_cloud.reordered(order)
    assert reordered_cloud.count == n
    np.testing.assert_array_equal(
        reordered_cloud.positions, small_cloud.positions[order]
    )

    spz_path = tmp_path / "asset_reordered.spz"
    write_spz(reordered_cloud, spz_path)

    sidecar_path = tmp_path / "asset_reordered.astl.json"
    write_provenance_sidecar(provenance, spz_path, sidecar_path, permutation=order)

    bin_path = tmp_path / "asset_reordered.provenance.bin"
    on_disk = np.frombuffer(bin_path.read_bytes(), dtype=np.uint8)

    expected = encode_provenance_u8(provenance[order])
    np.testing.assert_array_equal(on_disk, expected)

    # And the geometry at position i in the exported splat corresponds to the
    # provenance value at position i in the sidecar buffer.
    loaded_provenance = read_provenance_sidecar(sidecar_path)
    for i in range(n):
        original_index = order[i]
        np.testing.assert_allclose(
            loaded_provenance[i], provenance[original_index], atol=1.0 / 255.0
        )
