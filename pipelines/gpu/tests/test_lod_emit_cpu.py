"""CPU tests for LOD emission in the GPU producer (no torch/gsplat/CUDA).

Two test sub-areas:
1. ``_write_lod`` integration: builds an in-memory SplatCloud, calls the helper,
   and asserts descriptor schema + "full" tier presence.
2. ``generate_lod_indices`` unit test on a small cloud with a synthetic target
   count (N//2) — does NOT require a cloud >= 100_000. Documents that the
   producer's named TIER_BUDGETS tiers only trigger when the cloud exceeds the
   smallest budget (100_000 for "lowpoly").
"""

from __future__ import annotations

import json
from pathlib import Path

import astel_lod
import numpy as np
import pytest
from astel_splat_io.cloud import SplatCloud

from astel_gpu.packaging import _write_lod


def _make_cloud(n: int, seed: int = 0) -> SplatCloud:
    """Minimal synthetic SplatCloud — unit quats, random positions/scales/opacity."""
    rng = np.random.default_rng(seed)
    quats = np.zeros((n, 4), dtype=np.float32)
    quats[:, 0] = 1.0
    return SplatCloud(
        positions=rng.standard_normal((n, 3)).astype(np.float32),
        colors_dc=rng.standard_normal((n, 3)).astype(np.float32),
        opacity=rng.standard_normal(n).astype(np.float32),
        log_scales=(-3.0 + rng.standard_normal((n, 3))).astype(np.float32),
        quats=quats,
    )


# ---------------------------------------------------------------------------
# 1. _write_lod integration test — small cloud (200 splats)
# ---------------------------------------------------------------------------


def test_write_lod_creates_descriptor(tmp_path: Path) -> None:
    """_write_lod emits l3.lod.json with the correct schema and a 'full' tier."""
    cloud = _make_cloud(200)
    descriptor = _write_lod(cloud, tmp_path)

    # Return value is not None.
    assert descriptor is not None

    # Descriptor schema.
    assert descriptor["schema"] == "astel.lod/v0"

    # Tiers list is non-empty.
    tiers = descriptor["tiers"]
    assert isinstance(tiers, list)
    assert len(tiers) >= 1

    # 'full' tier is always present, with correct count and file.
    full_tiers = [t for t in tiers if t["name"] == "full"]
    assert len(full_tiers) == 1
    assert full_tiers[0]["count"] == cloud.count
    assert full_tiers[0]["file"] == "l3.ply"


def test_write_lod_json_file_exists(tmp_path: Path) -> None:
    """l3.lod.json is written to disk and is valid JSON."""
    cloud = _make_cloud(200)
    _write_lod(cloud, tmp_path)

    lod_json_path = tmp_path / "l3.lod.json"
    assert lod_json_path.exists()
    assert lod_json_path.stat().st_size > 0

    # Loadable JSON.
    data = json.loads(lod_json_path.read_text(encoding="utf-8"))
    assert data["schema"] == "astel.lod/v0"


def test_write_lod_small_cloud_no_downsample_ply(tmp_path: Path) -> None:
    """For a cloud smaller than the smallest TIER_BUDGETS value (100_000),
    no downsampled PLY files are written — only the descriptor + 'full' tier.

    Documents: TIER_BUDGETS["lowpoly"] == 100_000, so a 200-splat cloud
    produces NO named downsample tiers (200 < 100_000 is false in the
    'budget < n' check direction — actually 100_000 > 200, so budget >= n,
    so no tier is added). Only 'full' is emitted.
    """
    cloud = _make_cloud(200)
    descriptor = _write_lod(cloud, tmp_path)
    assert descriptor is not None

    # Only the 'full' tier should appear (no budgets are < 200 among
    # TIER_BUDGETS: lowpoly=100k, standard=1M, cinematic=5M — all >= 200).
    tier_names = [t["name"] for t in descriptor["tiers"]]
    assert "full" in tier_names

    # No l3.lod.<name>.ply files emitted.
    lod_plys = list(tmp_path.glob("l3.lod.*.ply"))
    assert lod_plys == [], f"Unexpected LOD PLYs: {lod_plys}"


def test_write_lod_returns_none_on_bad_cloud(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_write_lod returns None on failure — best-effort, never fatal.

    Patches build_lod_descriptor (always called, regardless of cloud size)
    to raise, which exercises the broad try/except guard.
    """
    cloud = _make_cloud(10)

    # Monkeypatch build_lod_descriptor to raise to simulate an unexpected error.
    def _bad_build(*args: object, **kwargs: object) -> None:
        raise RuntimeError("simulated descriptor build failure")

    monkeypatch.setattr(astel_lod, "build_lod_descriptor", _bad_build)
    result = _write_lod(cloud, tmp_path)
    assert result is None


# ---------------------------------------------------------------------------
# 2. generate_lod_indices unit test — subsample path on a small cloud
# ---------------------------------------------------------------------------


def test_generate_lod_indices_subsample_count(tmp_path: Path) -> None:
    """generate_lod_indices returns the right count and reordered() matches.

    This tests the downsample path WITHOUT a 100k cloud: we use N=200 and a
    synthetic target of N//2 = 100.  The cloud's reordered(indices) must have
    exactly 100 splats.

    Note: the producer's _write_lod only emits named TIER_BUDGETS tiers when
    budget STRICTLY < N (== the cloud count), so this unit test intentionally
    bypasses _write_lod and calls astel_lod directly to exercise the index
    generation code path.
    """
    cloud = _make_cloud(200)
    n = cloud.count
    target = n // 2  # 100

    # generate_lod_indices returns a list with one index array per target count.
    index_arrays = astel_lod.generate_lod_indices(
        cloud.opacity, cloud.log_scales, [target]
    )
    assert len(index_arrays) == 1
    indices = index_arrays[0]

    # Correct length.
    assert len(indices) == target

    # Indices are sorted ascending (a post-condition of generate_lod_indices).
    assert (indices[:-1] < indices[1:]).all()

    # All indices in range.
    assert int(indices.min()) >= 0
    assert int(indices.max()) < n

    # reordered() gives a SplatCloud with the right count.
    sub_cloud = cloud.reordered(indices)
    assert sub_cloud.count == target
    assert sub_cloud.positions.shape == (target, 3)


def test_generate_lod_indices_multiple_targets() -> None:
    """Multiple target counts in one call: all index arrays have correct lengths."""
    cloud = _make_cloud(200)
    n = cloud.count
    targets = [n // 4, n // 2, n * 3 // 4]  # 50, 100, 150

    index_arrays = astel_lod.generate_lod_indices(
        cloud.opacity, cloud.log_scales, targets
    )
    assert len(index_arrays) == len(targets)
    for arr, target in zip(index_arrays, targets, strict=True):
        assert len(arr) == target
