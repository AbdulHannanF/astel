from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np

from astel_splat_io.cloud import SplatCloud
from astel_splat_io.sog import SOG_VERSION, read_sog, write_sog

# Quantization tolerances: 256-entry quantile codebooks for scales/sh0, plus
# the 16-bit log-domain position encoding and smallest-three quaternions.
SCALE_CODEBOOK_TOL = 0.15
COLOR_CODEBOOK_TOL = 0.15
ROTATION_COMPONENT_TOL = (0.70710678118654752440 / 255.0) * 2.0


def test_sog_meta_json_shape(tmp_path: Path, small_cloud: SplatCloud) -> None:
    out = tmp_path / "cloud.sog"
    write_sog(small_cloud, out)

    with zipfile.ZipFile(out) as zf:
        names = set(zf.namelist())
        meta = json.loads(zf.read("meta.json").decode("utf-8"))

    assert {
        "meta.json",
        "means_l.webp",
        "means_u.webp",
        "scales.webp",
        "quats.webp",
        "sh0.webp",
    } <= names

    assert meta["version"] == SOG_VERSION
    assert meta["count"] == small_cloud.count
    assert meta["width"] * meta["height"] >= small_cloud.count
    assert len(meta["means"]["mins"]) == 3
    assert len(meta["means"]["maxs"]) == 3
    assert len(meta["scales"]["codebook"]) == 256
    assert len(meta["sh0"]["codebook"]) == 256
    assert "shN" not in meta


def test_sog_round_trip_within_tolerance(
    tmp_path: Path, small_cloud: SplatCloud
) -> None:
    out = tmp_path / "cloud.sog"
    write_sog(small_cloud, out)
    loaded = read_sog(out)

    assert loaded.count == small_cloud.count

    # Positions go through a lossless-WebP-backed 16-bit log-domain encoding;
    # error should be tiny relative to the cloud's coordinate range.
    np.testing.assert_allclose(loaded.positions, small_cloud.positions, atol=1e-2)

    np.testing.assert_allclose(
        loaded.log_scales, small_cloud.log_scales, atol=SCALE_CODEBOOK_TOL
    )
    np.testing.assert_allclose(
        loaded.colors_dc, small_cloud.colors_dc, atol=COLOR_CODEBOOK_TOL
    )

    opacity_alpha_in = 1.0 / (1.0 + np.exp(-small_cloud.opacity.astype(np.float64)))
    opacity_alpha_out = 1.0 / (1.0 + np.exp(-loaded.opacity.astype(np.float64)))
    np.testing.assert_allclose(
        opacity_alpha_out, opacity_alpha_in, atol=1.0 / 255.0 * 1.5
    )


def test_sog_shN_raises_not_implemented(
    tmp_path: Path, small_cloud: SplatCloud
) -> None:
    out = tmp_path / "cloud.sog"
    write_sog(small_cloud, out)

    # Rewrite the bundle with a fabricated shN block to confirm read_sog
    # refuses to silently ignore it.
    data = out.read_bytes()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        meta = json.loads(zf.read("meta.json").decode("utf-8"))
        members = {name: zf.read(name) for name in zf.namelist()}

    meta["shN"] = {"bands": 3, "files": ["shN_centroids.webp", "shN_labels.webp"]}
    members["meta.json"] = json.dumps(meta).encode("utf-8")

    out2 = tmp_path / "cloud_with_shn.sog"
    with zipfile.ZipFile(out2, "w") as zf:
        for name, content in members.items():
            zf.writestr(name, content)

    try:
        read_sog(out2)
    except NotImplementedError:
        pass
    else:
        raise AssertionError("expected NotImplementedError for shN bundle")
