"""Golden round-trip tests for AstelPackage (manifest-v0.md section 1)."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from astel_format import (
    MIMETYPE_BYTES,
    AstelPackage,
    HallucinationReport,
    QualityReport,
    ScaleConfidence,
    build_minimal_package,
)
from astel_format.models import GeometricError


def _make_quality_report() -> QualityReport:
    return QualityReport(
        geometric_error=GeometricError(
            units="mm",
            reference_layer="l1",
            chamfer_mm=None,
            reason="no measured reference",
        ),
        scale_confidence=ScaleConfidence(meters_per_unit=1.0, ci_low=0.8, ci_high=1.2),
        hallucination=HallucinationReport(
            measured_fraction=0.0, generated_fraction=1.0
        ),
    )


def _build_package(small_ply_path: Path, small_ply_count: int) -> AstelPackage:
    return build_minimal_package(
        asset_id="018f6e2e-0000-7000-8000-000000000000",
        created="2026-06-13T00:00:00Z",
        generator_name="astel-pipeline-stub",
        generator_version="0.1.0",
        source_modality="text",
        l3_ply_path=small_ply_path,
        l3_count=small_ply_count,
        l3_provenance=[1.0] * small_ply_count,
        quality_report=_make_quality_report(),
    )


def test_mimetype_is_first_entry_and_stored(
    small_ply_path: Path, small_ply_count: int, tmp_path: Path
) -> None:
    pkg = _build_package(small_ply_path, small_ply_count)
    out = tmp_path / "asset.astel"
    pkg.write(out)

    with zipfile.ZipFile(out) as zf:
        infos = zf.infolist()
        assert infos[0].filename == "mimetype"
        assert infos[0].compress_type == zipfile.ZIP_STORED
        assert zf.read("mimetype") == MIMETYPE_BYTES


def test_manifest_validates_against_schema(
    small_ply_path: Path, small_ply_count: int, tmp_path: Path
) -> None:
    pkg = _build_package(small_ply_path, small_ply_count)
    out = tmp_path / "asset.astel"
    pkg.write(out)

    with zipfile.ZipFile(out) as zf:
        manifest = json.loads(zf.read("manifest.json"))

    # Re-validation via the public API (raises on failure).
    from astel_format.schema_validation import validate_manifest_dict

    validate_manifest_dict(manifest)
    assert manifest["format_version"] == "0.1.0"


def test_provenance_accessor_count_matches_gaussian_count_and_is_aligned(
    small_ply_path: Path, small_ply_count: int, tmp_path: Path
) -> None:
    pkg = _build_package(small_ply_path, small_ply_count)
    out = tmp_path / "asset.astel"
    pkg.write(out)

    loaded = AstelPackage.read(out)
    manifest = loaded.manifest

    l3 = manifest.layers.l3
    assert l3 is not None
    assert l3.count == small_ply_count

    [channel] = manifest.provenance.channels
    assert channel.layer == "l3"
    assert channel.count == small_ply_count

    accessor = manifest.buffers.accessors[channel.accessor]
    assert accessor.count == small_ply_count
    assert accessor.component_type == "UNORM8"
    assert accessor.type == "SCALAR"

    # Index-aligned: the provenance buffer has exactly `count` bytes (UNORM8
    # = 1 byte/primitive, tightly packed, manifest-v0.md section 5.2).
    buffer_view = manifest.buffers.buffer_views[accessor.buffer_view]
    buffer_entry = manifest.buffers.buffers[buffer_view.buffer]
    prov_bytes = loaded.files[buffer_entry.uri]
    assert len(prov_bytes) == small_ply_count
    assert all(b == 255 for b in prov_bytes)  # provenance 1.0 -> UNORM8 255


def test_unknown_keys_and_extensions_survive_roundtrip(
    small_ply_path: Path, small_ply_count: int, tmp_path: Path
) -> None:
    pkg = _build_package(small_ply_path, small_ply_count)

    # Inject an unknown additive key inside `extras` (free-form, ignored by
    # validators) plus a vendor `extensions` block, re-validate into the
    # model (extra="allow"), and confirm they survive write -> read -> write
    # at the JSON level. Top-level objects like `astel` have
    # `additionalProperties: false`, so genuinely unknown keys there are a
    # schema validation error (the schema wins, per manifest-v0.md); the
    # forward-migration escape hatches are `extras`/`extensions`.
    manifest_dict = pkg.to_manifest_dict()
    manifest_dict["extensions"] = {
        "vendor_acme": {"widget_count": 7, "nested": {"a": [1, 2, 3]}}
    }
    manifest_dict["extras"] = {
        "note": "free-form",
        "unknown_future_field": "preserve-me",
    }

    from astel_format.models import Manifest

    pkg.manifest = Manifest.model_validate(manifest_dict)

    out = tmp_path / "asset.astel"
    pkg.write(out)

    loaded = AstelPackage.read(out)
    round_tripped = loaded.to_manifest_dict()

    assert round_tripped["extensions"] == {
        "vendor_acme": {"widget_count": 7, "nested": {"a": [1, 2, 3]}}
    }
    assert round_tripped["extras"] == {
        "note": "free-form",
        "unknown_future_field": "preserve-me",
    }

    # Byte-for-byte at the JSON level (same structure, re-serialised).
    assert json.dumps(round_tripped, sort_keys=True) == json.dumps(
        manifest_dict, sort_keys=True
    )


def test_l0_layer_optional_minimal_package_has_only_l3(
    small_ply_path: Path, small_ply_count: int, tmp_path: Path
) -> None:
    pkg = _build_package(small_ply_path, small_ply_count)
    out = tmp_path / "asset.astel"
    pkg.write(out)

    loaded = AstelPackage.read(out)
    assert loaded.manifest.layers.l3 is not None
    assert loaded.manifest.layers.l0 is None
    manifest_dict = loaded.to_manifest_dict()
    assert "l0" not in manifest_dict["layers"]
