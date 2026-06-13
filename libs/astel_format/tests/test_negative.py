"""Negative-path tests: missing files, path traversal, schema failures."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from astel_format import (
    AstelPackage,
    AstelValidationError,
    HallucinationReport,
    PathSecurityError,
    QualityReport,
    ScaleConfidence,
    build_minimal_package,
)
from astel_format.models import FileRef, GeometricError
from astel_format.paths import validate_member_path


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


def test_missing_referenced_file_is_error(
    small_ply_path: Path, small_ply_count: int
) -> None:
    pkg = _build_package(small_ply_path, small_ply_count)
    del pkg.files["layers/l3_refined/splats.ply"]

    with pytest.raises(AstelValidationError, match="missing file"):
        pkg.validate()


def test_path_traversal_in_file_ref_is_rejected(
    small_ply_path: Path, small_ply_count: int
) -> None:
    pkg = _build_package(small_ply_path, small_ply_count)

    l3 = pkg.manifest.layers.l3
    assert l3 is not None
    assert l3.files is not None
    l3.files[0] = FileRef(path="../escape/splats.ply", role="master", format="ply")
    pkg.files["../escape/splats.ply"] = pkg.files.pop("layers/l3_refined/splats.ply")

    with pytest.raises(PathSecurityError, match="traversal"):
        pkg.validate()


def test_absolute_path_in_file_ref_is_rejected(
    small_ply_path: Path, small_ply_count: int
) -> None:
    pkg = _build_package(small_ply_path, small_ply_count)

    l3 = pkg.manifest.layers.l3
    assert l3 is not None
    assert l3.files is not None
    l3.files[0] = FileRef(path="/etc/passwd", role="master", format="ply")
    pkg.files["/etc/passwd"] = pkg.files.pop("layers/l3_refined/splats.ply")

    with pytest.raises(PathSecurityError, match="absolute"):
        pkg.validate()


def test_validate_member_path_rejects_backslashes_and_empty_segments() -> None:
    with pytest.raises(PathSecurityError):
        validate_member_path("layers\\l3_refined\\splats.ply")
    with pytest.raises(PathSecurityError):
        validate_member_path("")
    with pytest.raises(PathSecurityError):
        validate_member_path("layers//splats.ply")
    # Valid path is returned unchanged.
    assert (
        validate_member_path("layers/l3_refined/splats.ply")
        == "layers/l3_refined/splats.ply"
    )


def test_manifest_failing_schema_validation_raises(
    small_ply_path: Path, small_ply_count: int
) -> None:
    pkg = _build_package(small_ply_path, small_ply_count)

    # Remove a required top-level key.
    manifest_dict = pkg.to_manifest_dict()
    del manifest_dict["provenance"]

    from astel_format.schema_validation import validate_manifest_dict

    with pytest.raises(AstelValidationError, match="provenance"):
        validate_manifest_dict(manifest_dict)


def test_reading_zip_with_missing_referenced_file_errors(
    small_ply_path: Path, small_ply_count: int, tmp_path: Path
) -> None:
    pkg = _build_package(small_ply_path, small_ply_count)
    out = tmp_path / "asset.astel"
    pkg.write(out)

    # Hand-craft a corrupted copy: rewrite without the L3 splats file.
    corrupted = tmp_path / "corrupted.astel"
    with zipfile.ZipFile(out) as src, zipfile.ZipFile(corrupted, "w") as dst:
        for info in src.infolist():
            if info.filename == "layers/l3_refined/splats.ply":
                continue
            data = src.read(info.filename)
            if info.filename == "mimetype":
                dst.writestr(zipfile.ZipInfo("mimetype"), data, zipfile.ZIP_STORED)
            else:
                dst.writestr(info.filename, data)

    with pytest.raises(AstelValidationError, match="missing file"):
        AstelPackage.read(corrupted)


def test_reading_zip_with_unreferenced_extra_file_is_ignored(
    small_ply_path: Path, small_ply_count: int, tmp_path: Path
) -> None:
    pkg = _build_package(small_ply_path, small_ply_count)
    out = tmp_path / "asset.astel"
    pkg.write(out)

    extended = tmp_path / "extended.astel"
    with zipfile.ZipFile(out) as src, zipfile.ZipFile(extended, "w") as dst:
        for info in src.infolist():
            data = src.read(info.filename)
            if info.filename == "mimetype":
                dst.writestr(zipfile.ZipInfo("mimetype"), data, zipfile.ZIP_STORED)
            else:
                dst.writestr(info.filename, data)
        dst.writestr("scratch/notes.txt", b"tooling scratch space")

    loaded = AstelPackage.read(extended)
    assert "scratch/notes.txt" not in loaded.files


def test_mimetype_must_be_first_entry(
    small_ply_path: Path, small_ply_count: int, tmp_path: Path
) -> None:
    pkg = _build_package(small_ply_path, small_ply_count)
    out = tmp_path / "asset.astel"
    pkg.write(out)

    reordered = tmp_path / "reordered.astel"
    with zipfile.ZipFile(out) as src, zipfile.ZipFile(reordered, "w") as dst:
        infos = src.infolist()
        # Write manifest.json first, mimetype second.
        for info in sorted(infos, key=lambda i: i.filename != "manifest.json"):
            data = src.read(info.filename)
            if info.filename == "mimetype":
                dst.writestr(zipfile.ZipInfo("mimetype"), data, zipfile.ZIP_STORED)
            else:
                dst.writestr(info.filename, data)

    with pytest.raises(AstelValidationError, match="first zip entry"):
        AstelPackage.read(reordered)


def test_mimetype_bytes_must_match_exactly(
    small_ply_path: Path, small_ply_count: int, tmp_path: Path
) -> None:
    pkg = _build_package(small_ply_path, small_ply_count)
    out = tmp_path / "asset.astel"
    pkg.write(out)

    bad = tmp_path / "bad_mimetype.astel"
    with zipfile.ZipFile(out) as src, zipfile.ZipFile(bad, "w") as dst:
        for info in src.infolist():
            data = src.read(info.filename)
            if info.filename == "mimetype":
                dst.writestr(
                    zipfile.ZipInfo("mimetype"), b"application/zip", zipfile.ZIP_STORED
                )
            else:
                dst.writestr(info.filename, data)

    with pytest.raises(AstelValidationError, match="mimetype"):
        AstelPackage.read(bad)
