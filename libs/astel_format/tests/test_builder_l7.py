"""Tests for L7 dynamics layer binding in build_minimal_package."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from astel_format import AstelPackage, build_minimal_package
from astel_format.models import (
    GeometricError,
    HallucinationReport,
    QualityReport,
    ScaleConfidence,
)
from astel_format.schema_validation import validate_manifest_dict


def _make_quality_report() -> QualityReport:
    return QualityReport.model_validate(
        {
            "geometric_error": GeometricError(
                units="mm",
                reference_layer="l1",
                chamfer_mm=None,
                reason="no measured reference",
            ),
            "scale_confidence": ScaleConfidence(
                meters_per_unit=1.0, ci_low=0.8, ci_high=1.2
            ),
            "hallucination": HallucinationReport(
                measured_fraction=0.0, generated_fraction=1.0
            ),
            "origin": "generated",
        }
    )


def _build_base(
    small_ply_path: Path,
    small_ply_count: int,
    **kwargs: object,
) -> AstelPackage:
    return build_minimal_package(
        asset_id="018f6e2e-0000-7000-8000-000000000l07",
        created="2026-06-19T00:00:00Z",
        generator_name="astel-test",
        generator_version="0.1.0",
        source_modality="video",
        l3_ply_path=small_ply_path,
        l3_count=small_ply_count,
        l3_provenance=[0.5] * small_ply_count,
        quality_report=_make_quality_report(),
        **kwargs,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def deformation_bin(tmp_path: Path) -> Path:
    """Tiny fake deformation field binary."""
    p = tmp_path / "deformation.bin"
    p.write_bytes(b"\x00\x01\x02\x03\xff")
    return p


@pytest.fixture
def timeline_json(tmp_path: Path) -> Path:
    """Minimal timeline JSON."""
    p = tmp_path / "timeline.json"
    p.write_text(
        json.dumps(
            {
                "fps": 30,
                "frame_count": 60,
                "duration_s": 2.0,
                "loop": False,
                "keyframes": [],
            }
        )
    )
    return p


# ---------------------------------------------------------------------------
# Happy-path: L7 binds correctly
# ---------------------------------------------------------------------------


def test_l7_dynamics_binds_and_validates(
    small_ply_path: Path,
    small_ply_count: int,
    deformation_bin: Path,
    timeline_json: Path,
    tmp_path: Path,
) -> None:
    pkg = _build_base(
        small_ply_path,
        small_ply_count,
        l7_deformation_path=deformation_bin,
        l7_timeline_path=timeline_json,
    )

    # manifest layer is populated before writing
    l7 = pkg.manifest.layers.l7
    assert l7 is not None
    assert l7.kind == "dynamics"
    assert l7.status == "present"
    assert l7.derived_from == ["l3"]

    # dynamics sub-block
    assert l7.dynamics is not None
    assert l7.dynamics.representation == "deformation_field"  # default
    assert l7.dynamics.deformation_path == "layers/l7_dynamics/deformation.bin"
    assert l7.dynamics.timeline_path == "layers/l7_dynamics/timeline.json"

    # both FileRefs present with correct roles
    assert l7.files is not None
    roles = {f.role for f in l7.files}
    assert roles == {"deformation", "timeline"}
    formats = {f.format for f in l7.files}
    assert "bin" in formats
    assert "json" in formats

    # raw file bytes are embedded in the package
    def_key = "layers/l7_dynamics/deformation.bin"
    tl_key = "layers/l7_dynamics/timeline.json"
    assert pkg.files[def_key] == deformation_bin.read_bytes()
    assert pkg.files[tl_key] == timeline_json.read_bytes()

    # full manifest dict validates against JSON Schema
    validate_manifest_dict(pkg.to_manifest_dict())


def test_l7_explicit_representation(
    small_ply_path: Path,
    small_ply_count: int,
    deformation_bin: Path,
    timeline_json: Path,
    tmp_path: Path,
) -> None:
    pkg = _build_base(
        small_ply_path,
        small_ply_count,
        l7_deformation_path=deformation_bin,
        l7_timeline_path=timeline_json,
        l7_representation="keyframes",
    )
    l7 = pkg.manifest.layers.l7
    assert l7 is not None
    assert l7.dynamics is not None
    assert l7.dynamics.representation == "keyframes"


# ---------------------------------------------------------------------------
# Round-trip: write → read back, L7 survives
# ---------------------------------------------------------------------------


def test_l7_round_trip(
    small_ply_path: Path,
    small_ply_count: int,
    deformation_bin: Path,
    timeline_json: Path,
    tmp_path: Path,
) -> None:
    pkg = _build_base(
        small_ply_path,
        small_ply_count,
        l7_deformation_path=deformation_bin,
        l7_timeline_path=timeline_json,
        l7_representation="baked_per_frame",
    )

    out = tmp_path / "with_l7.astel"
    pkg.write(out)
    loaded = AstelPackage.read(out)

    l7 = loaded.manifest.layers.l7
    assert l7 is not None
    assert l7.kind == "dynamics"
    assert l7.status == "present"
    assert l7.derived_from == ["l3"]
    assert l7.dynamics is not None
    assert l7.dynamics.representation == "baked_per_frame"
    assert l7.dynamics.deformation_path == "layers/l7_dynamics/deformation.bin"
    assert l7.dynamics.timeline_path == "layers/l7_dynamics/timeline.json"

    # files survive round-trip with correct bytes
    def_key = "layers/l7_dynamics/deformation.bin"
    tl_key = "layers/l7_dynamics/timeline.json"
    assert loaded.files[def_key] == deformation_bin.read_bytes()
    assert loaded.files[tl_key] == timeline_json.read_bytes()

    # JSON Schema still validates
    validate_manifest_dict(loaded.to_manifest_dict())


# ---------------------------------------------------------------------------
# Backward compatibility: no L7 params → package unchanged
# ---------------------------------------------------------------------------


def test_no_l7_params_omits_layer(
    small_ply_path: Path,
    small_ply_count: int,
    tmp_path: Path,
) -> None:
    pkg = _build_base(small_ply_path, small_ply_count)
    assert pkg.manifest.layers.l7 is None
    out = tmp_path / "no_l7.astel"
    pkg.write(out)
    loaded = AstelPackage.read(out)
    assert loaded.manifest.layers.l7 is None


# ---------------------------------------------------------------------------
# Error: partial L7 supply raises ValueError
# ---------------------------------------------------------------------------


def test_l7_only_deformation_raises(
    small_ply_path: Path,
    small_ply_count: int,
    deformation_bin: Path,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="l7_timeline_path"):
        _build_base(
            small_ply_path,
            small_ply_count,
            l7_deformation_path=deformation_bin,
            # l7_timeline_path intentionally omitted
        )


def test_l7_only_timeline_raises(
    small_ply_path: Path,
    small_ply_count: int,
    timeline_json: Path,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="l7_deformation_path"):
        _build_base(
            small_ply_path,
            small_ply_count,
            l7_timeline_path=timeline_json,
            # l7_deformation_path intentionally omitted
        )
