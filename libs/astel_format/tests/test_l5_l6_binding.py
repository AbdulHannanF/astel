"""Tests for L5/L6 layer binding in build_minimal_package."""

from __future__ import annotations

import json
import struct
from pathlib import Path

from astel_format import AstelPackage, build_minimal_package
from astel_format.models import (
    GeometricError,
    HallucinationReport,
    LayerArticulation,
    QualityReport,
    ScaleConfidence,
)
from astel_format.schema_validation import validate_manifest_dict


def _make_quality_report(origin: str | None = None) -> QualityReport:
    kwargs: dict[str, object] = {
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
    }
    if origin is not None:
        kwargs["origin"] = origin
    return QualityReport.model_validate(kwargs)


def _make_stl_bytes() -> bytes:
    """Minimal binary STL: header + 1 triangle."""
    header = b"\x00" * 80
    n_triangles = struct.pack("<I", 1)
    normal = struct.pack("<3f", 0.0, 0.0, 1.0)
    v0 = struct.pack("<3f", 0.0, 0.0, 0.0)
    v1 = struct.pack("<3f", 1.0, 0.0, 0.0)
    v2 = struct.pack("<3f", 0.0, 1.0, 0.0)
    attr = struct.pack("<H", 0)
    return header + n_triangles + normal + v0 + v1 + v2 + attr


def _build_base(
    tmp_path: Path,
    small_ply_path: Path,
    small_ply_count: int,
    **kwargs: object,
) -> AstelPackage:
    return build_minimal_package(
        asset_id="018f6e2e-0000-7000-8000-000000000001",
        created="2026-06-18T00:00:00Z",
        generator_name="astel-test",
        generator_version="0.1.0",
        source_modality="text",
        l3_ply_path=small_ply_path,
        l3_count=small_ply_count,
        l3_provenance=[0.0] * small_ply_count,
        quality_report=_make_quality_report(origin="generated"),
        **kwargs,  # type: ignore[arg-type]
    )


# ---- without L5/L6: baseline still validates --------------------------------


def test_package_without_l5_l6_validates(
    small_ply_path: Path, small_ply_count: int, tmp_path: Path
) -> None:
    pkg = _build_base(tmp_path, small_ply_path, small_ply_count)
    out = tmp_path / "no_l5_l6.astel"
    pkg.write(out)
    loaded = AstelPackage.read(out)
    assert loaded.manifest.layers.l5 is None
    assert loaded.manifest.layers.l6 is None


# ---- QualityReport origin field validates -----------------------------------


def test_quality_report_with_origin_validates(
    small_ply_path: Path, small_ply_count: int, tmp_path: Path
) -> None:
    pkg = _build_base(tmp_path, small_ply_path, small_ply_count)
    out = tmp_path / "with_origin.astel"
    pkg.write(out)
    loaded = AstelPackage.read(out)
    assert loaded.manifest.quality_report.origin == "generated"


def test_quality_report_without_origin_validates(
    small_ply_path: Path, small_ply_count: int, tmp_path: Path
) -> None:
    # origin is optional — build without passing it so it stays unset/omitted.
    qr = _make_quality_report()  # no origin kwarg -> field is unset -> omitted
    pkg = build_minimal_package(
        asset_id="018f6e2e-0000-7000-8000-000000000002",
        created="2026-06-18T00:00:00Z",
        generator_name="astel-test",
        generator_version="0.1.0",
        source_modality="text",
        l3_ply_path=small_ply_path,
        l3_count=small_ply_count,
        l3_provenance=[0.0] * small_ply_count,
        quality_report=qr,
    )
    out = tmp_path / "no_origin.astel"
    pkg.write(out)
    loaded = AstelPackage.read(out)
    # When origin is absent from the manifest, the model default is None.
    assert loaded.manifest.quality_report.origin is None


def test_quality_report_stub_origin_validates(
    small_ply_path: Path, small_ply_count: int, tmp_path: Path
) -> None:
    pkg = build_minimal_package(
        asset_id="018f6e2e-0000-7000-8000-000000000003",
        created="2026-06-18T00:00:00Z",
        generator_name="astel-test",
        generator_version="0.1.0",
        source_modality="text",
        l3_ply_path=small_ply_path,
        l3_count=small_ply_count,
        l3_provenance=[0.0] * small_ply_count,
        quality_report=_make_quality_report(origin="stub"),
    )
    out = tmp_path / "stub_origin.astel"
    pkg.write(out)
    loaded = AstelPackage.read(out)
    assert loaded.manifest.quality_report.origin == "stub"


# ---- L4 appearance binding --------------------------------------------------


def test_l4_appearance_binds_and_validates(
    small_ply_path: Path, small_ply_count: int, tmp_path: Path
) -> None:
    env_path = tmp_path / "l4-env.json"
    env_path.write_text(json.dumps({"schema": "astel.l4-env/v0", "sh_rgb": []}))
    summary_path = tmp_path / "l4.json"
    summary_path.write_text(json.dumps({"schema": "astel.l4-appearance/v0"}))
    # The albedo "baked-PBR" cloud is just another .ply (reuse the L3 fixture).
    albedo_path = small_ply_path

    pkg = _build_base(
        tmp_path,
        small_ply_path,
        small_ply_count,
        l4_env_path=env_path,
        l4_albedo_path=albedo_path,
        l4_summary_path=summary_path,
    )
    out = tmp_path / "l4.astel"
    pkg.write(out)

    loaded = AstelPackage.read(out)
    l4 = loaded.manifest.layers.l4
    assert l4 is not None
    assert l4.kind == "appearance"
    assert l4.status == "present"
    assert l4.derived_from == ["l3"]
    assert l4.appearance is not None
    assert l4.appearance.bound_to == "l3"
    assert l4.appearance.env_map_path == "layers/l4_appearance/l4-env.json"
    assert l4.appearance.baked_pbr_path is not None
    assert l4.appearance.baked_pbr_path.endswith(".ply")
    roles = {f.role for f in (l4.files or [])}
    assert {"env_map", "baked_preview"} <= roles
    # The env + albedo bytes are embedded in the package.
    assert "layers/l4_appearance/l4-env.json" in loaded.files


def test_package_without_l4_omits_layer(
    small_ply_path: Path, small_ply_count: int, tmp_path: Path
) -> None:
    pkg = _build_base(tmp_path, small_ply_path, small_ply_count)
    out = tmp_path / "no_l4.astel"
    pkg.write(out)
    assert AstelPackage.read(out).manifest.layers.l4 is None


# ---- L5 binding -------------------------------------------------------------


def test_l5_isosurface_only_validates(
    small_ply_path: Path, small_ply_count: int, tmp_path: Path
) -> None:
    stl_path = tmp_path / "l5.stl"
    stl_path.write_bytes(_make_stl_bytes())

    pkg = _build_base(
        tmp_path, small_ply_path, small_ply_count, l5_isosurface_path=stl_path
    )
    out = tmp_path / "l5_only.astel"
    pkg.write(out)

    loaded = AstelPackage.read(out)
    l5 = loaded.manifest.layers.l5
    assert l5 is not None
    assert l5.kind == "collision"
    assert l5.status == "present"
    assert l5.derived_from == ["l3"]
    assert l5.collision is not None
    iso = l5.collision.isosurface
    assert iso is not None
    assert iso.print_physics_only is True
    # FileRef present with role=isosurface
    assert l5.files is not None
    roles = {f.role for f in l5.files}
    assert "isosurface" in roles


def test_l5_full_set_validates(
    small_ply_path: Path, small_ply_count: int, tmp_path: Path
) -> None:
    stl_path = tmp_path / "l5.stl"
    stl_path.write_bytes(_make_stl_bytes())

    mass_path = tmp_path / "l5-mass.json"
    mass_path.write_text(json.dumps({"volume": 0.001, "total_mass_kg": 0.7}))

    # Use a .npz as convex set (no trimesh needed for this test)
    convex_path = tmp_path / "l5-convex.npz"
    convex_path.write_bytes(b"fake-npz")

    pkg = _build_base(
        tmp_path,
        small_ply_path,
        small_ply_count,
        l5_isosurface_path=stl_path,
        l5_convex_set_path=convex_path,
        l5_mass_props_path=mass_path,
    )
    out = tmp_path / "l5_full.astel"
    pkg.write(out)

    loaded = AstelPackage.read(out)
    l5 = loaded.manifest.layers.l5
    assert l5 is not None
    assert l5.collision is not None
    assert l5.collision.convex_set_path is not None
    assert l5.collision.mass_props_path is not None
    roles = {f.role for f in (l5.files or [])}
    assert roles >= {"isosurface", "convex_set", "mass_props"}


def test_l5_not_present_when_iso_omitted(
    small_ply_path: Path, small_ply_count: int, tmp_path: Path
) -> None:
    pkg = _build_base(tmp_path, small_ply_path, small_ply_count)
    assert pkg.manifest.layers.l5 is None


# ---- L6 binding -------------------------------------------------------------


def _make_l6_json(tmp_path: Path) -> Path:
    data = {
        "schema": "astel.physics-material/v0",
        "status": "ok",
        "spec": {
            "regions": [
                {
                    "region": "body",
                    "material": "oak wood",
                    "material_class": "rigid",
                    "density_kg_m3": 700.0,
                    "friction": 0.5,
                    "restitution": 0.3,
                }
            ],
            "articulation": [],
            "notes": "single rigid piece",
        },
    }
    p = tmp_path / "l6.json"
    p.write_text(json.dumps(data))
    return p


def test_l6_regions_only_validates(
    small_ply_path: Path, small_ply_count: int, tmp_path: Path
) -> None:
    l6_path = _make_l6_json(tmp_path)

    pkg = _build_base(
        tmp_path, small_ply_path, small_ply_count, l6_regions_path=l6_path
    )
    out = tmp_path / "l6_only.astel"
    pkg.write(out)

    loaded = AstelPackage.read(out)
    l6 = loaded.manifest.layers.l6
    assert l6 is not None
    assert l6.kind == "physics_material"
    assert l6.status == "present"
    assert l6.derived_from == ["l3"]
    assert l6.physics_material is not None
    assert l6.physics_material.regions_path is not None
    files = l6.files or []
    assert any(f.role == "regions" for f in files)


def test_l6_with_articulation_validates(
    small_ply_path: Path, small_ply_count: int, tmp_path: Path
) -> None:
    l6_path = _make_l6_json(tmp_path)
    articulation = [
        LayerArticulation(type="revolute", parent_region=0, child_region=1)
    ]

    pkg = _build_base(
        tmp_path,
        small_ply_path,
        small_ply_count,
        l6_regions_path=l6_path,
        l6_articulation=articulation,
    )
    out = tmp_path / "l6_artic.astel"
    pkg.write(out)

    loaded = AstelPackage.read(out)
    l6 = loaded.manifest.layers.l6
    assert l6 is not None
    assert l6.physics_material is not None
    artic = l6.physics_material.articulation
    assert artic is not None and len(artic) == 1
    assert artic[0].type == "revolute"


def test_l5_and_l6_together_validate(
    small_ply_path: Path, small_ply_count: int, tmp_path: Path
) -> None:
    stl_path = tmp_path / "l5.stl"
    stl_path.write_bytes(_make_stl_bytes())
    l6_path = _make_l6_json(tmp_path)

    pkg = _build_base(
        tmp_path,
        small_ply_path,
        small_ply_count,
        l5_isosurface_path=stl_path,
        l6_regions_path=l6_path,
    )
    out = tmp_path / "l5_l6.astel"
    pkg.write(out)

    loaded = AstelPackage.read(out)
    assert loaded.manifest.layers.l5 is not None
    assert loaded.manifest.layers.l6 is not None
    # L3 is still intact
    assert loaded.manifest.layers.l3 is not None

    # Full manifest dict validates against the JSON Schema
    validate_manifest_dict(loaded.to_manifest_dict())
