"""CPU tests for the full layer-stack writer seam (no torch/gsplat/CUDA).

``write_layer_stack`` operates on a numpy ``SplatCloud`` + a pydantic
``QualityReport``, so the GPU producer's artifact contract is verified entirely
on CPU: all expected files are emitted, the ``.astel`` package round-trips, and
the honesty fields are intact.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from astel_format.models import LayerArticulation
from astel_format.package import AstelPackage
from astel_splat_io.cloud import SplatCloud

from astel_gpu.packaging import (
    build_engine_setup,
    build_package_quality_report,
    seed_cloud,
    write_layer_stack,
)


def _cloud(n: int) -> SplatCloud:
    rng = np.random.default_rng(0)
    quats = np.zeros((n, 4), dtype=np.float32)
    quats[:, 0] = 1.0
    return SplatCloud(
        positions=rng.standard_normal((n, 3)).astype(np.float32),
        colors_dc=rng.standard_normal((n, 3)).astype(np.float32),
        opacity=rng.standard_normal(n).astype(np.float32),
        log_scales=(-3.0 + rng.standard_normal((n, 3))).astype(np.float32),
        quats=quats,
    )


def _report_dict(count: int) -> dict[str, Any]:
    return {
        "schema": "astel.quality-report/v0",
        "origin": "measured",
        "splats": count,
        "fidelity": {"psnr_db": 25.0},
    }


def test_seed_cloud_strides_l3() -> None:
    cloud = _cloud(240)
    l0 = seed_cloud(cloud, divisor=24)
    assert l0.count == 10  # 240 / 24
    # First seed point is the first L3 point (stride starts at 0).
    assert np.allclose(l0.positions[0], cloud.positions[0])


def test_write_layer_stack_emits_full_contract(tmp_path: Path) -> None:
    cloud = _cloud(500)
    report = build_package_quality_report(
        modality="text", origin_note="unit-test"
    )
    names = write_layer_stack(
        cloud,
        tmp_path,
        task_id="task-abc",
        modality="text",
        prompt="a brass teapot",
        seed=7,
        report_dict=_report_dict(cloud.count),
        package_report=report,
        solidify_l5=False,
    )
    expected = {
        "l0.ply",
        "l3.ply",
        "l3.lod.json",
        "l3.spz",
        "l3.sog",
        "l3.glb",
        "l4-albedo.ply",
        "l4-env.json",
        "l4.json",
        "l4-relight.json",
        "engine.json",
        "package.astel",
        "quality-report.json",
    }
    assert set(names) == expected
    for name in expected:
        assert (tmp_path / name).stat().st_size > 0


def test_write_layer_stack_emits_loadable_gltf(tmp_path: Path) -> None:
    """The l3.glb export is a real KHR_gaussian_splatting GLB that round-trips."""
    from astel_splat_io.gltf import read_gltf

    cloud = _cloud(300)
    write_layer_stack(
        cloud,
        tmp_path,
        task_id="task-glb",
        modality="text",
        prompt="a brass teapot",
        seed=1,
        report_dict=_report_dict(cloud.count),
        package_report=build_package_quality_report(
            modality="text", origin_note="unit-test"
        ),
        solidify_l5=False,
        appearance_l4=False,
    )
    back = read_gltf(tmp_path / "l3.glb")
    assert back.count == cloud.count
    np.testing.assert_allclose(back.positions, cloud.positions, atol=1e-5)


def test_write_layer_stack_binds_l4_appearance(tmp_path: Path) -> None:
    cloud = _cloud(800)
    report = _report_dict(cloud.count)
    write_layer_stack(
        cloud,
        tmp_path,
        task_id="task-l4",
        modality="image",
        prompt="",
        seed=4,
        report_dict=report,
        package_report=build_package_quality_report(
            modality="image", origin_note="generative"
        ),
        solidify_l5=False,
    )
    # L4 summary threaded into the served report, honestly flagged.
    assert "appearance" in report
    assert report["appearance"]["schema"] == "astel.l4-appearance/v0"
    assert 0.0 <= report["appearance"]["lighting_confidence"] <= 1.0
    # L4 layer bound into the package manifest.
    pkg = AstelPackage.read(tmp_path / "package.astel")
    l4 = pkg.manifest.layers.l4
    assert l4 is not None
    assert l4.kind == "appearance"
    assert l4.appearance is not None
    assert l4.appearance.env_map_path is not None
    assert l4.appearance.baked_pbr_path is not None


def test_appearance_l4_can_be_disabled(tmp_path: Path) -> None:
    cloud = _cloud(200)
    names = write_layer_stack(
        cloud,
        tmp_path,
        task_id="task-no-l4",
        modality="image",
        prompt="",
        seed=5,
        report_dict=_report_dict(cloud.count),
        package_report=build_package_quality_report(
            modality="image", origin_note="generative"
        ),
        solidify_l5=False,
        appearance_l4=False,
    )
    assert not any(n.startswith("l4") for n in names)
    pkg = AstelPackage.read(tmp_path / "package.astel")
    assert pkg.manifest.layers.l4 is None


def test_write_layer_stack_includes_l2_when_given(tmp_path: Path) -> None:
    l3 = _cloud(300)
    l2 = _cloud(900)
    names = write_layer_stack(
        l3,
        tmp_path,
        task_id="gen-1",
        modality="image",
        prompt="",
        seed=1,
        report_dict=_report_dict(l3.count),
        package_report=build_package_quality_report(
            modality="image", origin_note="generative"
        ),
        l2_cloud=l2,
        solidify_l5=False,
    )
    assert "l2.ply" in names
    assert (tmp_path / "l2.ply").stat().st_size > 0


def test_package_round_trips_and_is_honest(tmp_path: Path) -> None:
    cloud = _cloud(120)
    write_layer_stack(
        cloud,
        tmp_path,
        task_id="task-honest",
        modality="image",
        prompt=None or "",
        seed=3,
        report_dict=_report_dict(cloud.count),
        package_report=build_package_quality_report(
            modality="image", origin_note="generative"
        ),
        solidify_l5=False,
    )
    pkg = AstelPackage.read(tmp_path / "package.astel")
    qr = pkg.manifest.quality_report
    assert qr is not None
    # No ground-truth -> geometric error is explicitly None with a reason.
    assert qr.geometric_error is not None
    assert qr.geometric_error.chamfer_mm is None
    assert qr.geometric_error.reason
    # Fully generated, honest provenance.
    assert qr.hallucination is not None
    assert qr.hallucination.measured_fraction == 0.0
    assert qr.hallucination.generated_fraction == 1.0


def test_package_quality_report_scale_is_ungrounded() -> None:
    qr = build_package_quality_report(modality="text", origin_note="x")
    assert qr.scale_confidence is not None
    assert qr.scale_confidence.meters_per_unit == 1.0
    assert qr.scale_confidence.ci_method == "gpu-no-estimate"


def _sphere_cloud(n: int, radius: float = 1.0) -> SplatCloud:
    """A surfel cloud sampling a sphere: thin radial axis so normals point out."""
    i = np.arange(n, dtype=np.float64)
    phi = np.pi * (3.0 - np.sqrt(5.0))
    y = 1.0 - 2.0 * (i + 0.5) / n
    r = np.sqrt(np.clip(1.0 - y * y, 0.0, 1.0))
    theta = phi * i
    pos = np.stack([np.cos(theta) * r, y, np.sin(theta) * r], axis=1) * radius
    # Orient each surfel so its thinnest axis (x in local frame) is radial: build
    # a quaternion rotating +x onto the radial direction is overkill for the test
    # — instead make all axes equal-ish but mark the radial via positions; the
    # producer uses surfel_normals which here just needs a thin axis. Use identity
    # quats with x as the thin axis and rely on centroid-outward orientation.
    quats = np.zeros((n, 4), dtype=np.float32)
    quats[:, 0] = 1.0
    log_scales = np.tile(
        np.array([-6.0, -2.0, -2.0], dtype=np.float32), (n, 1)
    )  # x thinnest
    return SplatCloud(
        positions=pos.astype(np.float32),
        colors_dc=np.zeros((n, 3), dtype=np.float32),
        opacity=np.zeros(n, dtype=np.float32),
        log_scales=log_scales,
        quats=quats,
    )


def test_solidify_l5_emits_stl_and_report_solidity(tmp_path: Path) -> None:
    cloud = _sphere_cloud(4000, radius=1.0)
    report = _report_dict(cloud.count)
    names = write_layer_stack(
        cloud,
        tmp_path,
        task_id="solid-1",
        modality="image",
        prompt="",
        seed=2,
        report_dict=report,
        package_report=build_package_quality_report(
            modality="image", origin_note="generative"
        ),
        solidify_l5=True,
    )
    assert "l5.stl" in names
    assert "l5-mass.json" in names
    assert (tmp_path / "l5.stl").stat().st_size > 84  # header + at least 1 tri
    # Solidity summary threaded into the report.
    assert "solidity" in report
    sol = report["solidity"]
    assert sol["volume"] > 0.0
    assert sol["mesh"]["faces"] > 0
    assert sol["stl"] == "l5.stl"


def _l6_articulated_spec() -> dict[str, Any]:
    return {
        "schema": "astel.physics-material/v0",
        "status": "ok",
        "spec": {
            "regions": [
                {"region": "box", "material": "oak", "material_class": "rigid",
                 "density_kg_m3": 700.0, "friction": 0.5, "restitution": 0.2},
                {"region": "lid", "material": "steel", "material_class": "rigid",
                 "density_kg_m3": 7850.0, "friction": 0.4, "restitution": 0.3},
            ],
            "articulation": [
                {"parent": "box", "child": "lid", "joint_type": "hinge"}
            ],
        },
    }


def test_build_engine_setup_metric_mass_from_l6_join() -> None:
    """With solidify + an L6 mass join, engine.json carries the metric mass_kg,
    model-unit COM/inertia, and per-region friction the plugins set."""
    solidity = {
        "volume": 0.5,
        "center_of_mass": [0.1, 0.2, 0.3],
        "inertia_diagonal": [0.4, 0.5, 0.6],
    }
    l6_mass = {"total_mass_kg": 9.76, "scale_grounded": True}
    regions = [
        {"region": "body", "material": "oak", "density_kg_m3": 700.0,
         "friction": 0.5, "restitution": 0.2},
    ]
    articulation = [
        LayerArticulation(type="revolute", parent_region=0, child_region=1),
    ]
    setup = build_engine_setup(
        meters_per_unit=2.0,
        splat_file="l3.spz",
        solidity=solidity,
        l6_mass=l6_mass,
        l6_regions=regions,
        articulation=articulation,
    )
    assert setup["scale_grounded"] is True
    mp = setup["l5"]["mass_props"]
    assert mp["mass_kg"] == 9.76  # from the L6 join, not model-unit volume
    assert mp["volume_m3"] == 0.5 * (2.0**3)  # metric volume
    assert mp["center_of_mass"] == [0.1, 0.2, 0.3]  # model units (plugin scales)
    assert setup["l6"]["regions"][0]["friction"] == 0.5
    assert setup["l6"]["articulation"][0]["joint_type"] == "revolute"


def test_build_engine_setup_honest_without_l6_mass() -> None:
    """Solidify but no L6 join → mass_kg stays 0.0 (plugins fall back to unit
    mass); a model-unit 'mass at unit density' is never passed off as kg."""
    setup = build_engine_setup(
        meters_per_unit=1.0,
        splat_file="l3.spz",
        solidity={"volume": 0.5, "center_of_mass": [0, 0, 0],
                  "inertia_diagonal": [1, 1, 1]},
        l6_mass=None,
        l6_regions=None,
        articulation=None,
    )
    assert setup["l5"]["mass_props"]["mass_kg"] == 0.0
    assert setup["l6"] is None
    assert setup["scale_grounded"] is False
    assert any("mass_kg unavailable" in n for n in setup["notes"])
    assert any("scale ungrounded" in n for n in setup["notes"])


def test_l6_articulation_and_metric_scale_end_to_end(tmp_path: Path) -> None:
    """A pre-placed l6.json with an articulated, multi-region spec binds through
    write_layer_stack: the URDF-ish joint vocab is mapped to the manifest enum,
    region names resolve to int indices, and a longest_axis_m estimate grounds
    the package scale (meters_per_unit != 1.0).

    Regression guard: a "hinge" joint used to raise inside the best-effort guard
    and silently drop the whole L6 bind. The L6 articulation binding runs
    independently of solidify, so this stays CPU-cheap (no CoACD); the metric
    mass-join math is covered by the compute_l6_masses /
    meters_per_unit_from_longest_axis unit tests."""
    cloud = _cloud(400)
    (tmp_path / "l6.json").write_text(json.dumps(_l6_articulated_spec()))

    write_layer_stack(
        cloud,
        tmp_path,
        task_id="l6-e2e",
        modality="text",
        prompt="a wooden box with a steel lid",
        seed=9,
        report_dict=_report_dict(cloud.count),
        package_report=build_package_quality_report(
            modality="text", origin_note="generative"
        ),
        solidify_l5=False,
        longest_axis_m=0.3,  # grounds meters_per_unit off the model extent
    )

    pkg = AstelPackage.read(tmp_path / "package.astel")
    # Articulation: hinge -> revolute, region names -> int indices.
    l6_layer = pkg.manifest.layers.l6
    assert l6_layer is not None
    assert l6_layer.physics_material is not None
    artic = l6_layer.physics_material.articulation
    assert artic is not None and len(artic) == 1
    assert artic[0].type == "revolute"
    assert artic[0].parent_region == 0  # "box"
    assert artic[0].child_region == 1  # "lid"
    assert artic[0].axis is None  # the LLM gives no axis -> none invented
    # Package scale grounded from the size estimate (not the identity 1.0).
    assert pkg.manifest.scale.meters_per_unit != 1.0
    assert pkg.manifest.coordinate_system.meters_per_unit != 1.0

    # engine.json — the flat physics descriptor the Unity/UE5 plugins consume —
    # carries the SAME data the manifest binds, denormalised. This is the
    # contract the plugins read; assert it reflects real L6 values.
    engine = json.loads((tmp_path / "engine.json").read_text())
    assert engine["schema"] == "astel.engine-setup/v0"
    assert engine["meters_per_unit"] != 1.0
    assert engine["scale_grounded"] is True
    assert engine["splat_file"] == "l3.spz"
    # No solidify in this CPU-cheap test → no L5 mass props, honestly null.
    assert engine["l5"] is None
    names = [r["name"] for r in engine["l6"]["regions"]]
    assert names == ["box", "lid"]
    assert engine["l6"]["regions"][0]["friction"] == 0.5
    assert engine["l6"]["regions"][1]["density_kg_m3"] == 7850.0
    art = engine["l6"]["articulation"]
    assert len(art) == 1
    assert art[0]["joint_type"] == "revolute"  # hinge -> revolute
    assert art[0]["region_a"] == 0 and art[0]["region_b"] == 1
    assert "axis" not in art[0]  # no axis invented


def test_longest_axis_m_grounds_package_scale_without_l6(tmp_path: Path) -> None:
    """Even without L6, a longest_axis_m estimate grounds the package scale."""
    cloud = _cloud(300)
    write_layer_stack(
        cloud,
        tmp_path,
        task_id="scale-only",
        modality="image",
        prompt="",
        seed=11,
        report_dict=_report_dict(cloud.count),
        package_report=build_package_quality_report(
            modality="image", origin_note="generative"
        ),
        solidify_l5=False,
        longest_axis_m=0.3,
    )
    pkg = AstelPackage.read(tmp_path / "package.astel")
    assert pkg.manifest.scale.meters_per_unit != 1.0


def test_no_longest_axis_m_leaves_scale_ungrounded(tmp_path: Path) -> None:
    """Without an estimate the package scale stays the ungrounded identity."""
    cloud = _cloud(300)
    write_layer_stack(
        cloud,
        tmp_path,
        task_id="scale-ungrounded",
        modality="image",
        prompt="",
        seed=12,
        report_dict=_report_dict(cloud.count),
        package_report=build_package_quality_report(
            modality="image", origin_note="generative"
        ),
        solidify_l5=False,
    )
    pkg = AstelPackage.read(tmp_path / "package.astel")
    assert pkg.manifest.scale.meters_per_unit == 1.0
