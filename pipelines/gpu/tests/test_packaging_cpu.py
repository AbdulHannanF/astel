"""CPU tests for the full layer-stack writer seam (no torch/gsplat/CUDA).

``write_layer_stack`` operates on a numpy ``SplatCloud`` + a pydantic
``QualityReport``, so the GPU producer's artifact contract is verified entirely
on CPU: all expected files are emitted, the ``.astel`` package round-trips, and
the honesty fields are intact.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from astel_format.package import AstelPackage
from astel_splat_io.cloud import SplatCloud

from astel_gpu.packaging import (
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
        "l3.spz",
        "l3.sog",
        "package.astel",
        "quality-report.json",
    }
    assert set(names) == expected
    for name in expected:
        assert (tmp_path / name).stat().st_size > 0


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
