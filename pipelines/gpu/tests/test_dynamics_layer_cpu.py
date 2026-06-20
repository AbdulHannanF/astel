"""CPU tests for L7 dynamics layer wiring (no torch/gsplat/CUDA).

Tests:
1. ``write_dynamics_layer`` writes l7-deformation.bin and l7-timeline.json.
2. ``build_minimal_package`` (via ``write_layer_stack``) binds L7 into the
   ``.astel`` manifest when deformation + timeline paths are supplied.
3. The package round-trips: ``manifest.layers.l7.kind == "dynamics"``.

All operations are numpy-only and CPU-pure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import astel_dynamics
import numpy as np
from astel_format.package import AstelPackage
from astel_splat_io.cloud import SplatCloud

from astel_gpu.packaging import (
    build_package_quality_report,
    write_dynamics_layer,
    write_layer_stack,
)


def _make_cloud(n: int, seed: int = 42) -> SplatCloud:
    """Synthetic SplatCloud for testing."""
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


def _make_dynamics(
    base_positions: np.ndarray,
    n_frames: int = 4,
    n_nodes: int = 5,
) -> tuple[astel_dynamics.DeformationField, astel_dynamics.Timeline]:
    """Fit a simple deformation field: small random per-frame translations."""
    rng = np.random.default_rng(7)
    n = base_positions.shape[0]
    # Per-frame positions: base + tiny Gaussian noise (LBS should fit this well).
    frames = base_positions[np.newaxis, :, :] + rng.standard_normal(
        (n_frames, n, 3)
    ).astype(np.float64) * 0.01

    field, _report = astel_dynamics.fit_deformation_field(
        base_positions, frames, n_nodes=n_nodes
    )
    fps = 24.0
    timeline = astel_dynamics.Timeline(
        fps=fps,
        frame_count=n_frames,
        duration_s=n_frames / fps,
        loop=True,
    )
    return field, timeline


# ---------------------------------------------------------------------------
# 1. write_dynamics_layer writes both files
# ---------------------------------------------------------------------------


def test_write_dynamics_layer_creates_files(tmp_path: Path) -> None:
    """write_dynamics_layer emits l7-deformation.bin and l7-timeline.json."""
    cloud = _make_cloud(50)
    field, timeline = _make_dynamics(cloud.positions.astype(np.float64))

    def_path, tl_path = write_dynamics_layer(field, timeline, tmp_path)

    assert def_path == tmp_path / "l7-deformation.bin"
    assert tl_path == tmp_path / "l7-timeline.json"

    assert def_path.exists() and def_path.stat().st_size > 0
    assert tl_path.exists() and tl_path.stat().st_size > 0


def test_write_dynamics_layer_round_trip(tmp_path: Path) -> None:
    """Binary files written by write_dynamics_layer round-trip via astel_dynamics."""
    cloud = _make_cloud(30)
    field, timeline = _make_dynamics(cloud.positions.astype(np.float64), n_frames=3)

    def_path, tl_path = write_dynamics_layer(field, timeline, tmp_path)

    # Round-trip the deformation field.
    field2 = astel_dynamics.read_deformation_bin(def_path)
    assert field2.n_gaussians == field.n_gaussians
    assert field2.n_nodes == field.n_nodes
    assert field2.n_frames == field.n_frames
    np.testing.assert_allclose(field2.node_positions, field.node_positions, atol=1e-5)

    # Round-trip the timeline.
    timeline2 = astel_dynamics.read_timeline_json(tl_path)
    assert timeline2.fps == timeline.fps
    assert timeline2.frame_count == timeline.frame_count
    assert timeline2.loop == timeline.loop


# ---------------------------------------------------------------------------
# 2. build_minimal_package binds L7 when paths are supplied
# ---------------------------------------------------------------------------


def test_write_layer_stack_binds_l7(tmp_path: Path) -> None:
    """write_layer_stack with l7_deformation_path + l7_timeline_path emits an
    L7 layer in the .astel manifest with kind == 'dynamics'."""
    # Prepare a minimal L3 cloud and dynamics layer files in a sub-dir.
    dyn_dir = tmp_path / "dyn"
    dyn_dir.mkdir()
    cloud = _make_cloud(80)
    field, timeline = _make_dynamics(cloud.positions.astype(np.float64), n_frames=4)
    def_path, tl_path = write_dynamics_layer(field, timeline, dyn_dir)

    package_report = build_package_quality_report(
        modality="video", origin_note="unit-test dynamics binding"
    )
    report = {
        "schema": "astel.quality-report/v0",
        "origin": "measured",
        "splats": cloud.count,
        "fidelity": {"psnr_db": 20.0},
    }

    out_dir = tmp_path / "out"
    write_layer_stack(
        cloud,
        out_dir,
        task_id="task-l7",
        modality="video",
        prompt="",
        seed=0,
        report_dict=report,
        package_report=package_report,
        solidify_l5=False,
        appearance_l4=False,
        l7_deformation_path=def_path,
        l7_timeline_path=tl_path,
        l7_representation="deformation_field",
    )

    pkg = AstelPackage.read(out_dir / "package.astel")
    l7_layer = pkg.manifest.layers.l7
    assert l7_layer is not None, "L7 layer missing from manifest"
    assert l7_layer.kind == "dynamics"
    assert l7_layer.dynamics is not None
    assert l7_layer.dynamics.representation == "deformation_field"


def test_write_layer_stack_no_l7_when_paths_absent(tmp_path: Path) -> None:
    """write_layer_stack without L7 paths leaves manifest.layers.l7 == None
    (static image/text path is byte-identical to pre-wiring behaviour)."""
    cloud = _make_cloud(50)
    package_report = build_package_quality_report(
        modality="text", origin_note="static path — no L7"
    )
    report = {
        "schema": "astel.quality-report/v0",
        "origin": "generated",
        "splats": cloud.count,
        "fidelity": {"psnr_db": 22.0},
    }

    write_layer_stack(
        cloud,
        tmp_path,
        task_id="task-no-l7",
        modality="text",
        prompt="a cup",
        seed=1,
        report_dict=report,
        package_report=package_report,
        solidify_l5=False,
        appearance_l4=False,
        # No l7_deformation_path / l7_timeline_path supplied.
    )

    pkg = AstelPackage.read(tmp_path / "package.astel")
    assert pkg.manifest.layers.l7 is None


# ---------------------------------------------------------------------------
# 3. Default representation parameter
# ---------------------------------------------------------------------------


def test_write_dynamics_layer_default_representation(tmp_path: Path) -> None:
    """Default representation kwarg is 'deformation_field'."""
    cloud = _make_cloud(20)
    field, timeline = _make_dynamics(cloud.positions.astype(np.float64), n_frames=2)

    dyn_dir = tmp_path / "dyn"
    dyn_dir.mkdir()
    def_path, tl_path = write_dynamics_layer(field, timeline, dyn_dir)

    out_dir = tmp_path / "out"
    package_report = build_package_quality_report(
        modality="video", origin_note="default-repr test"
    )
    report: dict[str, Any] = {
        "schema": "astel.quality-report/v0",
        "origin": "generated",
        "splats": cloud.count,
    }

    write_layer_stack(
        cloud,
        out_dir,
        task_id="task-repr",
        modality="video",
        prompt="",
        seed=2,
        report_dict=report,
        package_report=package_report,
        solidify_l5=False,
        appearance_l4=False,
        l7_deformation_path=def_path,
        l7_timeline_path=tl_path,
        # l7_representation=None → builder defaults to "deformation_field"
    )

    pkg = AstelPackage.read(out_dir / "package.astel")
    l7 = pkg.manifest.layers.l7
    assert l7 is not None
    assert l7.dynamics is not None
    assert l7.dynamics.representation == "deformation_field"
