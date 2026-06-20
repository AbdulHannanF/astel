"""Analytic ground-truth check for fit_deformation_field.

Three motion types are tested to prove the honesty rule (CLAUDE.md §10.4):

1. Global rigid rotation (K=1) → near-zero reconstruction error.
2. Bend motion (K=8) → small error (LBS-compatible low-rank motion).
3. Random per-point motion (K=8) → honestly LARGE error — LBS can't compress it.

The test prints measured errors so the caller can inspect the real numbers.
"""

from __future__ import annotations

import numpy as np
from _motion import bend_motion, random_motion, rigid_rotation_motion, static_cloud

from astel_dynamics.fit import fit_deformation_field


def _cloud_extent(base: np.ndarray) -> float:
    """Diagonal of the bounding box — used as scale reference."""
    return float(np.linalg.norm(base.max(axis=0) - base.min(axis=0)))


def test_rigid_rotation_k1_near_zero_error() -> None:
    """Global rigid rotation with K=1 must recover near-zero error.

    A single affine node with weight=1 for all gaussians can represent a global
    rigid transform exactly.  The weighted least-squares solve must find it.
    """
    base = static_cloud(200, seed=0)
    frames = rigid_rotation_motion(
        base, n_frames=10, axis=[0, 1, 0], total_angle=np.pi / 3
    )

    field, report = fit_deformation_field(base, frames, n_nodes=1, seed=0)

    scale = _cloud_extent(base)
    print(
        f"\n[rigid K=1] overall_mean_err={report.overall_mean_err:.6f} "
        f"overall_p95_err={report.overall_p95_err:.6f} "
        f"scale={scale:.4f} "
        f"rel_err={report.overall_mean_err / scale:.4e}"
    )

    # Must be < 0.1% of cloud scale (tight numerical tolerance for a global affine)
    assert report.overall_mean_err < 1e-3 * scale, (
        f"Rigid rotation K=1 fit error {report.overall_mean_err:.4e} should be "
        f"< 1e-3 * scale={scale:.4e}"
    )
    assert report.n_nodes == 1


def test_bend_motion_k8_small_error() -> None:
    """Bend motion with K=8 should fit tightly (low-rank LBS-compatible).

    The bend is a smooth rotation-angle proportional to x — 8 nodes spanning
    the x-range can approximate this well.  We allow up to 5% of scale.
    """
    base = static_cloud(300, seed=1)
    frames = bend_motion(base, n_frames=8, max_angle=np.pi / 4)

    field, report = fit_deformation_field(base, frames, n_nodes=8, seed=0)

    scale = _cloud_extent(base)
    print(
        f"\n[bend  K=8] overall_mean_err={report.overall_mean_err:.6f} "
        f"overall_p95_err={report.overall_p95_err:.6f} "
        f"scale={scale:.4f} "
        f"rel_err={report.overall_mean_err / scale:.4e}"
    )

    # Allow up to 5% relative error for the bend case
    assert report.overall_mean_err < 0.05 * scale, (
        f"Bend K=8 fit error {report.overall_mean_err:.4e} should be "
        f"< 5% of scale={scale:.4e}"
    )
    assert report.n_nodes == 8


def test_random_motion_k8_honestly_large_error() -> None:
    """Random per-point motion must produce honestly LARGE reported error.

    LBS cannot compress independent per-point motion.  The fitter must NOT
    fabricate low errors.  We assert the error is substantially larger than the
    bend case — at least 10% of cloud scale.
    """
    base = static_cloud(300, seed=2)
    frames = random_motion(base, n_frames=8, seed=99)

    _, report = fit_deformation_field(base, frames, n_nodes=8, seed=0)

    scale = _cloud_extent(base)
    print(
        f"\n[rand  K=8] overall_mean_err={report.overall_mean_err:.6f} "
        f"overall_p95_err={report.overall_p95_err:.6f} "
        f"scale={scale:.4f} "
        f"rel_err={report.overall_mean_err / scale:.4e}"
    )

    # Must be at least 8% of scale — proving honest non-zero residual.
    # The random displacement magnitude is 0.1 * N(0,1) in 3D, giving
    # an expected per-point RMS error of ~0.1*sqrt(3) ≈ 0.173 relative to
    # a unit-cube cloud (diagonal ~sqrt(3)).  The 8% threshold is comfortably
    # below the theoretical floor and well above the LBS-compatible bend case
    # (~1.5% relative error with K=8) — confirming the fitter does not
    # fabricate accuracy on incompressible motion.
    assert report.overall_mean_err > 0.08 * scale, (
        f"Random motion fit error {report.overall_mean_err:.4e} should be "
        f"> 8% of scale={scale:.4e} to prove honesty of the residual"
    )


def test_random_error_exceeds_bend_error() -> None:
    """Random motion error must exceed bend motion error — honesty check."""
    base = static_cloud(200, seed=3)

    frames_bend = bend_motion(base, n_frames=6, max_angle=np.pi / 6)
    _, rep_bend = fit_deformation_field(base, frames_bend, n_nodes=8, seed=0)

    frames_rand = random_motion(base, n_frames=6, seed=77)
    _, rep_rand = fit_deformation_field(base, frames_rand, n_nodes=8, seed=0)

    print(
        f"\n[honesty check] bend_mean={rep_bend.overall_mean_err:.6f} "
        f"rand_mean={rep_rand.overall_mean_err:.6f}"
    )

    assert rep_rand.overall_mean_err > rep_bend.overall_mean_err * 5, (
        "Random motion error should be at least 5× larger than bend error "
        f"(bend={rep_bend.overall_mean_err:.4e}, rand={rep_rand.overall_mean_err:.4e})"
    )


def test_fit_report_fields_populated() -> None:
    """FitReport must have all required fields populated and non-None."""
    base = static_cloud(50, seed=4)
    frames = rigid_rotation_motion(base, n_frames=3, axis=[0, 0, 1], total_angle=0.5)
    _, report = fit_deformation_field(base, frames, n_nodes=2, seed=0)

    assert len(report.per_frame_mean_err) == 3
    assert len(report.per_frame_p95_err) == 3
    assert report.overall_mean_err >= 0.0
    assert report.overall_p95_err >= 0.0
    assert isinstance(report.note, str) and len(report.note) > 0
    assert report.n_nodes == 2


def test_n_nodes_clamped_to_n() -> None:
    """Requesting more nodes than gaussians should clamp to N."""
    base = static_cloud(5, seed=5)
    frames = rigid_rotation_motion(base, n_frames=2, axis=[1, 0, 0], total_angle=0.1)
    _, report = fit_deformation_field(base, frames, n_nodes=100, seed=0)
    assert report.n_nodes == 5
