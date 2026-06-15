"""CPU tests for generative L2->L3 pure seams (no gsplat/CUDA/weights)."""

from __future__ import annotations

import math

import torch

from astel_gpu.gaussians import GaussianParams
from astel_gpu.generative import build_generative_quality_report, normalize_params


def _params(means: torch.Tensor) -> GaussianParams:
    n = means.shape[0]
    return GaussianParams(
        means=means,
        scales=torch.full((n, 3), 0.5),
        quats=torch.tile(torch.tensor([1.0, 0.0, 0.0, 0.0]), (n, 1)),
        opacities=torch.full((n,), 0.7),
        colors=torch.rand(n, 3),
    )


def test_normalize_centers_and_unit_radius() -> None:
    means = torch.tensor(
        [[10.0, 10.0, 10.0], [10.0, 10.0, 14.0], [10.0, 14.0, 10.0]]
    )
    p = _params(means)
    out, center, radius = normalize_params(p)

    # Centroid maps to the origin.
    assert torch.allclose(out.means.mean(dim=0), torch.zeros(3), atol=1e-5)
    # Max distance from origin is exactly 1 after scaling.
    assert math.isclose(float(out.means.norm(dim=-1).max()), 1.0, rel_tol=1e-5)
    # Scales shrink by the same radius factor; invariants untouched.
    assert torch.allclose(out.scales, p.scales / radius)
    assert torch.allclose(out.opacities, p.opacities)
    assert torch.allclose(out.quats, p.quats)
    assert torch.allclose(center, means.mean(dim=0))


def test_generative_report_is_honest_about_generation() -> None:
    report = build_generative_quality_report(
        count=50000, l2_count=65536, psnr_db=28.0, n_holdout_views=4,
        image_path="x.webp",
    )
    # Generated => no measured geometry or metric scale, fully generated provenance.
    assert report["geometric_error"]["chamfer_mm_vs_l1"] is None
    assert report["scale"]["longest_axis_m"] is None
    assert report["provenance"]["generated_ratio"] == 1.0
    assert report["provenance"]["measured_ratio"] == 0.0
    assert report["representation"] == "2dgs"
    assert report["fidelity"]["psnr_db"] == 28.0
