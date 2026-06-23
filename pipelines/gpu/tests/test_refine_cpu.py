"""Tests for the densified refine loop.

The pure pieces (gradient loss, optimizer construction) run on CPU. The full
gsplat loop runs only where a CUDA gsplat kernel can compile (Box A) via the
``requires_gsplat_runtime`` fixture.
"""

from __future__ import annotations

import torch

from astel_gpu.gaussians import GaussianParams
from astel_gpu.l3_refine import alpha_mask_loss, foreground_mask_from_targets
from astel_gpu.refine import build_optimizer, gradient_loss


def _img(v: int = 2, s: int = 8) -> torch.Tensor:
    return torch.rand(v, s, s, 3)


def test_gradient_loss_zero_for_identical_images() -> None:
    img = _img()
    assert float(gradient_loss(img, img)) == 0.0


def test_gradient_loss_positive_for_edge_mismatch() -> None:
    target = torch.zeros(1, 8, 8, 3)
    target[:, :, 4:, :] = 1.0  # a hard vertical edge
    flat = torch.full((1, 8, 8, 3), 0.5)  # no edge
    assert float(gradient_loss(flat, target)) > 0.0


def test_foreground_mask_from_targets_picks_object() -> None:
    targets = torch.zeros(2, 8, 8, 3)
    targets[:, 2:6, 2:6, :] = 0.7  # a bright object on black
    mask = foreground_mask_from_targets(targets)
    assert mask.shape == (2, 8, 8, 1)
    assert float(mask[:, 2:6, 2:6].min()) == 1.0  # object is foreground
    assert float(mask[:, 0, 0].max()) == 0.0  # corner background is not


def test_alpha_mask_loss_zero_when_alpha_matches_mask() -> None:
    mask = torch.zeros(1, 8, 8, 1)
    mask[:, 2:6, 2:6] = 1.0
    assert float(alpha_mask_loss(mask, mask)) == 0.0


def test_alpha_mask_loss_penalises_faded_object() -> None:
    mask = torch.zeros(1, 8, 8, 1)
    mask[:, 2:6, 2:6] = 1.0
    faded = mask * 0.3  # the object went semi-transparent (the collapse failure)
    assert float(alpha_mask_loss(faded, mask)) > 0.0


def test_build_optimizer_has_3dgs_param_groups() -> None:
    n = 16
    params = GaussianParams(
        means=torch.rand(n, 3, requires_grad=True),
        scales=torch.rand(n, 3, requires_grad=True),
        quats=torch.rand(n, 4, requires_grad=True),
        opacities=torch.rand(n, requires_grad=True),
        colors=torch.rand(n, 3, requires_grad=True),
    )
    opt = build_optimizer(params, lr=5e-3, spatial_lr_scale=2.0, means_lr_scale=0.5)
    groups = opt.param_groups
    assert len(groups) == 3
    assert groups[0]["lr"] == 5e-3 * 2.0 * 0.5  # means
    assert groups[1]["lr"] == 5e-3 * 2.0  # scales
    assert groups[2]["lr"] == 5e-3  # quats/opacity/colors


def test_refine_improves_psnr_and_adapts_count(
    requires_gsplat_runtime: None,
) -> None:
    # GPU-only: render a target cloud, refine a perturbed copy toward it with ADC.
    from astel_gpu.cameras import build_camera_rig
    from astel_gpu.gaussians import build_random_init_cloud, build_target_cloud
    from astel_gpu.l3_refine import render_2dgs_colors
    from astel_gpu.refine import refine_with_densification
    from astel_gpu.smoke_refit import RenderInputs

    device = torch.device("cuda")
    target = build_target_cloud(4000, seed=0, device=device)
    viewmats, ks = build_camera_rig(8, 128)
    inputs = RenderInputs(viewmats=viewmats.to(device), ks=ks.to(device),
                          image_size=128)
    with torch.no_grad():
        targets = render_2dgs_colors(target, inputs)

    init = build_random_init_cloud(4000, seed=1, device=device)
    final, metrics = refine_with_densification(
        init, targets, inputs, iters=200, warmup=50, interval=50, stop=200,
        generator=torch.Generator(device="cuda").manual_seed(0),
    )
    assert metrics["final_psnr_db"] > metrics["init_psnr_db"]
    assert metrics["densify_steps"] >= 1
    assert final.count > 0
