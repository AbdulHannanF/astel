"""Real L3 refinement — densified, position-free, multi-view-target optimisation.

This is the Tier-1 replacement for the generative L3 *distillation*
(:func:`astel_gpu.generative.run_l2_to_l3`, which freezes positions and re-fits a
fixed-count surfel cloud to the L2 generator's OWN renders). The three differences
that let this one actually improve an asset:

1. **External multi-view targets.** ``targets`` are supervision images. Pass the L2
   self-renders (a better-distillation, still bounded by L2) OR — the real win —
   multi-view-consistent images from a stronger generator (TRELLIS.2 / MVDream /
   SDS-enhanced renders). The optimiser can then exceed L2 toward those targets.
2. **Adaptive Density Control** (:mod:`astel_gpu.densify`) — clone/split/prune/
   opacity-reset, so the cloud grows detail where the gradient demands it and
   sheds floaters, instead of being stuck at TripoSplat's fixed count.
3. **Unfrozen positions** — gaussians move to fit the targets; ADC's pruning +
   opacity reset is the principled cure for the floaters that made the distillation
   freeze positions in the first place.

The per-pixel loss adds an optional **perceptual** term on top of ``L1 + D-SSIM``;
the default :func:`gradient_loss` is a dependency-free edge/sharpness proxy, and a
true LPIPS module plugs into the same ``perceptual`` callable.

RUNTIME: :func:`refine_with_densification` renders via gsplat (GPU). Its pure
pieces — :func:`gradient_loss`, :func:`build_optimizer` — are unit-tested on CPU;
the full loop is exercised by a gsplat-guarded GPU test (Box A). Existing callers
are untouched: this is opt-in.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import torch

from .densify import DensifyConfig, DensityController, reset_opacity
from .export import psnr
from .gaussians import GaussianParams
from .l3_refine import (
    DEFAULT_LAMBDA_NORMAL,
    alpha_mask_loss,
    foreground_mask_from_targets,
    render_2dgs_colors,
    render_2dgs_train_alpha,
)
from .smoke_refit import RenderInputs, d_ssim_loss, make_trainable

#: A perceptual loss term: ``(pred, target) -> scalar``. Both are ``(V,H,W,3)`` in
#: ``[0, 1]``. Defaults to :func:`gradient_loss`; swap in LPIPS here.
PerceptualLoss = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _image_gradients(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward-difference image gradients of a ``(V, H, W, 3)`` tensor."""
    dx = t[:, :, 1:, :] - t[:, :, :-1, :]
    dy = t[:, 1:, :, :] - t[:, :-1, :, :]
    return dx, dy


def gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Edge/sharpness proxy: L1 between the image gradients of pred and target.

    Pure and CPU-testable. Penalising gradient mismatch (not just per-pixel colour)
    pushes the optimiser to reproduce crisp edges/texture instead of the blurry
    minimum-L1 average — a cheap stand-in for a full perceptual loss, and a genuine
    sharpness gain over ``L1 + D-SSIM`` alone.
    """
    pdx, pdy = _image_gradients(pred)
    tdx, tdy = _image_gradients(target)
    return (pdx - tdx).abs().mean() + (pdy - tdy).abs().mean()


def build_optimizer(
    params: GaussianParams,
    *,
    lr: float = 5.0e-3,
    spatial_lr_scale: float = 1.0,
    means_lr_scale: float = 1.0,
) -> torch.optim.Optimizer:
    """Adam over the trainable params, mirroring the 3DGS per-group LR scheme.

    ``means``/``scales`` carry the spatial LR scale; ``means_lr_scale`` further
    scales positions only (the refiner uses ``1.0`` — positions MUST move, unlike
    the frozen distillation). Pure: no gsplat, CPU-constructible.
    """
    return torch.optim.Adam(
        [
            {"params": [params.means], "lr": lr * spatial_lr_scale * means_lr_scale},
            {"params": [params.scales], "lr": lr * spatial_lr_scale},
            {"params": [params.quats, params.opacities, params.colors], "lr": lr},
        ]
    )


def _forward_clamp(params: GaussianParams) -> GaussianParams:
    """Soft-clamp scales/opacity/colour for the forward pass (params stay free)."""
    return GaussianParams(
        means=params.means,
        scales=params.scales.abs().clamp_min(1e-4),
        quats=params.quats,
        opacities=params.opacities.clamp(1e-4, 1.0),
        colors=params.colors.clamp(0.0, 1.0),
    )


def _detach(params: GaussianParams) -> GaussianParams:
    """A detached, contiguous clone — the handoff between optimiser rebuilds."""
    return GaussianParams(
        means=params.means.detach().clone(),
        scales=params.scales.detach().clone(),
        quats=params.quats.detach().clone(),
        opacities=params.opacities.detach().clone(),
        colors=params.colors.detach().clone(),
    )


def refine_with_densification(
    init: GaussianParams,
    targets: torch.Tensor,
    inputs: RenderInputs,
    *,
    iters: int,
    lr: float = 5.0e-3,
    spatial_lr_scale: float = 1.0,
    lambda_normal: float = DEFAULT_LAMBDA_NORMAL,
    lambda_dist: float = 0.0,
    lambda_perceptual: float = 0.1,
    lambda_alpha: float = 0.5,
    perceptual: PerceptualLoss | None = None,
    densify_config: DensifyConfig | None = None,
    warmup: int = 100,
    interval: int = 100,
    stop: int | None = None,
    opacity_reset_interval: int = 900,
    generator: torch.Generator | None = None,
) -> tuple[GaussianParams, dict[str, Any]]:
    """Refine ``init`` as 2DGS surfels toward ``targets`` with ADC + perceptual loss.

    Returns ``(final_params, metrics)``. ``metrics`` records the init/final PSNR,
    the densification history (per-step clone/split/prune counts and the final
    count), and timing. The cloud size changes across the run, so the Adam
    optimiser is rebuilt after each densify/opacity-reset (lost momentum is
    immaterial over a short refine).
    """
    perceptual = perceptual or gradient_loss
    use_distloss = lambda_dist > 0.0
    stop = stop if stop is not None else iters
    device = init.means.device
    # The object silhouette the rendered alpha is pinned to — prevents the unfrozen
    # optimiser from fading the object to transparent on a black background (the
    # dark-collapse that made the densified refine lose to the frozen distillation).
    target_mask = foreground_mask_from_targets(targets) if lambda_alpha > 0.0 else None

    params = make_trainable(init)
    optimizer = build_optimizer(
        params, lr=lr, spatial_lr_scale=spatial_lr_scale, means_lr_scale=1.0
    )
    ctrl = DensityController(
        params.count,
        densify_config,
        device,
        warmup=warmup,
        interval=interval,
        stop=stop,
        opacity_reset_interval=opacity_reset_interval,
        generator=generator,
    )

    with torch.no_grad():
        init_psnr_db = psnr(render_2dgs_colors(_forward_clamp(params), inputs), targets)

    densify_history: list[dict[str, Any]] = []
    start = time.perf_counter()
    for it in range(iters):
        optimizer.zero_grad(set_to_none=True)
        forward_params = _forward_clamp(params)
        rendered, alphas, normals, surf_normals, distort = render_2dgs_train_alpha(
            forward_params, inputs, distloss=use_distloss
        )
        l1 = torch.abs(rendered - targets).mean()
        ssim_term = d_ssim_loss(rendered, targets)
        normal_err = (1.0 - (normals * surf_normals).sum(dim=-1)).mean()
        reg = lambda_normal * normal_err + lambda_dist * distort.mean()
        perceptual_term = lambda_perceptual * perceptual(rendered, targets)
        alpha_term = (
            lambda_alpha * alpha_mask_loss(alphas, target_mask)
            if target_mask is not None
            else rendered.new_zeros(())
        )
        loss = 0.8 * l1 + 0.2 * ssim_term + reg + perceptual_term + alpha_term
        loss.backward()  # type: ignore[no-untyped-call]
        ctrl.record(params.means.grad)
        optimizer.step()

        rebuilt = False
        if ctrl.should_reset_opacity(it):
            params = make_trainable(reset_opacity(_detach(params)))
            rebuilt = True
        if ctrl.should_densify(it):
            new_params, dstats = ctrl.step(_detach(params))
            dstats["iter"] = it
            densify_history.append(dstats)
            params = make_trainable(new_params)
            rebuilt = True
        if rebuilt:
            optimizer = build_optimizer(
                params, lr=lr, spatial_lr_scale=spatial_lr_scale, means_lr_scale=1.0
            )

    if device.type == "cuda":
        torch.cuda.synchronize()
    wall_time_s = time.perf_counter() - start

    final_params = _forward_clamp(_detach(params))
    with torch.no_grad():
        final_psnr_db = psnr(render_2dgs_colors(final_params, inputs), targets)

    metrics: dict[str, Any] = {
        "stage": "l3-refine-densified",
        "init_psnr_db": init_psnr_db,
        "final_psnr_db": final_psnr_db,
        "iters": iters,
        "lambda_normal": lambda_normal,
        "lambda_dist": lambda_dist,
        "lambda_perceptual": lambda_perceptual,
        "lambda_alpha": lambda_alpha,
        "splats_in": init.count,
        "splats_out": final_params.count,
        "densify_steps": len(densify_history),
        "densify_history": densify_history,
        "wall_time_s": wall_time_s,
    }
    return final_params, metrics
