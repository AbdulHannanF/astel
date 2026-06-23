"""Reconstruct a 3D gaussian splat from MV-Adapter's view-consistent images.

Everything lives in gsplat's OpenCV convention at the MV-Adapter viewpoints
(:func:`ortho_cameras` builds them from a
:class:`astel_gpu.text_to_multiview.MultiViewSpec`), so NO TripoSplat frame alignment
is needed. rembg mattes each view onto black; a sphere of gaussians is fit to the
views with photometric + alpha-silhouette + adaptive density control (the same
densified engine as :mod:`astel_gpu.refine`). 2DGS does not support orthographic
cameras in gsplat, so this uses 3DGS ``rasterization(camera_model="ortho")``.

Pure / CPU-testable seams: :func:`ortho_cameras`, :func:`sphere_init`,
:func:`reconstruction_loss`. The fit loop (:func:`reconstruct`) and matting
(:func:`matte_views`) need the GPU / rembg and are exercised through ``run-python.cmd``.
"""

from __future__ import annotations

import math
import time
from typing import Any

import torch

from .cameras import look_at_viewmats
from .densify import DensifyConfig, DensityController, reset_opacity
from .gaussians import GaussianParams
from .l3_refine import alpha_mask_loss
from .refine import build_optimizer, gradient_loss
from .smoke_refit import d_ssim_loss, make_trainable


def ortho_cameras(
    azimuth_deg: tuple[int, ...] | list[int],
    elevation_deg: float,
    resolution: int,
    *,
    frustum: float = 1.1,
    distance: float = 3.0,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build ``(viewmats, Ks)`` orthographic cameras at the MV-Adapter viewpoints.

    Convention (gsplat OpenCV, +Z up): azimuth 0 is viewed from -Y, azimuth
    increases toward +X; ``Ks`` encode the ortho scale ``fx = resolution / frustum``.
    Pure: returns CPU/GPU tensors, no gsplat. The ``distance`` only sets depth
    ordering for an orthographic camera (it does not change the projected size).
    """
    az = torch.tensor([math.radians(a) for a in azimuth_deg], dtype=torch.float32)
    el = math.radians(elevation_deg)
    ce, se = math.cos(el), math.sin(el)
    centres = torch.stack(
        [torch.sin(az) * ce, -torch.cos(az) * ce, torch.full_like(az, se)], dim=1
    ) * distance
    viewmats = look_at_viewmats(
        centres, torch.zeros(3), torch.tensor([0.0, 0.0, 1.0])
    )
    fx = resolution / frustum
    half = resolution / 2.0
    k = torch.tensor([[fx, 0, half], [0, fx, half], [0, 0, 1.0]], dtype=torch.float32)
    ks = k[None].repeat(len(az), 1, 1)
    return viewmats.to(device), ks.to(device)


def sphere_init(
    n: int,
    *,
    radius: float = 0.5,
    scale: float = 0.018,
    opacity: float = 0.1,
    color: float = 0.5,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> GaussianParams:
    """A solid sphere of ``n`` gaussians — the from-scratch reconstruction start.

    Alpha-silhouette supervision carves it to the visual hull and ADC grows detail.
    Pure / CPU-runnable.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    d = torch.randn(n, 3, generator=g)
    d = d / d.norm(dim=-1, keepdim=True)
    r = radius * torch.rand(n, generator=g)[:, None] ** (1.0 / 3.0)
    means = (d * r).to(device)
    return GaussianParams(
        means=means,
        scales=torch.full((n, 3), scale, device=device),
        quats=torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(n, 1),
        opacities=torch.full((n,), opacity, device=device),
        colors=torch.full((n, 3), color, device=device),
    )


def reconstruction_loss(
    rgb: torch.Tensor,
    alpha: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor,
    *,
    w_l1: float = 0.8,
    w_ssim: float = 0.2,
    w_alpha: float = 0.6,
    w_perceptual: float = 0.2,
) -> torch.Tensor:
    """Photometric (L1 + D-SSIM) + alpha-silhouette + edge-perceptual loss.

    The alpha term pins the rendered silhouette to the matte so the sphere carves to
    the object; the perceptual (gradient) term keeps edges/engraving crisp. Pure.
    """
    return (
        w_l1 * (rgb - targets).abs().mean()
        + w_ssim * d_ssim_loss(rgb, targets)
        + w_alpha * alpha_mask_loss(alpha, masks)
        + w_perceptual * gradient_loss(rgb, targets)
    )


def matte_views(
    images: list[Any],
    *,
    model: str = "isnet-general-use",
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """rembg-matte PIL views onto black -> ``(targets, masks)`` torch tensors.

    ``targets`` is ``(V, H, W, 3)`` composited on black; ``masks`` is ``(V, H, W, 1)``
    the object alpha. rembg is imported lazily (heavy, GPU box only).
    """
    import numpy as np  # noqa: PLC0415
    from rembg import (  # type: ignore[import-untyped]  # noqa: PLC0415,E501
        new_session,
        remove,
    )

    session = new_session(model)
    tgt: list[Any] = []
    msk: list[Any] = []
    for im in images:
        rgba = np.asarray(remove(im.convert("RGB"), session=session), dtype=np.float32)
        rgba = rgba / 255.0
        alpha = rgba[..., 3:4]
        tgt.append(rgba[..., :3] * alpha)
        msk.append(alpha)
    targets = torch.from_numpy(np.stack(tgt)).to(device)
    masks = torch.from_numpy(np.stack(msk)).to(device)
    return targets, masks


def _fclamp(p: GaussianParams) -> GaussianParams:
    return GaussianParams(
        p.means, p.scales.abs().clamp_min(1e-4), p.quats,
        p.opacities.clamp(1e-4, 1.0), p.colors.clamp(0.0, 1.0),
    )


def _detach(p: GaussianParams) -> GaussianParams:
    return GaussianParams(
        p.means.detach().clone(), p.scales.detach().clone(), p.quats.detach().clone(),
        p.opacities.detach().clone(), p.colors.detach().clone(),
    )


def _subset(p: GaussianParams, m: torch.Tensor) -> GaussianParams:
    return GaussianParams(
        p.means[m], p.scales[m], p.quats[m], p.opacities[m], p.colors[m]
    )


def render_ortho(
    p: GaussianParams, viewmats: torch.Tensor, ks: torch.Tensor, resolution: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """3DGS orthographic render -> ``(rgb, alpha)``. gsplat (GPU)."""
    import gsplat  # noqa: PLC0415

    colors, alphas, _meta = gsplat.rasterization(
        means=p.means, quats=p.quats, scales=p.scales, opacities=p.opacities,
        colors=p.colors, viewmats=viewmats, Ks=ks, width=resolution,
        height=resolution, camera_model="ortho", render_mode="RGB", packed=False,
    )
    return colors[..., :3].clamp(0.0, 1.0), alphas


def reconstruct(
    targets: torch.Tensor,
    masks: torch.Tensor,
    viewmats: torch.Tensor,
    ks: torch.Tensor,
    resolution: int,
    *,
    init: GaussianParams | None = None,
    init_count: int = 100_000,
    iters: int = 2200,
    lr: float = 5.0e-3,
    densify_config: DensifyConfig | None = None,
    warmup: int = 200,
    interval: int = 100,
    stop: int = 1800,
    opacity_reset_interval: int = 700,
    prune_opacity: float = 0.02,
    seed: int = 0,
) -> tuple[GaussianParams, dict[str, Any]]:
    """Fit a gaussian cloud to matted multi-view targets (GPU, densified).

    Returns ``(cloud, metrics)``. A sphere init is carved by the alpha-silhouette
    supervision and grown by ADC; a gentle final opacity prune drops only
    near-transparent gaussians (keeping thin engraving detail, unlike a
    connected-components clean which over-prunes a from-scratch fit).
    """
    device = targets.device
    cfg = densify_config or DensifyConfig(
        grad_threshold=5.0e-5, percent_dense=0.01, max_gaussians=1_200_000
    )
    init = init if init is not None else sphere_init(init_count, device=device)
    params = make_trainable(init)
    opt = build_optimizer(params, lr=lr, spatial_lr_scale=1.0, means_lr_scale=1.0)
    ctrl = DensityController(
        params.count, cfg, device, warmup=warmup, interval=interval, stop=stop,
        opacity_reset_interval=opacity_reset_interval,
        generator=torch.Generator(device=device).manual_seed(seed),
    )

    start = time.perf_counter()
    history: list[dict[str, Any]] = []
    for it in range(iters):
        opt.zero_grad(set_to_none=True)
        fp = _fclamp(params)
        rgb, alpha = render_ortho(fp, viewmats, ks, resolution)
        loss = reconstruction_loss(rgb, alpha, targets, masks)
        loss.backward()  # type: ignore[no-untyped-call]
        ctrl.record(params.means.grad)
        opt.step()
        rebuilt = False
        if ctrl.should_reset_opacity(it):
            params = make_trainable(reset_opacity(_detach(params)))
            rebuilt = True
        if ctrl.should_densify(it):
            new_params, stats = ctrl.step(_detach(params))
            stats["iter"] = it
            history.append(stats)
            params = make_trainable(new_params)
            rebuilt = True
        if rebuilt:
            opt = build_optimizer(
                params, lr=lr, spatial_lr_scale=1.0, means_lr_scale=1.0
            )

    if device.type == "cuda":
        torch.cuda.synchronize()
    final = _fclamp(_detach(params))
    final = _subset(final, final.opacities > prune_opacity)
    metrics: dict[str, Any] = {
        "stage": "mv-reconstruct",
        "splats_out": final.count,
        "iters": iters,
        "densify_steps": len(history),
        "wall_time_s": time.perf_counter() - start,
    }
    return final, metrics
