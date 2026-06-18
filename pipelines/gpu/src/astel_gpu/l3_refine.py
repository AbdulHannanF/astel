"""L3 surface-aligned refinement (2DGS) — the A/B counterpart to raw 3DGS.

gsplat ships native 2D Gaussian Splatting (``rasterization_2dgs``): the splats
become surfels with REAL per-splat normals, and the rasterizer also returns a
depth-derived surface-normal map plus an L1 depth-distortion term. On top of the
RGB ``L1 + D-SSIM`` loss we add the two standard 2DGS regularizers — normal
consistency (rendered normals vs. depth normals) and optional depth distortion —
which is what pulls the gaussians onto the surface instead of leaving the
volumetric floaters raw 3DGS produces.

The optimized surfel CENTRES feed the SAME DTU ObsMask/Plane geometry protocol
as the raw-3DGS baseline (:mod:`astel_gpu.capture_eval`), so the A/B is
apples-to-apples: identical init cloud, identical metric, only the representation
+ losses change.

HONESTY — GOF is the documented runner-up, NOT implemented here. DECISIONS #1
framed the L3 choice as "2DGS surfels vs 3DGS + GOF extraction". Gaussian Opacity
Fields needs a custom ray-gaussian opacity rasterizer + tetrahedral isosurface
extraction that gsplat does not provide; 2DGS is gsplat-native and yields surfel
normals directly usable by L4 (BRDF) / L5 (SDF/collision). We therefore measure
2DGS-vs-3DGS first; GOF (or a depth/normal-prior path like PGSR/DN-Splatter) is
revisited only if 2DGS fails to beat the baseline on real DTU geometry.
"""

from __future__ import annotations

import torch

from .export import psnr
from .gaussians import GaussianParams
from .smoke_refit import RenderInputs, d_ssim_loss, make_trainable

#: Standard 2DGS normal-consistency weight (Huang et al. 2024 use 0.05).
DEFAULT_LAMBDA_NORMAL = 0.05
#: Depth-distortion weight. The published value (100/1000) is tuned for
#: unit/NDC scenes; on a metric (mm) DTU scene the distortion magnitude is
#: large, so we default it OFF and let normal consistency carry the surface
#: signal (it is scale-invariant — a dot product of unit normals). Enable via
#: ``lambda_dist`` if a scene needs it.
DEFAULT_LAMBDA_DIST = 0.0


def _colors_per_cam(params: GaussianParams, n_cams: int) -> torch.Tensor:
    """Broadcast per-gaussian RGB to per-camera ``[C, N, 3]``.

    gsplat 1.5.3's ``rasterization_2dgs`` does NOT auto-expand per-gaussian
    colors to ``[C, N, D]`` when ``sh_degree is None`` (the expansion is
    commented out in rendering.py, unlike ``rasterization``), so
    ``rasterize_to_pixels_2dgs`` asserts on a bare ``[N, D]`` tensor. We expand
    ourselves.
    """
    return params.colors[None].expand(n_cams, -1, -1)


def render_2dgs_colors(params: GaussianParams, inputs: RenderInputs) -> torch.Tensor:
    """RGB-only surfel render, ``(V, H, W, 3)`` in ``[0, 1]`` — for eval/PSNR."""
    import gsplat

    n_cams = int(inputs.viewmats.shape[0])
    colors, *_ = gsplat.rasterization_2dgs(
        means=params.means,
        quats=params.quats,
        scales=params.scales,
        opacities=params.opacities,
        colors=_colors_per_cam(params, n_cams),
        viewmats=inputs.viewmats,
        Ks=inputs.ks,
        width=inputs.w,
        height=inputs.h,
        packed=False,
        render_mode="RGB",
    )
    out: torch.Tensor = colors[..., :3].clamp(0.0, 1.0)
    return out


def render_2dgs_train(
    params: GaussianParams, inputs: RenderInputs, *, distloss: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Surfel render WITH surface tensors, for the regularized training loss.

    Uses ``render_mode="RGB+ED"`` so the depth-derived ``surf_normals`` (and,
    with ``distloss=True``, the distortion map) are populated — both require
    depth rendering in gsplat. Returns ``(colors, normals, surf_normals,
    distort)``; ``colors`` is ``(V, H, W, 3)``, the rest match the rasterizer's
    ``(V, H, W, {3,3,1})`` outputs.
    """
    import gsplat

    n_cams = int(inputs.viewmats.shape[0])
    colors, _alphas, normals, surf_normals, distort, _median, _meta = (
        gsplat.rasterization_2dgs(
            means=params.means,
            quats=params.quats,
            scales=params.scales,
            opacities=params.opacities,
            colors=_colors_per_cam(params, n_cams),
            viewmats=inputs.viewmats,
            Ks=inputs.ks,
            width=inputs.w,
            height=inputs.h,
            packed=False,
            render_mode="RGB+ED",
            distloss=distloss,
        )
    )
    return colors[..., :3].clamp(0.0, 1.0), normals, surf_normals, distort


def surface_reg_loss(
    render_normals: torch.Tensor,
    surf_normals: torch.Tensor,
    distort: torch.Tensor,
    *,
    lambda_normal: float,
    lambda_dist: float,
) -> torch.Tensor:
    """2DGS surface regularization — pure, CPU-testable, no gsplat needed.

    ``normal`` term = ``1 - <render_normals, surf_normals>`` averaged over all
    pixels (both inputs are per-pixel 3-vectors; the rasterizer returns them
    pre-normalized). ``dist`` term = mean of the distortion map. Returns a scalar
    tensor ``lambda_normal * normal + lambda_dist * dist``.
    """
    normal_err = (1.0 - (render_normals * surf_normals).sum(dim=-1)).mean()
    dist_err = distort.mean()
    return lambda_normal * normal_err + lambda_dist * dist_err


def optimize_2dgs(
    init: GaussianParams,
    targets: torch.Tensor,
    inputs: RenderInputs,
    iters: int,
    lr: float = 5e-3,
    spatial_lr_scale: float = 1.0,
    lambda_normal: float = DEFAULT_LAMBDA_NORMAL,
    lambda_dist: float = DEFAULT_LAMBDA_DIST,
    means_lr_scale: float = 1.0,
) -> tuple[GaussianParams, float]:
    """Optimize ``init`` as 2DGS surfels to match ``targets``.

    Mirrors :func:`astel_gpu.smoke_refit.optimize` (same Adam param groups,
    same ``spatial_lr_scale`` metric-coords trick, same forward soft-clamps) but
    renders via :func:`render_2dgs` and adds :func:`surface_reg_loss`. Returns
    ``(optimized_params, init_psnr_db)``.

    ``means_lr_scale`` (default ``1.0``) further scales the *position* learning
    rate only. The capture path keeps it at ``1.0`` (positions must move to fit
    real photos). The generative path sets it low/zero: there ``init`` is already
    an excellent high-count generator cloud (TripoSplat L2), so letting positions
    take full 5e-3 steps for 1500 iters makes splats drift off-surface into
    floaters that degrade the asset and inflate its bounding radius (measured:
    262k L3 floaters vs. a clean L2). A small/zero position LR turns this into a
    *surfelization* — scales/opacity/colour/quats adapt to flatten the gaussians
    into surfels while the proven geometry is preserved.
    """
    params = make_trainable(init)
    use_distloss = lambda_dist > 0.0

    with torch.no_grad():
        init_psnr_db = psnr(render_2dgs_colors(params, inputs), targets)

    optimizer = torch.optim.Adam(
        [
            {"params": [params.means], "lr": lr * spatial_lr_scale * means_lr_scale},
            {"params": [params.scales], "lr": lr * spatial_lr_scale},
            {"params": [params.quats, params.opacities, params.colors], "lr": lr},
        ]
    )

    for _ in range(iters):
        optimizer.zero_grad(set_to_none=True)
        forward_params = GaussianParams(
            means=params.means,
            scales=params.scales.abs().clamp_min(1e-4),
            quats=params.quats,
            opacities=params.opacities.clamp(1e-4, 1.0),
            colors=params.colors.clamp(0.0, 1.0),
        )
        rendered, normals, surf_normals, distort = render_2dgs_train(
            forward_params, inputs, distloss=use_distloss
        )
        l1 = torch.abs(rendered - targets).mean()
        ssim_term = d_ssim_loss(rendered, targets)
        reg = surface_reg_loss(
            normals,
            surf_normals,
            distort,
            lambda_normal=lambda_normal,
            lambda_dist=lambda_dist,
        )
        loss = 0.8 * l1 + 0.2 * ssim_term + reg
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()

    with torch.no_grad():
        final_params = GaussianParams(
            means=params.means,
            scales=params.scales.abs().clamp_min(1e-4),
            quats=params.quats,
            opacities=params.opacities.clamp(1e-4, 1.0),
            colors=params.colors.clamp(0.0, 1.0),
        )

    return final_params, init_psnr_db
