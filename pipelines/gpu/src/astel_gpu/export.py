"""Convert :class:`astel_gpu.gaussians.GaussianParams` to an INRIA-layout
:class:`astel_splat_io.cloud.SplatCloud` and write it to ``.ply``.

Field-order and value conventions (CLAUDE.md, load-bearing):
``x y z, f_dc_0..2, opacity (logit), scale_0..2 (log), rot_0..3 (w,x,y,z)``.
``albedo = 0.5 + SH_C0 * f_dc`` and ``alpha = sigmoid(opacity_logit)``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from astel_splat_io.cloud import SH_C0, SplatCloud
from astel_splat_io.ply import write_ply

from .gaussians import GaussianParams


def to_splat_cloud(params: GaussianParams) -> SplatCloud:
    """Convert training-space ``params`` to an INRIA-convention SplatCloud."""
    positions = params.means.detach().cpu().numpy().astype(np.float32)

    colors = params.colors.detach().cpu().numpy().astype(np.float32)
    colors_dc = ((colors - 0.5) / SH_C0).astype(np.float32)

    scales = params.scales.detach().cpu().numpy().astype(np.float32)
    log_scales = np.log(np.clip(scales, 1e-8, None)).astype(np.float32)

    opacities = params.opacities.detach().cpu().numpy().astype(np.float32)
    eps = 1e-6
    alpha = np.clip(opacities, eps, 1.0 - eps)
    opacity_logit = np.log(alpha / (1.0 - alpha)).astype(np.float32)

    quats = params.quats.detach().cpu().numpy().astype(np.float32)
    norm = np.linalg.norm(quats, axis=1, keepdims=True)
    norm = np.where(norm == 0.0, 1.0, norm)
    quats = (quats / norm).astype(np.float32)

    return SplatCloud(
        positions=positions,
        colors_dc=colors_dc,
        opacity=opacity_logit,
        log_scales=log_scales,
        quats=quats,
    )


def gaussian_params_from_splat_cloud(
    cloud: SplatCloud, device: torch.device
) -> GaussianParams:
    """Inverse of :func:`to_splat_cloud`: INRIA SplatCloud -> training-space params.

    Maps the archival/logit conventions back to the activated tensors
    ``gsplat.rasterization`` consumes directly: ``scales = exp(log_scales)``,
    ``opacities = sigmoid(opacity_logit)``, ``colors = 0.5 + SH_C0 * colors_dc``
    (clamped to ``[0, 1]``), quats passed through (already w,x,y,z normalised).
    Used to feed a generated L2 cloud (e.g. TripoSplat) into the L3 refiner.
    """
    means = torch.from_numpy(cloud.positions.astype(np.float32)).to(device)
    scales = torch.from_numpy(np.exp(cloud.log_scales).astype(np.float32)).to(device)
    quats = torch.from_numpy(cloud.quats.astype(np.float32)).to(device)
    opacities = torch.sigmoid(
        torch.from_numpy(cloud.opacity.astype(np.float32))
    ).to(device)
    colors = (0.5 + SH_C0 * torch.from_numpy(cloud.colors_dc.astype(np.float32))).clamp(
        0.0, 1.0
    ).to(device)
    return GaussianParams(
        means=means, scales=scales, quats=quats, opacities=opacities, colors=colors
    )


def write_gaussian_ply(params: GaussianParams, path: str | Path) -> int:
    """Convert ``params`` to a SplatCloud and write it as an INRIA ``.ply``."""
    cloud = to_splat_cloud(params)
    return int(write_ply(cloud, path))


@torch.no_grad()
def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean PSNR (dB) between ``pred`` and ``target`` images in ``[0, 1]``."""
    mse = torch.mean((pred - target) ** 2).clamp_min(1e-10)
    return float(10.0 * torch.log10(1.0 / mse))
