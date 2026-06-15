"""A deterministic, KNOWN synthetic object for ground-truth geometry eval.

Builds a sphere-shell point cloud (a "globe") with a fixed seed and a fixed
metric scale: the longest axis (diameter) is exactly
:data:`SYNTHETIC_LONGEST_AXIS_M` (0.20 m) by construction. Because this scene
is fully controlled, the points returned by :func:`build_ground_truth_points`
ARE the ground truth -- no estimation involved.

This is used by :mod:`astel_gpu.synthetic_eval` to produce the first REAL
measured Chamfer distance and scale numbers for the Truth Meter, clearly
labeled as a controlled synthetic benchmark (not a real-world capture).
"""

from __future__ import annotations

import torch

from .gaussians import GaussianParams

#: Longest axis (diameter) of the synthetic object, in metres, by construction.
SYNTHETIC_LONGEST_AXIS_M = 0.20

#: Radius of the sphere shell, in metres (diameter == SYNTHETIC_LONGEST_AXIS_M).
_RADIUS_M = SYNTHETIC_LONGEST_AXIS_M / 2.0

#: Thickness of the rendered shell splats relative to the radius.
_SHELL_THICKNESS = 0.04 * _RADIUS_M


def build_ground_truth_points(n: int, seed: int) -> torch.Tensor:
    """Return ``(n, 3)`` points sampled deterministically on a sphere shell.

    Points are placed on a Fibonacci-spiral sphere of radius ``_RADIUS_M`` so
    the longest axis (diameter) of the resulting cloud is exactly
    :data:`SYNTHETIC_LONGEST_AXIS_M` by construction.
    """
    indices = torch.arange(0, n, dtype=torch.float64)
    golden_ratio = (1.0 + 5.0**0.5) / 2.0
    theta = 2.0 * torch.pi * indices / golden_ratio
    z = 1.0 - (indices + 0.5) * 2.0 / n
    r_xy = torch.sqrt(torch.clamp(1.0 - z * z, min=0.0))
    x = r_xy * torch.cos(theta)
    y = r_xy * torch.sin(theta)
    points = torch.stack([x, y, z], dim=1) * _RADIUS_M
    return points.to(torch.float32)


def build_synthetic_target_cloud(
    n: int, seed: int, device: torch.device
) -> GaussianParams:
    """Build the known synthetic target as a :class:`GaussianParams` cloud.

    The cloud's ``means`` ARE the ground-truth points from
    :func:`build_ground_truth_points` (this is the "L1 reference" cloud). Per-
    point colors are a deterministic function of position (a latitude/longitude
    gradient) so the rendered views carry usable signal for the refit.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)

    means = build_ground_truth_points(n, seed=seed).to(
        device=device, dtype=torch.float32
    )

    scales = (
        _SHELL_THICKNESS * (0.6 + 0.8 * torch.rand(n, 3, generator=gen))
    ).to(device=device, dtype=torch.float32)

    quats = torch.zeros(n, 4, dtype=torch.float32, device=device)
    quats[:, 0] = 1.0  # identity orientation; gsplat normalizes

    opacities = torch.full((n,), 0.95, dtype=torch.float32, device=device)

    # Deterministic color gradient by latitude/longitude: coral -> azure.
    points_cpu = means.detach().cpu()
    norm = points_cpu.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    unit = points_cpu / norm
    lat = 0.5 + 0.5 * unit[:, 2:3]  # in [0, 1]
    coral = torch.tensor([0.94, 0.50, 0.45])
    azure = torch.tensor([0.20, 0.45, 0.85])
    rgb = coral[None, :] * (1.0 - lat) + azure[None, :] * lat
    colors = rgb.to(device=device, dtype=torch.float32).clamp(0.0, 1.0)

    return GaussianParams(
        means=means,
        scales=scales,
        quats=quats,
        opacities=opacities,
        colors=colors,
    )
