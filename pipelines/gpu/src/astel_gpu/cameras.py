"""Pinhole camera rig helpers for the smoke-test render-then-refit pipeline.

All cameras look at the world origin from points evenly distributed on a
sphere (a Fibonacci spiral), in OpenCV/COLMAP camera convention: +Z forward
(into the scene), +X right, +Y down. ``gsplat.rasterization`` expects
world-to-camera ``viewmats`` in this convention.
"""

from __future__ import annotations

import math

import torch


def fibonacci_sphere(n: int, radius: float) -> torch.Tensor:
    """Return ``(n, 3)`` camera centres roughly evenly spread on a sphere."""
    indices = torch.arange(0, n, dtype=torch.float64)
    golden_ratio = (1.0 + math.sqrt(5.0)) / 2.0
    theta = 2.0 * math.pi * indices / golden_ratio
    # z in (-1, 1), avoiding the poles for a stable "up" vector.
    z = 1.0 - (indices + 0.5) * 2.0 / n
    r_xy = torch.sqrt(torch.clamp(1.0 - z * z, min=0.0))
    x = r_xy * torch.cos(theta)
    y = r_xy * torch.sin(theta)
    points = torch.stack([x, y, z], dim=1) * radius
    return points.to(torch.float32)


def look_at_viewmats(
    centres: torch.Tensor, target: torch.Tensor, world_up: torch.Tensor
) -> torch.Tensor:
    """Build ``(n, 4, 4)`` world-to-camera matrices for cameras at ``centres``.

    Camera convention: +Z forward (towards ``target``), +X right, +Y down
    (OpenCV/COLMAP), matching what ``gsplat.rasterization`` expects for
    ``viewmats``.
    """
    n = centres.shape[0]
    forward = target[None, :] - centres  # (n, 3), points towards target
    forward = forward / forward.norm(dim=-1, keepdim=True)

    right = torch.cross(forward, world_up[None, :].expand(n, -1), dim=-1)
    right = right / right.norm(dim=-1, keepdim=True)

    down = torch.cross(forward, right, dim=-1)  # +Y down for OpenCV convention
    down = down / down.norm(dim=-1, keepdim=True)

    # Rotation rows are the camera axes expressed in world space (R: world->cam
    # is the transpose of [right | down | forward] as columns, i.e. these rows).
    rot = torch.stack([right, down, forward], dim=1)  # (n, 3, 3)
    trans = -torch.bmm(rot, centres[:, :, None]).squeeze(-1)  # (n, 3)

    viewmats = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(n, 1, 1)
    viewmats[:, :3, :3] = rot
    viewmats[:, :3, 3] = trans
    return viewmats


def pinhole_intrinsics(
    image_size: int, fov_deg: float = 50.0
) -> torch.Tensor:
    """Return a single ``(3, 3)`` pinhole intrinsics matrix ``K``."""
    focal = image_size / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    cx = cy = image_size / 2.0
    k = torch.tensor(
        [
            [focal, 0.0, cx],
            [0.0, focal, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    return k


def build_camera_rig(
    n_views: int, image_size: int, radius: float = 3.0, fov_deg: float = 50.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build ``(viewmats, Ks)`` for ``n_views`` cameras on a sphere.

    ``viewmats`` is ``(n_views, 4, 4)`` world-to-camera; ``Ks`` is
    ``(n_views, 3, 3)`` (the same intrinsics, repeated per view).
    """
    centres = fibonacci_sphere(n_views, radius)
    target = torch.zeros(3, dtype=torch.float32)
    world_up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
    viewmats = look_at_viewmats(centres, target, world_up)
    k = pinhole_intrinsics(image_size, fov_deg)
    ks = k.unsqueeze(0).repeat(n_views, 1, 1)
    return viewmats, ks
