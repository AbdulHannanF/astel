"""Render a splat asset (``.ply``) to preview PNG frames — a QA / thumbnail tool.

Loads an INRIA-layout ``.ply`` (via :mod:`astel_splat_io`), normalises it to a
unit frame, places a turntable of pinhole cameras around it, renders with gsplat
and writes PNG frames + a horizontal contact-sheet montage.

The renderer uses the **3DGS** rasterizer (:func:`astel_gpu.smoke_refit.render_views`)
on purpose: the web viewer (Spark/Three.js) rasterises a ``.ply`` as 3D gaussians,
so a 3DGS preview here matches what a user actually sees in the product viewer —
including for a 2DGS-surfel L3 asset, whose flat gaussians the web viewer also
draws as 3DGS. This is the utility behind the "is the generated asset
photorealistic?" visual check (CLAUDE.md §8 Truth Meter / QA).

CPU-testable seam: :func:`turntable_centres` is pure (no torch device / gsplat);
the actual render needs a GPU (gsplat) and is exercised through ``run-python.cmd``.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from astel_splat_io.ply import read_ply

from .cameras import look_at_viewmats, pinhole_intrinsics
from .export import gaussian_params_from_splat_cloud
from .generative import normalize_params
from .smoke_refit import RenderInputs, render_views


def turntable_centres(
    n: int, radius: float, elevation_deg: float, up_axis: int = 2
) -> torch.Tensor:
    """Return ``(n, 3)`` camera centres orbiting the origin at fixed elevation.

    Cameras are spaced evenly in azimuth around ``up_axis`` (default +Z, the
    convention the generative pipeline normalises assets into) at a constant
    ``elevation_deg`` above the equator, all at distance ``radius`` from the
    origin. Pure: returns a CPU float32 tensor, no gsplat / CUDA needed.
    """
    elev = math.radians(elevation_deg)
    cos_e = math.cos(elev)
    up_val = radius * math.sin(elev)
    plane = [a for a in range(3) if a != up_axis]
    centres = torch.zeros(n, 3, dtype=torch.float32)
    for i in range(n):
        az = 2.0 * math.pi * i / max(1, n)
        centres[i, plane[0]] = radius * cos_e * math.cos(az)
        centres[i, plane[1]] = radius * cos_e * math.sin(az)
        centres[i, up_axis] = up_val
    return centres


def turntable_inputs(
    n_frames: int,
    image_size: int,
    *,
    radius: float = 3.0,
    elevation_deg: float = 15.0,
    fov_deg: float = 50.0,
    device: torch.device | None = None,
) -> RenderInputs:
    """Build :class:`RenderInputs` for a turntable orbit at fixed elevation."""
    centres = turntable_centres(n_frames, radius, elevation_deg)
    target = torch.zeros(3, dtype=torch.float32)
    world_up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
    viewmats = look_at_viewmats(centres, target, world_up)
    ks = pinhole_intrinsics(image_size, fov_deg).unsqueeze(0).repeat(n_frames, 1, 1)
    if device is not None:
        viewmats, ks = viewmats.to(device), ks.to(device)
    return RenderInputs(viewmats=viewmats, ks=ks, image_size=image_size)


def render_ply_turntable(
    ply_path: str | Path,
    *,
    n_frames: int = 4,
    image_size: int = 512,
    radius: float = 3.0,
    elevation_deg: float = 15.0,
    device_str: str | None = None,
) -> np.ndarray:
    """Render ``ply_path`` as a turntable. Returns ``(n_frames, H, W, 3)`` uint8.

    The cloud is normalised to a unit frame (centroid → origin, unit radius) so
    the fixed turntable rig frames any asset, exactly as the generative L3
    distillation normalises before rendering.
    """
    device_str = device_str or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    cloud = read_ply(ply_path)
    params = gaussian_params_from_splat_cloud(cloud, device)
    params, _center, _radius = normalize_params(params)

    inputs = turntable_inputs(
        n_frames, image_size, radius=radius, elevation_deg=elevation_deg, device=device
    )
    with torch.no_grad():
        imgs = render_views(params, inputs)  # (n, H, W, 3) in [0, 1]
    arr: np.ndarray = (imgs.clamp(0.0, 1.0).cpu().numpy() * 255.0).astype(np.uint8)
    return arr


def _save_png(frame: np.ndarray, path: Path) -> None:
    from PIL import Image

    Image.fromarray(frame, mode="RGB").save(path)


def save_montage(frames: np.ndarray, path: Path, gap: int = 8) -> None:
    """Write a horizontal contact sheet of ``frames`` ``(n, H, W, 3)`` uint8."""
    from PIL import Image

    n, h, w, _ = frames.shape
    sheet = np.full((h, w * n + gap * (n - 1), 3), 255, dtype=np.uint8)
    for i in range(n):
        x = i * (w + gap)
        sheet[:, x : x + w] = frames[i]
    Image.fromarray(sheet, mode="RGB").save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--radius", type=float, default=3.0)
    parser.add_argument("--elevation-deg", type=float, default=15.0)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    frames = render_ply_turntable(
        args.ply,
        n_frames=args.frames,
        image_size=args.image_size,
        radius=args.radius,
        elevation_deg=args.elevation_deg,
    )
    for i in range(frames.shape[0]):
        _save_png(frames[i], args.out / f"frame_{i:02d}.png")
    save_montage(frames, args.out / "montage.png")
    print(f"wrote {frames.shape[0]} frames + montage.png to {args.out}")


if __name__ == "__main__":
    main()
