"""Render-then-refit smoke test for the gsplat differentiable rasterizer.

Self-contained (no dataset download):

1. Build a KNOWN target gaussian cloud (a small torus-knot ribbon).
2. Place ``n_views`` pinhole cameras on a sphere looking at the origin.
3. Render the target with ``gsplat.rasterization`` -> target images.
4. Initialize a FRESH random gaussian cloud and optimize its parameters
   (Adam, L1 + D-SSIM-ish loss) to match the target renders.
5. Report mean PSNR across views (initial vs. final) and write metrics +
   the optimized cloud as an INRIA-layout ``l3.ply``.

HONESTY NOTE: this is a self-consistency + convergence smoke test. The target
is rendered by gsplat and refit by gsplat — it proves the differentiable
rasterizer's forward+backward and the optimization loop work on this GPU. It
is NOT a ground-truth-geometry accuracy benchmark (that arrives with the
COLMAP / real-capture path).
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .cameras import build_camera_rig
from .export import psnr, write_gaussian_ply
from .gaussians import GaussianParams, build_random_init_cloud, build_target_cloud

DEFAULT_N_GAUSSIANS = 8_000
DEFAULT_N_VIEWS = 10
DEFAULT_IMAGE_SIZE = 256
DEFAULT_ITERS = 1500
SMOKE_PSNR_THRESHOLD_DB = 25.0


@dataclass
class RenderInputs:
    viewmats: torch.Tensor  # (n_views, 4, 4)
    ks: torch.Tensor  # (n_views, 3, 3)
    image_size: int  # square default; overridden by width/height if set
    width: int | None = None
    height: int | None = None

    @property
    def w(self) -> int:
        return self.width if self.width is not None else self.image_size

    @property
    def h(self) -> int:
        return self.height if self.height is not None else self.image_size


def render_views(
    params: GaussianParams, inputs: RenderInputs
) -> torch.Tensor:
    """Render ``params`` from every camera in ``inputs``. Returns (V, H, W, 3)."""
    import gsplat

    images, _alphas, _meta = gsplat.rasterization(
        means=params.means,
        quats=params.quats,
        scales=params.scales,
        opacities=params.opacities,
        colors=params.colors,
        viewmats=inputs.viewmats,
        Ks=inputs.ks,
        width=inputs.w,
        height=inputs.h,
        packed=False,
    )
    result: torch.Tensor = images[..., :3].clamp(0.0, 1.0)
    return result


def d_ssim_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """A lightweight 1 - SSIM proxy via a 5x5 box-filtered local SSIM.

    Operates on ``(V, H, W, 3)`` tensors in ``[0, 1]``; returns a scalar.
    """
    x = pred.permute(0, 3, 1, 2)
    y = target.permute(0, 3, 1, 2)
    kernel_size = 5
    pad = kernel_size // 2
    kernel = torch.ones(
        1, 1, kernel_size, kernel_size, device=x.device, dtype=x.dtype
    ) / (kernel_size * kernel_size)
    c = x.shape[1]
    kernel = kernel.expand(c, 1, kernel_size, kernel_size)

    def _blur(t: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.conv2d(t, kernel, padding=pad, groups=c)

    mu_x, mu_y = _blur(x), _blur(y)
    sigma_x = _blur(x * x) - mu_x * mu_x
    sigma_y = _blur(y * y) - mu_y * mu_y
    sigma_xy = _blur(x * y) - mu_x * mu_y

    c1, c2 = 0.01**2, 0.03**2
    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    )
    return (1.0 - ssim_map.mean()).clamp(0.0, 1.0)


def make_trainable(params: GaussianParams) -> GaussianParams:
    """Return a copy of ``params`` with ``requires_grad_`` set on all tensors."""
    return GaussianParams(
        means=params.means.clone().requires_grad_(True),
        scales=params.scales.clone().requires_grad_(True),
        quats=params.quats.clone().requires_grad_(True),
        opacities=params.opacities.clone().requires_grad_(True),
        colors=params.colors.clone().requires_grad_(True),
    )


def optimize(
    init: GaussianParams,
    targets: torch.Tensor,
    inputs: RenderInputs,
    iters: int,
    lr: float = 5e-3,
    spatial_lr_scale: float = 1.0,
) -> tuple[GaussianParams, float]:
    """Optimize ``init`` (in place, returned trainable) to match ``targets``.

    ``spatial_lr_scale`` multiplies the learning rate of the position- and
    size-carrying params (``means``, ``scales``) by the scene's spatial extent,
    the standard 3DGS trick: a fit in millimetres needs ~1000x larger position
    steps than one in unit-scale coords. Defaults to ``1.0`` (unchanged for the
    unit-scale synthetic/smoke callers).

    Returns ``(optimized_params, init_psnr_db)``.
    """
    params = make_trainable(init)

    with torch.no_grad():
        init_imgs = render_views(params, inputs)
        init_psnr_db = psnr(init_imgs, targets)

    optimizer = torch.optim.Adam(
        [
            {"params": [params.means], "lr": lr * spatial_lr_scale},
            {"params": [params.scales], "lr": lr * spatial_lr_scale},
            {"params": [params.quats, params.opacities, params.colors], "lr": lr},
        ]
    )

    for _ in range(iters):
        optimizer.zero_grad(set_to_none=True)
        # Keep scales positive and opacities/colors in [0, 1] via soft clamps
        # applied to the *forward* pass only; the underlying params stay
        # unconstrained for Adam.
        scales = params.scales.abs().clamp_min(1e-4)
        opacities = params.opacities.clamp(1e-4, 1.0)
        colors = params.colors.clamp(0.0, 1.0)
        forward_params = GaussianParams(
            means=params.means,
            scales=scales,
            quats=params.quats,
            opacities=opacities,
            colors=colors,
        )
        rendered = render_views(forward_params, inputs)
        l1 = torch.abs(rendered - targets).mean()
        ssim_term = d_ssim_loss(rendered, targets)
        loss = 0.8 * l1 + 0.2 * ssim_term
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


def run_smoke(
    iters: int = DEFAULT_ITERS,
    n_gaussians: int = DEFAULT_N_GAUSSIANS,
    n_views: int = DEFAULT_N_VIEWS,
    image_size: int = DEFAULT_IMAGE_SIZE,
    seed: int = 20260613,
    device_str: str = "cuda",
) -> tuple[GaussianParams, dict[str, Any]]:
    """Run the render-then-refit smoke test. Returns ``(final_params, metrics)``."""
    import gsplat

    device = torch.device(device_str)
    torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None

    target = build_target_cloud(n_gaussians, seed=seed, device=device)
    init = build_random_init_cloud(n_gaussians, seed=seed + 1, device=device)

    viewmats, ks = build_camera_rig(n_views, image_size)
    inputs = RenderInputs(
        viewmats=viewmats.to(device), ks=ks.to(device), image_size=image_size
    )

    with torch.no_grad():
        targets = render_views(target, inputs)

    start = time.perf_counter()
    final_params, init_psnr_db = optimize(init, targets, inputs, iters=iters)
    torch.cuda.synchronize() if device.type == "cuda" else None
    wall_time_s = time.perf_counter() - start

    with torch.no_grad():
        final_imgs = render_views(final_params, inputs)
        final_psnr_db = psnr(final_imgs, targets)

    peak_vram_gb = (
        torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else 0.0
    )

    metrics: dict[str, Any] = {
        "origin": "measured",
        "final_psnr_db": final_psnr_db,
        "init_psnr_db": init_psnr_db,
        "iters": iters,
        "wall_time_s": wall_time_s,
        "peak_vram_gb": peak_vram_gb,
        "gpu_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
        ),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gsplat_version": gsplat.__version__,
        "n_views": n_views,
        "n_gaussians": n_gaussians,
        "image_size": image_size,
        "note": (
            "Self-consistency + convergence smoke test - target rendered by "
            "gsplat and refit by gsplat; proves the differentiable "
            "rasterizer's forward+backward + the optimization loop work on "
            "this GPU. It is NOT a ground-truth-geometry accuracy benchmark "
            "(that arrives with the COLMAP/real-capture path)."
        ),
    }
    return final_params, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="gsplat render-then-refit smoke test.")
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    parser.add_argument("--n-gaussians", type=int, default=DEFAULT_N_GAUSSIANS)
    parser.add_argument("--n-views", type=int, default=DEFAULT_N_VIEWS)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--seed", type=int, default=20260613)
    parser.add_argument("--out", type=Path, default=Path("out"))
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    final_params, metrics = run_smoke(
        iters=args.iters,
        n_gaussians=args.n_gaussians,
        n_views=args.n_views,
        image_size=args.image_size,
        seed=args.seed,
        device_str=args.device,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    ply_path = args.out / "l3.ply"
    write_gaussian_ply(final_params, ply_path)

    metrics_path = args.out / "smoke-metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print(json.dumps(metrics, indent=2))
    print(f"Wrote {ply_path}")

    if metrics["final_psnr_db"] <= SMOKE_PSNR_THRESHOLD_DB:
        raise SystemExit(
            f"Smoke test FAILED: final PSNR {metrics['final_psnr_db']:.2f} dB "
            f"<= threshold {SMOKE_PSNR_THRESHOLD_DB} dB"
        )


if __name__ == "__main__":
    main()
