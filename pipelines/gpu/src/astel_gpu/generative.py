"""Generative L2 -> L3 wiring: TripoSplat L2 cloud -> surface-aligned 2DGS L3.

The capture path (:mod:`astel_gpu.capture_eval`) refines against REAL photos with
known poses. The generative path has no such ground truth — a single-image
generator (TripoSplat, L2) produces gaussians for an object that was never
photographed from every angle. So the L3 refinement here is a **distillation**:
render the L2 cloud from an orbit of synthetic views and refine a surface-aligned
2DGS surfel cloud (real per-splat normals, the chosen L3 representation — see
DECISIONS #1) to reproduce those renders. The result is a surfelised L3 asset
carrying normals for L4/L5, derived from the generated L2.

HONESTY — what the number means and does NOT:
  * The reported PSNR is **self-consistency / distillation fidelity**: how well the
    2DGS L3 reproduces the L2 generator's appearance on HELD-OUT orbit views. It is
    NOT accuracy versus any real object — generated content has no ground-truth scan.
  * ``geometric_error`` vs a GT cloud and metric ``scale`` are honestly ``None``:
    this asset is generated, normalised to a unit frame, with no metric grounding
    (scale is the separate VLM-estimation stage, not performed here).
  * provenance is fully generated (``generated_ratio = 1.0``) — the confidence
    channel must never imply this is measured reality.
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
from .capture_eval import split_train_test
from .export import (
    gaussian_params_from_splat_cloud,
    psnr,
    write_gaussian_ply,
)
from .gaussians import GaussianParams
from .l2_triposplat import run_l2
from .l3_refine import DEFAULT_LAMBDA_NORMAL, optimize_2dgs, render_2dgs_colors
from .smoke_refit import RenderInputs, render_views

DEFAULT_N_VIEWS = 24
#: Distillation supervision resolution. Raised 256 -> 512: at 256px a 262k-splat
#: cloud is starved of detail to fit (the target only has 256^2 pixels), so the
#: surfels can't learn fine structure. 512px lets the high-count L3 reproduce the
#: generator's wing-vein / fine-detail appearance (measured visual gain).
DEFAULT_IMAGE_SIZE = 512
#: Native generator budget. TripoSplat supports 32768..262144 (``_NUM_GAUSSIANS_MAX``)
#: and its own examples use 262144; we previously capped at 65536 (1/4 the budget),
#: which is below even the CLAUDE.md §3 "lowpoly-splat" (100k) tier. 262144 is the
#: generator's native max and is decisively sharper (measured).
DEFAULT_NUM_GAUSSIANS = 262144
DEFAULT_REFINE_ITERS = 600
#: Position-LR scale for the generative L3 refine. The TripoSplat L2 init is
#: already a clean, high-count, on-surface cloud; full-rate position steps for
#: 1500 iters drift splats into floaters that degrade the asset and inflate its
#: radius. ``0.0`` freezes positions, turning L3 into a *surfelization* of the
#: proven L2 geometry (scales/opacity/colour/quats still adapt). See
#: :func:`astel_gpu.l3_refine.optimize_2dgs`.
DEFAULT_MEANS_LR_SCALE = 0.0
DEFAULT_HOLDOUT_EVERY = 6


def normalize_params(
    params: GaussianParams,
) -> tuple[GaussianParams, torch.Tensor, float]:
    """Center on centroid, scale to unit radius. Returns (params, center, radius).

    A similarity transform applied to ``means`` (translate + scale) and ``scales``
    (scale only); colors/opacity/quats are invariant. Lets the fixed unit-sphere
    camera rig frame any generated object, and keeps the L3 optimizer at unit
    spatial scale. Pure (CPU-runnable); no gsplat needed.
    """
    center = params.means.mean(dim=0)
    centered = params.means - center
    radius = float(centered.norm(dim=-1).max().clamp_min(1e-8))
    means = centered / radius
    scales = params.scales / radius
    return (
        GaussianParams(
            means=means,
            scales=scales,
            quats=params.quats,
            opacities=params.opacities,
            colors=params.colors,
        ),
        center,
        radius,
    )


def build_generative_quality_report(
    *,
    count: int,
    l2_count: int,
    psnr_db: float,
    n_holdout_views: int,
    image_path: str,
) -> dict[str, Any]:
    """``astel.quality-report/v0`` for a generated, surfelised L3 asset."""
    return {
        "schema": "astel.quality-report/v0",
        "origin": "measured",
        "modality": "generative-image/triposplat-l2->2dgs-l3",
        "representation": "2dgs",
        "splats": count,
        "geometric_error": {
            "chamfer_mm_vs_l1": None,
            "method": None,
            "reason": (
                "Generated object: there is no ground-truth scan or measured L1 "
                "cloud to compare against. The L3 is distilled from the L2 "
                "generator, not reconstructed from real capture, so geometric "
                "accuracy vs reality is undefined here."
            ),
        },
        "fidelity": {
            "psnr_db": psnr_db,
            "ssim": None,
            "lpips": None,
            "n_holdout_views": n_holdout_views,
            "psnr_note": (
                "SELF-CONSISTENCY / distillation fidelity: how well the 2DGS L3 "
                "reproduces the TripoSplat L2 generator's appearance on HELD-OUT "
                "orbit views. NOT accuracy versus any real object."
            ),
        },
        "scale": {
            "longest_axis_m": None,
            "confidence": None,
            "method": "estimate",
            "reason": (
                "Generated asset normalised to a unit frame; no metric-scale "
                "grounding (SfM scale or VLM size estimate) is performed here."
            ),
        },
        "provenance": {"measured_ratio": 0.0, "generated_ratio": 1.0},
        "caveats": [
            "Fully GENERATED asset (image -> TripoSplat L2 -> 2DGS L3 distillation). "
            "Nothing here is measured against reality.",
            "fidelity.psnr_db is held-out self-consistency vs the L2 renders "
            "(distillation fidelity), not geometric accuracy.",
            f"L2 had {l2_count} gaussians; L3 surfelised to {count}.",
            "geometric_error and scale are explicitly None (not fabricated): no GT "
            "geometry and no metric scale exist for a generated object.",
        ],
    }


@dataclass
class L2L3Result:
    l2_params: GaussianParams
    l3_params: GaussianParams
    metrics: dict[str, Any]
    report: dict[str, Any]


def run_l2_to_l3(
    image_path: str | Path,
    *,
    num_gaussians: int = DEFAULT_NUM_GAUSSIANS,
    steps: int = 20,
    seed: int = 0,
    n_views: int = DEFAULT_N_VIEWS,
    image_size: int = DEFAULT_IMAGE_SIZE,
    refine_iters: int = DEFAULT_REFINE_ITERS,
    holdout_every: int = DEFAULT_HOLDOUT_EVERY,
    lambda_normal: float = DEFAULT_LAMBDA_NORMAL,
    lambda_dist: float = 0.0,
    means_lr_scale: float = DEFAULT_MEANS_LR_SCALE,
    device_str: str | None = None,
) -> L2L3Result:
    """image -> TripoSplat L2 -> normalise -> render orbit -> 2DGS L3 distillation."""
    device_str = device_str or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # L2: single image -> generated gaussians (TripoSplat).
    l2 = run_l2(
        image_path, num_gaussians=num_gaussians, steps=steps, seed=seed,
        device=device_str,
    )
    l2_params_native = gaussian_params_from_splat_cloud(l2.cloud, device)
    l2_params, _center, radius = normalize_params(l2_params_native)

    # Orbit rig (unit-sphere); split into train/held-out for an honest PSNR.
    viewmats, ks = build_camera_rig(n_views, image_size)
    inputs = RenderInputs(
        viewmats=viewmats.to(device), ks=ks.to(device), image_size=image_size
    )
    train_idx, test_idx = split_train_test(n_views, holdout_every)
    train_inputs = RenderInputs(
        viewmats=inputs.viewmats[train_idx], ks=inputs.ks[train_idx],
        image_size=image_size,
    )
    test_inputs = RenderInputs(
        viewmats=inputs.viewmats[test_idx], ks=inputs.ks[test_idx],
        image_size=image_size,
    )

    # Distillation targets = the L2 generator rendered as 3DGS.
    with torch.no_grad():
        train_targets = render_views(l2_params, train_inputs)
        test_targets = render_views(l2_params, test_inputs)

    start = time.perf_counter()
    l3_params, _init_psnr = optimize_2dgs(
        l2_params,
        train_targets,
        train_inputs,
        iters=refine_iters,
        spatial_lr_scale=1.0,
        lambda_normal=lambda_normal,
        lambda_dist=lambda_dist,
        means_lr_scale=means_lr_scale,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    refine_time_s = time.perf_counter() - start

    with torch.no_grad():
        test_psnr_db = psnr(render_2dgs_colors(l3_params, test_inputs), test_targets)

    peak_vram_gb = (
        torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else 0.0
    )

    metrics: dict[str, Any] = {
        "origin": "measured",
        "stage": "l2->l3",
        "l2_metrics": l2.metrics,
        "l2_gaussians": l2.cloud.count,
        "l3_gaussians": l3_params.count,
        "normalize_radius": radius,
        "selfconsistency_test_psnr_db": test_psnr_db,
        "n_views": n_views,
        "n_train_views": len(train_idx),
        "n_test_views": len(test_idx),
        "image_size": image_size,
        "refine_iters": refine_iters,
        "lambda_normal": lambda_normal,
        "lambda_dist": lambda_dist,
        "means_lr_scale": means_lr_scale,
        "refine_wall_time_s": refine_time_s,
        "peak_vram_gb": peak_vram_gb,
        "image_used": str(image_path),
    }
    report = build_generative_quality_report(
        count=l3_params.count,
        l2_count=l2.cloud.count,
        psnr_db=test_psnr_db,
        n_holdout_views=len(test_idx),
        image_path=str(image_path),
    )
    return L2L3Result(
        l2_params=l2_params, l3_params=l3_params, metrics=metrics, report=report
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--num-gaussians", type=int, default=DEFAULT_NUM_GAUSSIANS)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-views", type=int, default=DEFAULT_N_VIEWS)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--refine-iters", type=int, default=DEFAULT_REFINE_ITERS)
    parser.add_argument("--lambda-normal", type=float, default=DEFAULT_LAMBDA_NORMAL)
    parser.add_argument("--lambda-dist", type=float, default=0.0)
    parser.add_argument(
        "--means-lr-scale", type=float, default=DEFAULT_MEANS_LR_SCALE
    )
    parser.add_argument("--out", type=Path, default=Path("out_generative"))
    args = parser.parse_args()

    result = run_l2_to_l3(
        args.image,
        num_gaussians=args.num_gaussians,
        steps=args.steps,
        seed=args.seed,
        n_views=args.n_views,
        image_size=args.image_size,
        refine_iters=args.refine_iters,
        lambda_normal=args.lambda_normal,
        lambda_dist=args.lambda_dist,
        means_lr_scale=args.means_lr_scale,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    write_gaussian_ply(result.l2_params, args.out / "l2.ply")
    write_gaussian_ply(result.l3_params, args.out / "l3.ply")
    (args.out / "l2l3-metrics.json").write_text(json.dumps(result.metrics, indent=2))
    (args.out / "quality-report.json").write_text(json.dumps(result.report, indent=2))
    print(json.dumps({"metrics": result.metrics, "report": result.report}, indent=2))


if __name__ == "__main__":
    main()
