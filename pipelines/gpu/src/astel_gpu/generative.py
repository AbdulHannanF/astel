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
import os
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
from .geometry_qa import GeometryScore, score_cloud
from .l2_triposplat import run_l2
from .l3_refine import DEFAULT_LAMBDA_NORMAL, optimize_2dgs, render_2dgs_colors
from .smoke_refit import RenderInputs, render_views
from .splat_clean import CleanConfig, clean_gaussians

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
#: Env switch (``ASTEL_L3_REFINE``) selecting the Tier-1 densified refine
#: (:func:`astel_gpu.refine.refine_with_densification` — unfrozen positions +
#: adaptive density control + perceptual loss) over the default frozen
#: distillation. Off by default: the densified path needs a Box A GPU run to
#: validate and tune before it becomes the default.
_DENSIFY_ENV = "ASTEL_L3_REFINE"
#: Env switch (``ASTEL_L3_MV_ENHANCE``) turning on the SDXL multi-view target
#: enhancer (:mod:`astel_gpu.mv_enhance`): the L2 orbit renders are img2img-enhanced
#: into higher-detail supervision and the densified refine chases them. This is the
#: verified Tier-1 unlock — the refine only exceeds L2 with such external targets.
#: Implies the densified refine. Off by default (extra SDXL pass + needs tuning).
_MV_ENHANCE_ENV = "ASTEL_L3_MV_ENHANCE"
#: Env override (``ASTEL_L3_MV_STRENGTH``) for the img2img SDEdit strength. Low keeps
#: views anchored to geometry (consistent); high lets each view invent incompatible
#: detail and the refine averages to mush.
_MV_STRENGTH_ENV = "ASTEL_L3_MV_STRENGTH"
DEFAULT_MV_STRENGTH = 0.3


def _resolve_flag(flag: bool | None, env_name: str) -> bool:
    """Resolve a boolean switch (explicit arg, else env truthiness, else off)."""
    if flag is not None:
        return flag
    return os.environ.get(env_name, "").strip().lower() in {"1", "true", "yes", "on"}


def _resolve_densify(flag: bool | None) -> bool:
    """Resolve the densified-refine switch (explicit arg, else env, else off)."""
    return _resolve_flag(flag, _DENSIFY_ENV)


def _resolve_mv_enhance(flag: bool | None) -> bool:
    """Resolve the multi-view-enhance switch (explicit arg, else env, else off)."""
    return _resolve_flag(flag, _MV_ENHANCE_ENV)


def _resolve_mv_strength(value: float | None) -> float:
    """Resolve the img2img enhance strength (explicit arg, else env, else default)."""
    if value is not None:
        return value
    raw = os.environ.get(_MV_STRENGTH_ENV, "").strip()
    if not raw:
        return DEFAULT_MV_STRENGTH
    try:
        return float(raw)
    except ValueError:
        return DEFAULT_MV_STRENGTH


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
    geometry_qa: dict[str, Any] | None = None,
    image_qa: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """``astel.quality-report/v0`` for a generated, surfelised L3 asset.

    ``geometry_qa`` (the degenerate-asset scorecard) and ``image_qa`` (the chosen
    reference image's scorecard, text/image paths only) are embedded under a
    ``"qa"`` block when supplied, so the Truth Meter can show *why* a generated
    asset was accepted (or flag a degenerate one) without claiming any measured
    accuracy. Both are pure self-assessments of a generated asset.
    """
    qa: dict[str, Any] = {}
    if geometry_qa is not None:
        qa["geometry"] = geometry_qa
    if image_qa is not None:
        qa["image"] = image_qa
    report = {
        "schema": "astel.quality-report/v0",
        # Honesty contract (CLAUDE.md §1.3/§8.4): this asset is distilled from the
        # TripoSplat L2 generator, NOT reconstructed from real capture, so its
        # origin is "generated". The Truth Meter reads this exact field to choose
        # its provenance pill; "measured" here would render a false "reconstructed
        # from real capture with ground-truth comparison" claim over a generated
        # object. Every other field (provenance, caveats, geometric_error.reason)
        # already says generated; this keeps origin consistent with them.
        "origin": "generated",
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
    if qa:
        report["qa"] = qa
    return report


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
    clean_config: CleanConfig | None = None,
    device_str: str | None = None,
    image_qa: dict[str, Any] | None = None,
    densify: bool | None = None,
    external_targets: torch.Tensor | None = None,
    prompt: str = "",
    mv_enhance: bool | None = None,
    mv_strength: float | None = None,
) -> L2L3Result:
    """image -> TripoSplat L2 -> clean -> normalise -> orbit -> 2DGS L3.

    Default L3 is the frozen distillation (positions fixed, fixed count, supervised
    by the L2 generator's own renders). When ``densify`` is true (or
    ``ASTEL_L3_REFINE`` is set) the Tier-1 densified refine runs instead
    (:func:`astel_gpu.refine.refine_with_densification`: unfrozen positions +
    adaptive density control + perceptual loss). ``external_targets`` — multi-view
    images rendered on the SAME train rig by a stronger generator (TRELLIS.2 /
    MVDream / SDS-enhanced views) — replaces the L2 self-renders as supervision and
    is the path by which the refine exceeds L2; absent, the L2 self-renders are used
    (a better-distillation bounded by L2).

    ``image_qa`` (the chosen reference image's critic scorecard, supplied by the
    text/image producer paths) is embedded into the quality report's ``qa`` block
    alongside the geometry critic computed here.
    """
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

    # Floater removal BEFORE normalise/distillation. TripoSplat sprays low-opacity
    # smoke, oversized halo blobs, needle streaks, and disconnected floaters; the
    # frozen-position L3 distillation would otherwise bake them in (it only learns
    # to reproduce the L2 appearance). Cleaning here also fixes framing: a far
    # floater no longer dominates ``normalize_params``'s radius. See
    # :mod:`astel_gpu.splat_clean`.
    clean_config = clean_config if clean_config is not None else CleanConfig.from_env()
    l2_params_clean, l2_clean_stats = clean_gaussians(l2_params_native, clean_config)
    l2_params, _center, radius = normalize_params(l2_params_clean)

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

    # Supervision targets. Priority: caller-supplied external multi-view images
    # (Tier-1, must match the train rig) > SDXL-enhanced L2 orbit renders
    # (ASTEL_L3_MV_ENHANCE) > the L2 generator's own renders (distillation).
    use_mv_enhance = _resolve_mv_enhance(mv_enhance) and external_targets is None
    mv_metrics: dict[str, Any] | None = None
    with torch.no_grad():
        if external_targets is not None:
            train_targets = external_targets
        else:
            base_train = render_views(l2_params, train_inputs)
            if use_mv_enhance:
                from .mv_enhance import enhance_views  # noqa: PLC0415 (SDXL-heavy)

                train_targets, mv_metrics = enhance_views(
                    base_train,
                    prompt=prompt,
                    strength=_resolve_mv_strength(mv_strength),
                    seed=seed,
                    device=device_str,
                )
            else:
                train_targets = base_train
        test_targets = render_views(l2_params, test_inputs)

    # MV-enhanced targets are only worth chasing with the densified refine (it can
    # grow splats to match the injected detail); imply it.
    use_densify = _resolve_densify(densify) or use_mv_enhance
    refine_metrics: dict[str, Any] | None = None
    start = time.perf_counter()
    if use_densify:
        from .refine import refine_with_densification  # noqa: PLC0415 (gsplat-heavy)

        l3_params, refine_metrics = refine_with_densification(
            l2_params,
            train_targets,
            train_inputs,
            iters=refine_iters,
            spatial_lr_scale=1.0,
            lambda_normal=lambda_normal,
            lambda_dist=lambda_dist,
        )
    else:
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

    # Cheap final pass: distillation can drive a few surfels to near-zero opacity
    # or extreme in-plane elongation. ``spatial=False`` skips SOR (positions are
    # unchanged from the already-outlier-removed L2 input, so it would be wasted).
    l3_params, l3_clean_stats = clean_gaussians(
        l3_params, clean_config, spatial=False
    )

    with torch.no_grad():
        test_psnr_db = psnr(render_2dgs_colors(l3_params, test_inputs), test_targets)

    peak_vram_gb = (
        torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else 0.0
    )

    # Degenerate-asset critic over the produced cloud (cheap stats; no render).
    # Feeds the Truth Meter and the producer's optional best-of-K asset re-roll.
    geom_score: GeometryScore = score_cloud(
        l3_params,
        clean_removed_fraction=l2_clean_stats.get("removed_fraction"),
        selfconsistency_psnr_db=test_psnr_db,
    )

    metrics: dict[str, Any] = {
        "origin": "measured",
        "stage": "l2->l3",
        "l2_metrics": l2.metrics,
        "l2_gaussians": l2.cloud.count,
        "l2_gaussians_cleaned": l2_params.count,
        "l3_gaussians": l3_params.count,
        "l2_clean": l2_clean_stats,
        "l3_clean": l3_clean_stats,
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
        "geometry_qa": geom_score.to_dict(),
        "densify": use_densify,
        "external_targets": external_targets is not None,
        "mv_enhance": use_mv_enhance,
        "mv_enhance_metrics": mv_metrics,
        "refine_metrics": refine_metrics,
    }
    report = build_generative_quality_report(
        count=l3_params.count,
        l2_count=l2.cloud.count,
        psnr_db=test_psnr_db,
        n_holdout_views=len(test_idx),
        image_path=str(image_path),
        geometry_qa=geom_score.to_dict(),
        image_qa=image_qa,
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
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help=(
            "Disable floater/needle/blob removal (A/B against the cleaned asset). "
            "Cleaning is ON by default; tune thresholds via ASTEL_CLEAN* env vars."
        ),
    )
    parser.add_argument("--out", type=Path, default=Path("out_generative"))
    args = parser.parse_args()

    from dataclasses import replace

    clean_config = replace(CleanConfig.from_env(), enabled=not args.no_clean)
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
        clean_config=clean_config,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    write_gaussian_ply(result.l2_params, args.out / "l2.ply")
    write_gaussian_ply(result.l3_params, args.out / "l3.ply")
    (args.out / "l2l3-metrics.json").write_text(json.dumps(result.metrics, indent=2))
    (args.out / "quality-report.json").write_text(json.dumps(result.report, indent=2))
    print(json.dumps({"metrics": result.metrics, "report": result.report}, indent=2))


if __name__ == "__main__":
    main()
