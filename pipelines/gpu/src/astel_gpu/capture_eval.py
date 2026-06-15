"""Real-world capture geometry eval on DTU (the first non-synthetic Truth Meter).

Fits a gaussian cloud to a DTU scan's real photographs using DTU's supplied
metric camera poses (millimetres, GT frame), then measures REAL geometry vs the
structured-light ground-truth scan following DTU's own evaluation protocol
(``Matlab evaluation code/PointCompareMain.m``): accuracy = data points inside
the **ObsMask** observable volume -> nearest GT; completeness = GT points above
the ground **Plane** -> nearest data; distances capped at 60 mm. Because the
poses are in the GT frame, NO registration is needed.

HONESTY -- exactly what this measures and does not:
  * REAL geometric accuracy of our gaussian fit vs a REAL scanned object (mm),
    using DTU's official ObsMask/Plane masking (not a box proxy).
  * It uses DTU's LAB-CALIBRATED poses, not poses we estimated -- so it isolates
    splat-fitting geometry from pose error. The COLMAP front-end (poses from the
    images) is the separate :mod:`astel_gpu.capture_sfm` validation.
  * Scale is INHERITED from DTU's metric calibration, not estimated by us.
  * PSNR is HELD-OUT (fit on train views, measured on unseen test views), though
    still background-capped (object-only gaussians don't model the background).
  * Raw 3DGS, no densification / surface regularization -- a BASELINE the
    surface-aligned L3 (2DGS/PGSR) must beat.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .dtu import (
    DtuScan,
    load_dtu_scan,
    load_obsmask,
    load_plane,
    load_ply_points,
    points_above_plane,
    points_in_obsmask,
)
from .export import psnr, write_gaussian_ply
from .gaussians import GaussianParams
from .l3_refine import (
    DEFAULT_LAMBDA_DIST,
    DEFAULT_LAMBDA_NORMAL,
    optimize_2dgs,
    render_2dgs_colors,
)
from .metrics import nn_distances
from .smoke_refit import DEFAULT_ITERS, RenderInputs, optimize, render_views

DEFAULT_N_GAUSSIANS = 100_000
DEFAULT_DOWNSCALE = 4
DEFAULT_HOLDOUT_EVERY = 8
CONTRIBUTING_OPACITY_THRESHOLD = 0.5
#: DTU caps per-point distances at 60 mm before averaging (PointCompareMain.m).
MAX_DIST_MM = 60.0


def estimate_object_half_extent(scan: DtuScan) -> float:
    """GT-free object half-size (mm) from camera geometry (visible extent at depth)."""
    half_extents = []
    for vm, k in zip(scan.viewmats, scan.ks, strict=True):
        centre = -vm[:3, :3].T @ vm[:3, 3]
        distance = float(np.linalg.norm(centre - scan.object_center))
        fy = float(k[1, 1])
        half_extents.append(distance * (scan.height / 2.0) / fy)
    # Object fills roughly half the frame on a DTU tabletop capture.
    return float(np.median(half_extents)) * 0.5


def split_train_test(n_views: int, holdout_every: int) -> tuple[list[int], list[int]]:
    """Every ``holdout_every``-th view is held out for test; the rest train."""
    test = list(range(0, n_views, holdout_every))
    train = [i for i in range(n_views) if i not in set(test)]
    return train, test


def build_init_cloud(
    n: int, center: np.ndarray, spread: float, device: torch.device, seed: int
) -> GaussianParams:
    """Random init cloud centred on ``center`` (mm) with half-extent ``spread``."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    center_t = torch.tensor(center, dtype=torch.float32)
    means = center_t[None, :] + (torch.rand(n, 3, generator=gen) - 0.5) * 2.0 * spread
    scales = (spread / 30.0) * (0.5 + torch.rand(n, 3, generator=gen))
    quats = torch.zeros(n, 4)
    quats[:, 0] = 1.0
    quats += 0.01 * torch.randn(n, 4, generator=gen)
    opacities = torch.full((n,), 0.5)
    colors = torch.rand(n, 3, generator=gen)
    return GaussianParams(
        means=means.to(device),
        scales=scales.to(device),
        quats=quats.to(device),
        opacities=opacities.to(device),
        colors=colors.to(device),
    )


def _render_inputs(scan: DtuScan, idx: list[int], device: torch.device) -> RenderInputs:
    return RenderInputs(
        viewmats=torch.tensor(scan.viewmats[idx], dtype=torch.float32, device=device),
        ks=torch.tensor(scan.ks[idx], dtype=torch.float32, device=device),
        image_size=max(scan.width, scan.height),
        width=scan.width,
        height=scan.height,
    )


def build_capture_quality_report(
    *,
    count: int,
    psnr_db: float,
    n_holdout_views: int,
    accuracy_mm: float,
    completeness_mm: float,
    n_data_eval: int,
    n_gt_eval: int,
    fitted_longest_axis_mm: float,
    gt_longest_axis_mm: float,
    scan_name: str,
    representation: str = "3dgs",
) -> dict[str, Any]:
    """Build the ``astel.quality-report/v0`` dict with REAL real-world numbers."""
    overall = 0.5 * (accuracy_mm + completeness_mm)
    return {
        "schema": "astel.quality-report/v0",
        "origin": "measured",
        "modality": f"capture-dtu/{scan_name}",
        "representation": representation,
        "splats": count,
        "geometric_error": {
            "chamfer_mm_vs_l1": overall,
            "accuracy_data_to_gt_mm": accuracy_mm,
            "completeness_gt_to_data_mm": completeness_mm,
            "n_data_points_evaluated": n_data_eval,
            "n_gt_points_evaluated": n_gt_eval,
            "max_dist_cap_mm": MAX_DIST_MM,
            "opacity_threshold": CONTRIBUTING_OPACITY_THRESHOLD,
            "method": "dtu-obsmask-plane-object-volume",
            "reason": (
                "DTU ObsMask/Plane masking (PointCompareMain.m), restricted to "
                "the observable OBJECT volume: accuracy = fitted gaussians in "
                "the ObsMask -> nearest GT; completeness = GT in (ObsMask AND "
                "above the ground Plane) -> nearest fitted gaussian; per-point "
                "distances capped at 60 mm. We intersect ObsMask with the plane "
                "for completeness (DTU's leaderboard uses above-plane only) "
                "because Astel reconstructs the object, not the full scene. "
                "chamfer_mm_vs_l1 is the overall mean (accuracy+completeness)/2."
            ),
        },
        "fidelity": {
            "psnr_db": psnr_db,
            "ssim": None,
            "lpips": None,
            "n_holdout_views": n_holdout_views,
            "psnr_note": (
                "HELD-OUT PSNR (fit on train views, measured on unseen test "
                "views). Background-capped: object-only gaussians do not model "
                "the real background filling each frame."
            ),
        },
        "scale": {
            "longest_axis_m": fitted_longest_axis_mm / 1000.0,
            "gt_longest_axis_m": gt_longest_axis_mm / 1000.0,
            "relative_error": None,
            "confidence": 1.0,
            "method": "dtu-metric-poses-inherited",
            "reason": (
                "Scale is INHERITED from DTU's metric calibration (mm), not "
                "estimated -- there is no scale ERROR to report. A scale-"
                "ESTIMATION number requires the pose-free capture path."
            ),
        },
        "provenance": {"measured_ratio": 1.0, "generated_ratio": 0.0},
        "caveats": [
            "REAL-WORLD geometry via DTU's official ObsMask/Plane protocol vs a "
            "real structured-light scan. Supersedes the session-9 box proxy.",
            "Uses DTU's LAB-CALIBRATED poses, not poses Astel estimated -- it "
            "isolates splat-fitting geometry from pose error. The COLMAP "
            "front-end (poses from images) is the capture_sfm validation.",
            (
                "Raw 3DGS baseline (no densification / surface regularization) "
                "-- the number the surface-aligned L3 (2DGS) must beat."
                if representation == "3dgs"
                else "Surface-aligned L3 via gsplat 2DGS (normal-consistency "
                "regularization); compare its overall_mm against the raw-3DGS "
                "baseline run with identical init/iters/gaussian-count."
            ),
            "fidelity.psnr_db is held-out but background-capped (object-only fit).",
            "We skip DTU's 0.2 mm point-density reduction; effect on the mean "
            "is minor at our gaussian counts but noted for exact comparability.",
        ],
    }


def run_capture_eval(
    image_dir: Path,
    pos_dir: Path,
    gt_ply: Path,
    obsmask_path: Path,
    plane_path: Path,
    *,
    iters: int = DEFAULT_ITERS,
    n_gaussians: int = DEFAULT_N_GAUSSIANS,
    downscale: int = DEFAULT_DOWNSCALE,
    holdout_every: int = DEFAULT_HOLDOUT_EVERY,
    seed: int = 20260614,
    device_str: str = "cuda",
    representation: str = "3dgs",
    lambda_normal: float = DEFAULT_LAMBDA_NORMAL,
    lambda_dist: float = DEFAULT_LAMBDA_DIST,
) -> tuple[GaussianParams, dict[str, Any], dict[str, Any]]:
    """Run the DTU capture eval. Returns ``(final_params, eval_metrics, report)``.

    ``representation`` selects the L3 arm: ``"3dgs"`` (raw baseline via
    :func:`astel_gpu.smoke_refit.optimize`) or ``"2dgs"`` (surface-aligned via
    :func:`astel_gpu.l3_refine.optimize_2dgs`). Both share an identical init
    cloud, DTU ObsMask/Plane geometry protocol, and held-out PSNR split, so the
    only difference between arms is the representation + losses.
    """
    if representation not in ("3dgs", "2dgs"):
        raise ValueError(
            f"representation must be '3dgs' or '2dgs', got {representation!r}"
        )
    import gsplat

    device = torch.device(device_str)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    scan = load_dtu_scan(image_dir, pos_dir, downscale=downscale)
    train_idx, test_idx = split_train_test(scan.viewmats.shape[0], holdout_every)
    train_inputs = _render_inputs(scan, train_idx, device)
    test_inputs = _render_inputs(scan, test_idx, device)
    train_targets = torch.tensor(
        scan.images[train_idx], dtype=torch.float32, device=device
    )
    test_targets = torch.tensor(
        scan.images[test_idx], dtype=torch.float32, device=device
    )

    half_extent = estimate_object_half_extent(scan)
    init = build_init_cloud(n_gaussians, scan.object_center, half_extent, device, seed)

    start = time.perf_counter()
    if representation == "2dgs":
        final_params, init_psnr_db = optimize_2dgs(
            init,
            train_targets,
            train_inputs,
            iters=iters,
            spatial_lr_scale=half_extent,
            lambda_normal=lambda_normal,
            lambda_dist=lambda_dist,
        )
    else:
        final_params, init_psnr_db = optimize(
            init, train_targets, train_inputs, iters=iters, spatial_lr_scale=half_extent
        )
    if device.type == "cuda":
        torch.cuda.synchronize()
    wall_time_s = time.perf_counter() - start

    with torch.no_grad():
        if representation == "2dgs":
            test_imgs = render_2dgs_colors(final_params, test_inputs)
        else:
            test_imgs = render_views(final_params, test_inputs)
        test_psnr_db = psnr(test_imgs, test_targets)

        gt_np = load_ply_points(gt_ply)
        gt_all = torch.tensor(gt_np, dtype=torch.float32, device=device)
        obsmask = load_obsmask(obsmask_path)
        plane = load_plane(plane_path)

        # Reconstruction = opacity-filtered gaussians (the NN target for both).
        contributing = final_params.opacities > CONTRIBUTING_OPACITY_THRESHOLD
        data_all = final_params.means[contributing]
        if data_all.shape[0] == 0:
            raise RuntimeError("no surface gaussians; fit failed")
        data_np = data_all.detach().cpu().numpy()

        # DTU protocol: filter the QUERY side; NN target is the full other cloud.
        # accuracy query = data inside the ObsMask observable volume.
        # completeness query = GT inside the observable OBJECT volume
        # (ObsMask AND above the ground plane). We intersect ObsMask with the
        # plane (DTU's leaderboard completeness uses above-plane only) because
        # Astel reconstructs the OBJECT, not the full scene: scoring coverage of
        # the whole above-plane scene would penalise object-only modelling. The
        # ObsMask BB is the object region by DTU's construction.
        data_in_mask = points_in_obsmask(data_np, obsmask)
        gt_eval = points_above_plane(gt_np, plane) & points_in_obsmask(gt_np, obsmask)
        data_q = data_all[torch.from_numpy(data_in_mask).to(device)]
        gt_q = gt_all[torch.from_numpy(gt_eval).to(device)]
        if data_q.shape[0] == 0 or gt_q.shape[0] == 0:
            raise RuntimeError("empty eval set after ObsMask/Plane filtering")

        accuracy_mm = float(
            nn_distances(data_q, gt_all, 8192).clamp(max=MAX_DIST_MM).mean()
        )
        completeness_mm = float(
            nn_distances(gt_q, data_all, 8192).clamp(max=MAX_DIST_MM).mean()
        )

        fitted_longest_axis_mm = float(
            (data_q.max(dim=0).values - data_q.min(dim=0).values).max()
        )
        gt_longest_axis_mm = float(
            (gt_q.max(dim=0).values - gt_q.min(dim=0).values).max()
        )

    peak_vram_gb = (
        torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else 0.0
    )

    eval_metrics: dict[str, Any] = {
        "origin": "measured",
        "representation": representation,
        "lambda_normal": lambda_normal if representation == "2dgs" else None,
        "lambda_dist": lambda_dist if representation == "2dgs" else None,
        "test_psnr_db": test_psnr_db,
        "init_psnr_db": init_psnr_db,
        "accuracy_mm": accuracy_mm,
        "completeness_mm": completeness_mm,
        "overall_mm": 0.5 * (accuracy_mm + completeness_mm),
        "n_contributing_gaussians": int(contributing.sum()),
        "n_data_in_obsmask": int(data_in_mask.sum()),
        "n_gt_in_object_volume": int(gt_eval.sum()),
        "fitted_longest_axis_mm": fitted_longest_axis_mm,
        "gt_longest_axis_mm": gt_longest_axis_mm,
        "object_center_mm": scan.object_center.tolist(),
        "half_extent_mm": half_extent,
        "n_gt_points_total": int(gt_all.shape[0]),
        "iters": iters,
        "n_gaussians": n_gaussians,
        "n_train_views": len(train_idx),
        "n_test_views": len(test_idx),
        "downscale": downscale,
        "image_wh": [scan.width, scan.height],
        "wall_time_s": wall_time_s,
        "peak_vram_gb": peak_vram_gb,
        "gpu_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
        ),
        "torch_version": torch.__version__,
        "gsplat_version": gsplat.__version__,
    }

    report = build_capture_quality_report(
        count=final_params.count,
        psnr_db=test_psnr_db,
        n_holdout_views=len(test_idx),
        accuracy_mm=accuracy_mm,
        completeness_mm=completeness_mm,
        n_data_eval=int(data_in_mask.sum()),
        n_gt_eval=int(gt_eval.sum()),
        fitted_longest_axis_mm=fitted_longest_axis_mm,
        gt_longest_axis_mm=gt_longest_axis_mm,
        scan_name=Path(image_dir).parent.name,
        representation=representation,
    )
    return final_params, eval_metrics, report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DTU real-world capture geometry eval (ObsMask protocol)."
    )
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--pos-dir", type=Path, required=True)
    parser.add_argument("--gt-ply", type=Path, required=True)
    parser.add_argument("--obsmask", type=Path, required=True)
    parser.add_argument("--plane", type=Path, required=True)
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    parser.add_argument("--n-gaussians", type=int, default=DEFAULT_N_GAUSSIANS)
    parser.add_argument("--downscale", type=int, default=DEFAULT_DOWNSCALE)
    parser.add_argument("--holdout-every", type=int, default=DEFAULT_HOLDOUT_EVERY)
    parser.add_argument("--seed", type=int, default=20260614)
    parser.add_argument("--out", type=Path, default=Path("out_dtu"))
    parser.add_argument(
        "--representation", type=str, default="3dgs", choices=["3dgs", "2dgs"]
    )
    parser.add_argument("--lambda-normal", type=float, default=DEFAULT_LAMBDA_NORMAL)
    parser.add_argument("--lambda-dist", type=float, default=DEFAULT_LAMBDA_DIST)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    final_params, eval_metrics, report = run_capture_eval(
        args.image_dir,
        args.pos_dir,
        args.gt_ply,
        args.obsmask,
        args.plane,
        iters=args.iters,
        n_gaussians=args.n_gaussians,
        downscale=args.downscale,
        holdout_every=args.holdout_every,
        seed=args.seed,
        device_str=args.device,
        representation=args.representation,
        lambda_normal=args.lambda_normal,
        lambda_dist=args.lambda_dist,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    write_gaussian_ply(final_params, args.out / "l3.ply")
    (args.out / "capture-eval-metrics.json").write_text(
        json.dumps(eval_metrics, indent=2)
    )
    (args.out / "quality-report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"eval": eval_metrics, "report": report}, indent=2))


if __name__ == "__main__":
    main()
