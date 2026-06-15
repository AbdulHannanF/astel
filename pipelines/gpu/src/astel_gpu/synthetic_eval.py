"""Synthetic controlled-ground-truth geometry eval.

Produces the first REAL measured ``geometric_error`` (Chamfer, mm) and
``scale`` numbers for the Truth Meter, using a synthetic scene where ground
truth is KNOWN by construction (see :mod:`astel_gpu.synthetic`).

Pipeline:

1. Build the known synthetic target cloud (sphere shell, longest axis exactly
   0.20 m by construction) -- this is also the L1 ground-truth reference.
2. Render orbit views from known camera poses (reusing
   :mod:`astel_gpu.cameras`).
3. Refit a fresh random gaussian cloud to match those renders (reusing the
   optimization loop from :mod:`astel_gpu.smoke_refit`).
4. Compute PSNR (self-consistency, as in the smoke test) AND Chamfer distance
   (in mm) between the refit cloud's means and the known ground-truth points.
5. Write ``l3.ply``, ``synthetic-eval-metrics.json``, and a
   ``quality-report.json`` (schema ``astel.quality-report/v0``) with REAL
   ``geometric_error`` and ``scale`` fields.

HONESTY NOTE: this is a CONTROLLED SYNTHETIC scene. It validates the Chamfer
and scale measurement machinery, and measures the refit's geometric fidelity
against a KNOWN target -- it is NOT a real-world capture accuracy benchmark
(that requires the COLMAP/MapAnything real-capture path on real photos/video).

The headline ``geometric_error.chamfer_mm_vs_l1`` is measured over
*surface-defining* gaussians (opacity above
:data:`CONTRIBUTING_OPACITY_THRESHOLD`). Raw 3DGS means are not surface-pinned
by an image loss alone: near-transparent stray gaussians remain near the
random-init volume and inflate a naive all-means Chamfer. Both numbers are
reported (raw as ``chamfer_raw_all_means_mm``); the gap empirically motivates
the surface-aligned L3 representation (2DGS/SuGaR) decision.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from .cameras import build_camera_rig
from .export import psnr, write_gaussian_ply
from .gaussians import build_random_init_cloud
from .metrics import chamfer_distance, meters_to_millimeters
from .smoke_refit import (
    DEFAULT_IMAGE_SIZE,
    DEFAULT_ITERS,
    DEFAULT_N_VIEWS,
    RenderInputs,
    optimize,
    render_views,
)
from .synthetic import (
    SYNTHETIC_LONGEST_AXIS_M,
    build_ground_truth_points,
    build_synthetic_target_cloud,
)

DEFAULT_N_GAUSSIANS = 4_000
SYNTHETIC_PSNR_THRESHOLD_DB = 20.0

#: Opacity above which a gaussian is treated as surface-defining (contributing)
#: for the headline geometric-error measurement. Stray near-transparent
#: gaussians left near the random-init volume fall below this and are excluded
#: from the headline Chamfer (but are still reported in
#: ``chamfer_raw_all_means_mm``).
CONTRIBUTING_OPACITY_THRESHOLD = 0.5

#: Camera orbit radius (m): ~2.5x the object's longest axis so the 0.20 m
#: object comfortably fills the frame. The smoke test's default rig radius
#: (3.0 world units) is tuned for its ~unit-scale torus and would leave this
#: metric-scale object a tiny speck in-frame, making the refit under-constrained
#: and the geometric error a framing artifact rather than a fidelity measure.
_CAMERA_RADIUS_M = SYNTHETIC_LONGEST_AXIS_M * 2.5

#: Init-cloud half-extent (m): start gaussians in a box snug around the object
#: rather than the smoke default (1.5 -> ~15x the object), which strands most
#: gaussians far from the surface where no view ever constrains them.
_INIT_SPREAD_M = SYNTHETIC_LONGEST_AXIS_M


def build_synthetic_quality_report(
    *,
    count: int,
    psnr_db: float,
    chamfer_mm: dict[str, float],
    chamfer_filtered_mm: dict[str, float] | None,
    n_contributing: int,
    opacity_threshold: float,
    n_views: int,
) -> dict[str, Any]:
    """Build the ``astel.quality-report/v0`` dict for the synthetic eval.

    Unlike :func:`astel_gpu.produce.build_quality_report`, this report
    contains REAL measured ``geometric_error`` and ``scale`` numbers, because
    the scene's ground truth is known by construction. It is a SEPARATE eval
    tool and does not change the API's GPU producer path
    (``astel_gpu.produce``), whose ``geometric_error``/``scale`` remain
    honestly ``None``.

    The headline Chamfer is computed over surface-defining gaussians
    (``chamfer_filtered_mm``) when any exist; the raw all-means Chamfer is
    always reported alongside as ``chamfer_raw_all_means_mm``.
    """
    headline = chamfer_filtered_mm if chamfer_filtered_mm is not None else chamfer_mm
    method = (
        "synthetic-gt-chamfer-opacity-filtered"
        if chamfer_filtered_mm is not None
        else "synthetic-gt-chamfer-raw-all-means"
    )
    return {
        "schema": "astel.quality-report/v0",
        "origin": "measured",
        "modality": "synthetic-eval",
        "splats": count,
        "geometric_error": {
            "chamfer_mm_vs_l1": headline["symmetric"],
            "chamfer_a_to_b_mm": headline["a_to_b"],
            "chamfer_b_to_a_mm": headline["b_to_a"],
            "chamfer_raw_all_means_mm": chamfer_mm["symmetric"],
            "n_contributing_gaussians": n_contributing,
            "opacity_threshold": opacity_threshold,
            "method": method,
            "reason": (
                "Chamfer measured against a KNOWN synthetic ground-truth "
                "point cloud (a deterministic sphere shell), not an estimate. "
                "The headline value covers surface-defining gaussians only "
                f"(opacity > {opacity_threshold}); chamfer_raw_all_means_mm "
                "additionally includes near-transparent stray splats left "
                "near the random-init volume."
            ),
        },
        "fidelity": {
            "psnr_db": psnr_db,
            "ssim": None,
            "lpips": None,
            "n_holdout_views": n_views,
        },
        "scale": {
            "longest_axis_m": SYNTHETIC_LONGEST_AXIS_M,
            "confidence": 1.0,
            "method": "synthetic-known",
            "reason": (
                "Scale is known by construction: the synthetic ground-truth "
                "sphere shell was built with a longest axis (diameter) of "
                f"exactly {SYNTHETIC_LONGEST_AXIS_M} m. This is not an "
                "estimate."
            ),
        },
        "provenance": {"measured_ratio": 1.0, "generated_ratio": 0.0},
        "caveats": [
            "This is a CONTROLLED SYNTHETIC scene with ground truth known by "
            "construction (a deterministic sphere-shell point cloud at a "
            "fixed 0.20 m scale). It validates the Chamfer-distance and "
            "scale measurement machinery, and measures the refit's "
            "geometric fidelity against that known target.",
            "It is NOT a real-world capture accuracy benchmark. Real-world "
            "geometric accuracy requires the COLMAP/MapAnything "
            "real-capture path (M2) on real photos or video.",
            "geometric_error.chamfer_mm_vs_l1 covers surface-defining "
            f"gaussians (opacity > {opacity_threshold}); "
            "chamfer_raw_all_means_mm over ALL means is much larger because "
            "raw 3DGS means are not surface-pinned by image loss alone -- "
            "near-transparent stray gaussians remain near the random-init "
            "volume. This empirically motivates the surface-aligned L3 "
            "representation (2DGS/SuGaR) decision.",
            "fidelity.psnr_db is a render-then-refit self-consistency PSNR "
            "(gsplat renders the target, gsplat refits a fresh cloud to "
            "match it), as in the smoke test.",
        ],
    }


def run_synthetic_eval(
    iters: int = DEFAULT_ITERS,
    n_gaussians: int = DEFAULT_N_GAUSSIANS,
    n_views: int = DEFAULT_N_VIEWS,
    image_size: int = DEFAULT_IMAGE_SIZE,
    seed: int = 20260613,
    device_str: str = "cuda",
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Run the synthetic ground-truth eval.

    Returns ``(final_params, eval_metrics, quality_report)``.
    """
    import gsplat

    device = torch.device(device_str)
    torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None

    target = build_synthetic_target_cloud(n_gaussians, seed=seed, device=device)
    init = build_random_init_cloud(
        n_gaussians, seed=seed + 1, device=device, spread=_INIT_SPREAD_M
    )
    gt_points = build_ground_truth_points(n_gaussians, seed=seed).to(device=device)

    viewmats, ks = build_camera_rig(n_views, image_size, radius=_CAMERA_RADIUS_M)
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

        chamfer_mm = meters_to_millimeters(
            chamfer_distance(final_params.means, gt_points)
        )

        contributing_mask = final_params.opacities > CONTRIBUTING_OPACITY_THRESHOLD
        n_contributing = int(contributing_mask.sum().item())
        chamfer_filtered_mm: dict[str, float] | None
        if n_contributing > 0:
            chamfer_filtered_mm = meters_to_millimeters(
                chamfer_distance(final_params.means[contributing_mask], gt_points)
            )
        else:
            chamfer_filtered_mm = None

    peak_vram_gb = (
        torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else 0.0
    )

    eval_metrics: dict[str, Any] = {
        "origin": "measured",
        "final_psnr_db": final_psnr_db,
        "init_psnr_db": init_psnr_db,
        "chamfer_mm": chamfer_mm,
        "chamfer_filtered_mm": chamfer_filtered_mm,
        "n_contributing_gaussians": n_contributing,
        "opacity_threshold": CONTRIBUTING_OPACITY_THRESHOLD,
        "longest_axis_m": SYNTHETIC_LONGEST_AXIS_M,
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
            "Controlled synthetic ground-truth eval: target is a known "
            "sphere-shell point cloud (longest axis 0.20 m by construction). "
            "PSNR is a render-then-refit self-consistency metric; Chamfer is "
            "a REAL measured geometric error between the refit cloud's means "
            "and the known ground-truth points. The headline Chamfer covers "
            "surface-defining gaussians (opacity > "
            f"{CONTRIBUTING_OPACITY_THRESHOLD}); chamfer_mm is the raw "
            "all-means value. NOT a real-world capture accuracy benchmark."
        ),
    }

    quality_report = build_synthetic_quality_report(
        count=final_params.count,
        psnr_db=final_psnr_db,
        chamfer_mm=chamfer_mm,
        chamfer_filtered_mm=chamfer_filtered_mm,
        n_contributing=n_contributing,
        opacity_threshold=CONTRIBUTING_OPACITY_THRESHOLD,
        n_views=n_views,
    )

    return final_params, eval_metrics, quality_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthetic controlled-ground-truth geometry eval."
    )
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

    final_params, eval_metrics, quality_report = run_synthetic_eval(
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

    metrics_path = args.out / "synthetic-eval-metrics.json"
    metrics_path.write_text(json.dumps(eval_metrics, indent=2))

    report_path = args.out / "quality-report.json"
    report_path.write_text(json.dumps(quality_report, indent=2))

    summary = {
        "eval_metrics": eval_metrics,
        "quality_report": quality_report,
        "artifacts": [
            str(ply_path),
            str(metrics_path),
            str(report_path),
        ],
    }
    print(json.dumps(summary, indent=2))

    if eval_metrics["final_psnr_db"] <= SYNTHETIC_PSNR_THRESHOLD_DB:
        raise SystemExit(
            f"Synthetic eval FAILED: final PSNR "
            f"{eval_metrics['final_psnr_db']:.2f} dB <= threshold "
            f"{SYNTHETIC_PSNR_THRESHOLD_DB} dB"
        )


if __name__ == "__main__":
    main()
