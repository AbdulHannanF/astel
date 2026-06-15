"""GPU producer CLI: emit the full ``.astel`` layer-stack from the GPU pipeline.

Two real paths, selected by modality:

* **text** (non-empty ``--prompt``) — the text→3D generative path: prompt →
  canonicalized FLUX.1-schnell text-to-image (:mod:`astel_gpu.text_to_image`) →
  the same image → TripoSplat L2 → 2DGS L3 distillation as the image path. The
  PSNR is held-out self-consistency / distillation fidelity, never accuracy vs.
  reality (a generated object has no scan).

* **image** (``--image``) — the real generative path: single image → TripoSplat
  L2 (native gaussians) → 2DGS L3 distillation (:mod:`astel_gpu.generative`). The
  PSNR here is held-out self-consistency / distillation fidelity, never accuracy
  vs. reality (a generated object has no scan).

* **smoke** (fallback — empty text prompt, or video modality) — render-then-refit
  self-consistency (gsplat renders a target and refits a fresh cloud to it).
  Proves the rasterizer + optimizer work on this GPU; ``fidelity.psnr_db`` is a
  REAL measured number, but geometric error and scale are honestly ``None`` (no
  ground-truth, no metric grounding).

Both paths converge on :func:`astel_gpu.packaging.write_layer_stack`, so the GPU
producer emits the SAME artifact contract as the CPU stub
(``services/api/.../producer.py``): ``l0.ply``, ``l3.ply``, ``l3.spz``,
``l3.sog``, ``package.astel``, ``quality-report.json`` (+ ``l2.ply`` for the
generative path) + a measured-metrics sidecar.

Usage::

    python -m astel_gpu.produce --task-id ID --modality text --prompt "..." --out DIR
    python -m astel_gpu.produce --task-id ID --modality image --image IMG.png --out DIR
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .export import to_splat_cloud
from .generative import run_l2_to_l3
from .packaging import build_package_quality_report, write_layer_stack
from .smoke_refit import (
    DEFAULT_IMAGE_SIZE,
    DEFAULT_N_GAUSSIANS,
    DEFAULT_N_VIEWS,
    run_smoke,
)
from .text_to_image import build_flux_prompt, generate_image


def stable_seed(task_id: str) -> int:
    """Deterministically derive a small positive int seed from ``task_id``."""
    from hashlib import blake2b

    digest = blake2b(task_id.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, byteorder="big")


def build_quality_report(
    *, count: int, modality: str, psnr_db: float, n_views: int
) -> dict[str, Any]:
    """Build the ``astel.quality-report/v0`` dict for the smoke (text) path.

    ``geometric_error`` and ``scale`` are honestly unmeasured: the render-then-
    refit smoke has no ground-truth geometry or metric scale, so those fields are
    ``None`` with an explicit reason. Only ``fidelity.psnr_db`` (a real, measured
    self-consistency render metric) is a number.
    """
    return {
        "schema": "astel.quality-report/v0",
        "origin": "measured",
        "modality": modality,
        "splats": count,
        "geometric_error": {
            "chamfer_mm_vs_l1": None,
            "method": None,
            "reason": (
                "GPU render-then-refit producer has no ground-truth geometry "
                "or L1 reference cloud to compare against; geometric error "
                "arrives with the COLMAP/real-capture path (M2)."
            ),
        },
        "fidelity": {
            "psnr_db": psnr_db,
            "ssim": None,
            "lpips": None,
            "n_holdout_views": n_views,
        },
        "scale": {
            "longest_axis_m": None,
            "confidence": None,
            "method": "estimate",
            "reason": (
                "No metric-scale grounding (SfM scale or VLM size estimate) "
                "is performed by this producer; scale arrives with the "
                "capture/conditioning pipeline."
            ),
        },
        "provenance": {"measured_ratio": 0.0, "generated_ratio": 1.0},
        "caveats": [
            "fidelity.psnr_db is a REAL measured render-then-refit "
            "self-consistency PSNR (gsplat renders the target, gsplat refits "
            "a fresh cloud to match it). It proves the rasterizer + "
            "optimizer work on this GPU; it does NOT prove geometric "
            "accuracy against any real object.",
            "geometric_error and scale are explicitly None (not fabricated): "
            "this producer performs no reconstruction from real capture "
            "data and no metric-scale estimation.",
            f"The {modality} input was NOT used to condition this geometry: "
            "this is a render-then-refit smoke, the fallback for inputs without "
            "a wired generator (e.g. video, or an empty prompt). Text and image "
            "inputs DO have real generative paths today (text -> SDXL/FLUX -> "
            "TripoSplat L2 -> 2DGS L3; image -> TripoSplat L2 -> 2DGS L3).",
        ],
    }


def _produce_from_image(
    task_id: str,
    modality: str,
    prompt: str,
    image: Path,
    out_dir: Path,
    *,
    refine_iters: int,
    origin_note: str,
    extra_artifacts: list[str] | None = None,
) -> dict[str, Any]:
    """Shared core: image → TripoSplat L2 → 2DGS L3 → full layer stack.

    Used by both the image modality (:func:`_produce_generative`) and the text
    modality (:func:`_produce_text_generative`, where ``image`` is the
    FLUX-generated reference frame).
    """
    seed = stable_seed(task_id)
    result = run_l2_to_l3(image, seed=seed, refine_iters=refine_iters)

    l3_cloud = to_splat_cloud(result.l3_params)
    l2_cloud = to_splat_cloud(result.l2_params)
    report = result.report  # already an astel.quality-report/v0 dict
    package_report = build_package_quality_report(
        modality=modality, origin_note=origin_note
    )
    artifacts = write_layer_stack(
        l3_cloud,
        out_dir,
        task_id=task_id,
        modality=modality,
        prompt=prompt,
        seed=seed,
        report_dict=report,
        package_report=package_report,
        l2_cloud=l2_cloud,
    )

    metrics = result.metrics
    (out_dir / "l2l3-metrics.json").write_text(json.dumps(metrics, indent=2))
    artifacts = sorted([*artifacts, "l2l3-metrics.json", *(extra_artifacts or [])])

    return {
        "splats": l3_cloud.count,
        "seed_splats": max(1, l3_cloud.count // 24),
        "artifacts": artifacts,
        "metrics": metrics,
    }


def _produce_generative(
    task_id: str,
    modality: str,
    prompt: str,
    image: Path,
    out_dir: Path,
    *,
    refine_iters: int,
) -> dict[str, Any]:
    """Image → TripoSplat L2 → 2DGS L3 → full layer stack."""
    return _produce_from_image(
        task_id,
        modality,
        prompt,
        image,
        out_dir,
        refine_iters=refine_iters,
        origin_note=(
            "Generated: single image → TripoSplat L2 → 2DGS L3 distillation. "
            "Nothing is measured against reality."
        ),
    )


def _produce_text_generative(
    task_id: str,
    modality: str,
    prompt: str,
    out_dir: Path,
    *,
    refine_iters: int,
) -> dict[str, Any]:
    """Text → FLUX.1-schnell image → TripoSplat L2 → 2DGS L3 → full layer stack."""
    seed = stable_seed(task_id)
    flux_prompt = build_flux_prompt(prompt)
    image_path = out_dir / "text-reference.png"
    out_dir.mkdir(parents=True, exist_ok=True)

    text2img_metrics = generate_image(flux_prompt, image_path, seed=seed)
    (out_dir / "text2img-metrics.json").write_text(
        json.dumps(text2img_metrics, indent=2)
    )

    return _produce_from_image(
        task_id,
        modality,
        prompt,
        image_path,
        out_dir,
        refine_iters=refine_iters,
        origin_note=(
            "Generated from text: prompt → FLUX.1-schnell image → TripoSplat "
            "L2 → 2DGS L3. Nothing measured against reality."
        ),
        extra_artifacts=["text-reference.png", "text2img-metrics.json"],
    )


def _produce_smoke(
    task_id: str, modality: str, prompt: str, out_dir: Path, *, iters: int
) -> dict[str, Any]:
    """Render-then-refit self-consistency smoke → full layer stack."""
    seed = stable_seed(task_id)
    final_params, metrics = run_smoke(
        iters=iters,
        n_gaussians=DEFAULT_N_GAUSSIANS,
        n_views=DEFAULT_N_VIEWS,
        image_size=DEFAULT_IMAGE_SIZE,
        seed=seed,
    )

    l3_cloud = to_splat_cloud(final_params)
    report = build_quality_report(
        count=final_params.count,
        modality=modality,
        psnr_db=metrics["final_psnr_db"],
        n_views=metrics["n_views"],
    )
    package_report = build_package_quality_report(
        modality=modality,
        origin_note=(
            "Render-then-refit self-consistency smoke (no prompt conditioning "
            "yet — text→multiview→L2 is the next generative stage)."
        ),
    )
    artifacts = write_layer_stack(
        l3_cloud,
        out_dir,
        task_id=task_id,
        modality=modality,
        prompt=prompt,
        seed=seed,
        report_dict=report,
        package_report=package_report,
    )

    (out_dir / "smoke-metrics.json").write_text(json.dumps(metrics, indent=2))
    artifacts = sorted([*artifacts, "smoke-metrics.json"])

    return {
        "splats": final_params.count,
        "seed_splats": max(1, final_params.count // 24),
        "artifacts": artifacts,
        "metrics": metrics,
    }


def produce(
    task_id: str,
    modality: str,
    prompt: str,
    out_dir: Path,
    iters: int = 1500,
    image: Path | None = None,
    refine_iters: int = 1500,
) -> dict[str, Any]:
    """Run the GPU pipeline and write the full artifact stack into ``out_dir``.

    Dispatches to the image-generative path when ``modality == "image"`` and a
    readable ``image`` is supplied; to the text-generative path when
    ``modality == "text"`` and a non-empty ``prompt`` is supplied; otherwise the
    render-then-refit smoke path (e.g. empty text prompt, video modality).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if modality == "image" and image is not None and image.is_file():
        return _produce_generative(
            task_id, modality, prompt, image, out_dir, refine_iters=refine_iters
        )
    if modality == "text" and prompt.strip():
        return _produce_text_generative(
            task_id, modality, prompt, out_dir, refine_iters=refine_iters
        )
    return _produce_smoke(task_id, modality, prompt, out_dir, iters=iters)


def main() -> None:
    parser = argparse.ArgumentParser(description="Astel GPU artifact producer.")
    parser.add_argument("--task-id", type=str, required=True)
    parser.add_argument("--modality", type=str, default="text")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--iters", type=int, default=1500)
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Input image for the generative (image) modality.",
    )
    parser.add_argument("--refine-iters", type=int, default=1500)
    args = parser.parse_args()

    result = produce(
        args.task_id,
        args.modality,
        args.prompt,
        args.out,
        args.iters,
        image=args.image,
        refine_iters=args.refine_iters,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
