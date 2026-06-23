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
import os
from pathlib import Path
from typing import Any

from .export import to_splat_cloud
from .generative import DEFAULT_REFINE_ITERS, L2L3Result, run_l2_to_l3
from .packaging import build_package_quality_report, write_layer_stack
from .smoke_refit import (
    DEFAULT_IMAGE_SIZE,
    DEFAULT_N_GAUSSIANS,
    DEFAULT_N_VIEWS,
    run_smoke,
)
from .text_to_image import build_flux_prompt, generate_image_best_of_n

#: Optional best-of-K asset re-roll: run the L2->L3 stage K times with different
#: TripoSplat noise seeds and keep the soundest cloud (by the geometry critic),
#: stopping early once one passes. ``1`` (default) disables it — each extra roll
#: costs a full TripoSplat pass, so this is an opt-in quality/cost trade for the
#: cases where a single draw collapses. Override via ``ASTEL_ASSET_BEST_OF``.
DEFAULT_ASSET_BEST_OF = 1
_ASSET_BEST_OF_ENV = "ASTEL_ASSET_BEST_OF"


def active_asset_best_of(default: int = DEFAULT_ASSET_BEST_OF) -> int:
    """Asset best-of-K count in effect (``ASTEL_ASSET_BEST_OF`` env or default)."""
    raw = os.environ.get(_ASSET_BEST_OF_ENV)
    if raw is None or raw.strip() == "":
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def _run_l2l3_best_of_k(
    image: Path,
    *,
    base_seed: int,
    k: int,
    refine_iters: int,
    image_qa: dict[str, Any] | None,
    prompt: str = "",
) -> L2L3Result:
    """Run L2->L3 up to ``k`` times (seeds ``base_seed + i``); keep the soundest.

    Scores each result by its geometry critic (``metrics['geometry_qa']``), keeps
    the highest ``overall``, and stops early the moment one is ``accept``. ``k==1``
    is a single pass (no added cost). The kept result records the re-roll count
    and which roll won under ``metrics['asset_reroll']``.
    """
    best: L2L3Result | None = None
    best_overall = -1.0
    rolls_run = 0
    for i in range(max(1, k)):
        rolls_run += 1
        result = run_l2_to_l3(
            image, seed=base_seed + i, refine_iters=refine_iters, image_qa=image_qa,
            prompt=prompt,
        )
        qa = result.metrics.get("geometry_qa", {})
        overall = float(qa.get("overall", 0.0))
        if overall > best_overall:
            best_overall, best = overall, result
        if qa.get("accept", False):
            break
    assert best is not None  # the loop runs at least once
    best.metrics["asset_reroll"] = {
        "k": k,
        "rolls_run": rolls_run,
        "winning_overall": best_overall,
        "base_seed": base_seed,
    }
    return best


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
        # Render-then-refit self-consistency, NOT a reconstruction of any real
        # object: origin is "generated" (CLAUDE.md §1.3/§8.4). The Truth Meter
        # reads this field for its provenance pill; "measured" would falsely claim
        # ground-truth capture. The fidelity.psnr_db below is still a genuinely
        # measured self-consistency number -- that honesty lives in its own note.
        "origin": "generated",
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


def _stage_l6_json(l6_json_path: Path | None, out_dir: Path) -> None:
    """Copy a pre-computed ``l6.json`` into ``out_dir`` so packaging can bind L6.

    The API physics-material stage writes ``l6.json`` (per-region material /
    density / articulation) to the artifact store and passes its path here via
    ``--l6-json``; :func:`astel_gpu.packaging.write_layer_stack` then reads
    ``out_dir/l6.json`` to compute the L6<->L5 mass join and bind the L6 layer.
    No-op when absent (offline default, image modality, or a non-text path).
    """
    if l6_json_path is None or not l6_json_path.is_file():
        return
    (out_dir / "l6.json").write_bytes(l6_json_path.read_bytes())


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
    longest_axis_m: float | None = None,
    l6_json_path: Path | None = None,
    image_qa: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Shared core: image → TripoSplat L2 → 2DGS L3 → full layer stack.

    Used by both the image modality (:func:`_produce_generative`) and the text
    modality (:func:`_produce_text_generative`, where ``image`` is the
    FLUX-generated reference frame). ``longest_axis_m`` (when supplied) is the
    Generation Spec's metric size estimate; it grounds the L5/L6 mass + the
    package's ``meters_per_unit`` (CLAUDE.md §3 L1 metric scale). ``l6_json_path``
    (when supplied) is staged into ``out_dir`` so the L6 layer binds into the
    ``.astel`` package.
    """
    seed = stable_seed(task_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    _stage_l6_json(l6_json_path, out_dir)
    result = _run_l2l3_best_of_k(
        image,
        base_seed=seed,
        k=active_asset_best_of(),
        refine_iters=refine_iters,
        image_qa=image_qa,
        prompt=prompt,
    )

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
        longest_axis_m=longest_axis_m,
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
    longest_axis_m: float | None = None,
    l6_json_path: Path | None = None,
) -> dict[str, Any]:
    """Image → TripoSplat L2 → 2DGS L3 → full layer stack."""
    return _produce_from_image(
        task_id,
        modality,
        prompt,
        image,
        out_dir,
        refine_iters=refine_iters,
        longest_axis_m=longest_axis_m,
        l6_json_path=l6_json_path,
        origin_note=(
            "Generated: single image → TripoSplat L2 → 2DGS L3 surfelization. "
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
    longest_axis_m: float | None = None,
    l6_json_path: Path | None = None,
) -> dict[str, Any]:
    """Text → SDXL/FLUX best-of-N image → TripoSplat L2 → 2DGS L3 → full stack.

    Best-of-N (:func:`generate_image_best_of_n`) draws several candidate images and
    keeps the one the reference-image critic ranks highest, which is the primary
    fix for "same prompt, sometimes wrong": a bad text-to-image draw no longer
    determines the whole asset. The chosen image's scorecard is threaded into the
    quality report (Truth Meter) and every candidate's score is persisted.
    """
    seed = stable_seed(task_id)
    flux_prompt = build_flux_prompt(prompt)
    image_path = out_dir / "text-reference.png"
    out_dir.mkdir(parents=True, exist_ok=True)

    best = generate_image_best_of_n(flux_prompt, image_path, base_seed=seed)
    (out_dir / "text2img-candidates.json").write_text(
        json.dumps(best.to_sidecar(), indent=2)
    )
    # Keep the legacy per-image metrics sidecar (the chosen draw) for consumers
    # that already read it.
    (out_dir / "text2img-metrics.json").write_text(
        json.dumps(best.chosen_metrics, indent=2)
    )

    return _produce_from_image(
        task_id,
        modality,
        prompt,
        image_path,
        out_dir,
        refine_iters=refine_iters,
        longest_axis_m=longest_axis_m,
        l6_json_path=l6_json_path,
        image_qa=best.chosen_score.to_dict(),
        origin_note=(
            "Generated from text: prompt → SDXL/FLUX (best-of-N) image → "
            "TripoSplat L2 → 2DGS L3. Nothing measured against reality."
        ),
        extra_artifacts=[
            "text-reference.png",
            "text2img-metrics.json",
            "text2img-candidates.json",
        ],
    )


def _build_multiview_report(
    *,
    count: int,
    n_views: int,
    psnr_db: float,
    geometry_qa: dict[str, Any] | None,
) -> dict[str, Any]:
    """``astel.quality-report/v0`` for a text→multi-view-reconstruction asset.

    Honest provenance: origin ``generated`` (the views are diffusion-synthesised,
    not photographed), no ground-truth geometry/scale, but multi-view-CONSISTENT
    conditioning (real back/sides, unlike the single-image distillation path).
    """
    report: dict[str, Any] = {
        "schema": "astel.quality-report/v0",
        "origin": "generated",
        "modality": "generative-text/mv-adapter-multiview->3dgs-reconstruction",
        "representation": "3dgs",
        "splats": count,
        "geometric_error": {
            "chamfer_mm_vs_l1": None,
            "method": None,
            "reason": (
                "Generated object reconstructed from MV-Adapter multi-view-diffusion "
                "images (view-consistent but synthetic). There is no real-world "
                "ground-truth scan to compare against, so geometric accuracy vs "
                "reality is undefined here."
            ),
        },
        "fidelity": {
            "psnr_db": psnr_db,
            "ssim": None,
            "lpips": None,
            "n_holdout_views": 0,
            "psnr_note": (
                "Photometric fit to the generated multi-view images (training "
                "views) — how well the splat reproduces the diffusion views, NOT "
                "accuracy versus any real object."
            ),
        },
        "scale": {
            "longest_axis_m": None,
            "confidence": None,
            "method": "estimate",
            "reason": (
                "Generated asset normalised to a unit frame; no metric-scale "
                "grounding is performed here."
            ),
        },
        "provenance": {"measured_ratio": 0.0, "generated_ratio": 1.0},
        "caveats": [
            f"Fully GENERATED asset: text → MV-Adapter {n_views} consistent views → "
            "3DGS multi-view reconstruction. Nothing is measured against reality.",
            "Multi-view-consistent conditioning gives a real back/sides + per-view "
            "detail (unlike the single-image path), but the views are "
            "diffusion-synthesised, not photographed.",
            "geometric_error and scale are explicitly None (not fabricated): no "
            "ground-truth geometry or metric scale exists for a generated object.",
        ],
    }
    if geometry_qa is not None:
        report["qa"] = {"geometry": geometry_qa}
    return report


def _produce_text_multiview(
    task_id: str,
    modality: str,
    prompt: str,
    out_dir: Path,
    *,
    longest_axis_m: float | None = None,
    l6_json_path: Path | None = None,
) -> dict[str, Any]:
    """Text → MV-Adapter multi-view → 3DGS reconstruction → full layer stack.

    Opt-in via ``ASTEL_T2MV=1`` (source res ``ASTEL_T2MV_RES``, view count
    ``ASTEL_T2MV_VIEWS``). Unlike the default single-image path this conditions on
    several view-consistent images, so the back/sides are real and the detail is
    supervised from every side. See :mod:`astel_gpu.text_to_multiview` /
    :mod:`astel_gpu.mv_reconstruct`.
    """
    import torch  # noqa: PLC0415 (gsplat/diffusion-heavy path)

    from .export import psnr  # noqa: PLC0415
    from .generative import normalize_params  # noqa: PLC0415
    from .geometry_qa import score_cloud  # noqa: PLC0415
    from .mv_reconstruct import (  # noqa: PLC0415
        matte_views,
        ortho_cameras,
        reconstruct,
        render_ortho,
    )
    from .text_to_multiview import default_spec, generate_multiview  # noqa: PLC0415

    seed = stable_seed(task_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    _stage_l6_json(l6_json_path, out_dir)
    res = int(os.environ.get("ASTEL_T2MV_RES", "768"))
    spec = default_spec(int(os.environ.get("ASTEL_T2MV_VIEWS", "6")))
    device_str = "cuda" if torch.cuda.is_available() else "cpu"

    mv = generate_multiview(
        prompt, spec=spec, height=res, width=res, seed=seed, device=device_str
    )
    view_names: list[str] = []
    for i, az in enumerate(spec.azimuth_deg):
        name = f"text-multiview-{i:02d}-az{az:03d}.png"
        mv.images[i].save(out_dir / name)
        view_names.append(name)

    dev = torch.device(device_str)
    targets, masks = matte_views(mv.images, device=dev)
    viewmats, ks = ortho_cameras(spec.azimuth_deg, spec.elevation_deg, res, device=dev)
    cloud, recon_metrics = reconstruct(targets, masks, viewmats, ks, res, seed=seed)
    with torch.no_grad():
        fit_psnr = psnr(render_ortho(cloud, viewmats, ks, res)[0], targets)
    geometry_qa = score_cloud(cloud).to_dict()

    cloud_n, _center, _radius = normalize_params(cloud)
    l3_cloud = to_splat_cloud(cloud_n)
    report = _build_multiview_report(
        count=l3_cloud.count, n_views=spec.num_views, psnr_db=fit_psnr,
        geometry_qa=geometry_qa,
    )
    package_report = build_package_quality_report(
        modality=modality,
        origin_note=(
            "Generated: text → MV-Adapter multi-view diffusion → 3DGS multi-view "
            "reconstruction. View-consistent but synthetic; nothing measured."
        ),
    )
    artifacts = write_layer_stack(
        l3_cloud, out_dir, task_id=task_id, modality=modality, prompt=prompt,
        seed=seed, report_dict=report, package_report=package_report,
        longest_axis_m=longest_axis_m,
    )
    metrics = {
        **recon_metrics, "fit_psnr_db": fit_psnr, "source": "mv-adapter",
        "n_views": spec.num_views, "source_res": res,
        "mv_wall_time_s": mv.wall_time_s,
    }
    (out_dir / "mv-recon-metrics.json").write_text(json.dumps(metrics, indent=2))
    artifacts = sorted([*artifacts, "mv-recon-metrics.json", *view_names])
    return {
        "splats": l3_cloud.count,
        "seed_splats": max(1, l3_cloud.count // 24),
        "artifacts": artifacts,
        "metrics": metrics,
    }


def _produce_video(
    task_id: str,
    modality: str,
    prompt: str,
    out_dir: Path,
    *,
    iters: int,
    image: Path | None,
    refine_iters: int,
    longest_axis_m: float | None = None,
    l6_json_path: Path | None = None,
) -> dict[str, Any]:
    """Video modality dispatcher — honest about what it does and does not do.

    If a representative frame (``image``) is supplied and readable, runs the
    REAL static reconstruction path (image → TripoSplat L2 → 2DGS L3) with
    an explicit origin note that this is a static reconstruction from a single
    frame, NOT full 4DGS tracking.  L7 dynamics are NOT produced here — they
    require the GPU deformable-reconstruction stage (future work).

    If no frame is available, falls back to the smoke path with a clear caveat
    that the video was not used to condition the geometry and that dynamics/L7
    were not produced.

    HONESTY CONTRACT (CLAUDE.md §10.4): no L7 deformation is emitted on this
    path regardless of input — the builder only binds L7 when an explicit
    deformation file is passed (Part 3 of the wiring), which this path does
    not supply.  The capability exists (``write_dynamics_layer`` + tests) but
    real per-frame tracking is a dedicated GPU deformable-recon stage; to
    pretend otherwise would be a silent hallucination over real data.
    """
    if image is not None and image.is_file():
        return _produce_from_image(
            task_id,
            modality,
            prompt,
            image,
            out_dir,
            refine_iters=refine_iters,
            longest_axis_m=longest_axis_m,
            l6_json_path=l6_json_path,
            origin_note=(
                "Static reconstruction from a video frame (sharpest available); "
                "L7 dynamics tracking (4DGS) is NOT performed here — it requires "
                "the GPU deformable-recon stage. The asset is a static L3."
            ),
        )

    # No usable frame supplied: smoke fallback with honest caveats.
    return _produce_smoke(task_id, modality, prompt, out_dir, iters=iters)


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
    refine_iters: int = DEFAULT_REFINE_ITERS,
    longest_axis_m: float | None = None,
    l6_json_path: Path | None = None,
) -> dict[str, Any]:
    """Run the GPU pipeline and write the full artifact stack into ``out_dir``.

    Dispatches to the image-generative path when ``modality == "image"`` and a
    readable ``image`` is supplied; to the text-generative path when
    ``modality == "text"`` and a non-empty ``prompt`` is supplied; otherwise the
    render-then-refit smoke path (e.g. empty text prompt, video modality).

    ``longest_axis_m`` (the Generation Spec's metric size estimate) grounds the
    mass + scale on the two generative paths; the smoke path stays ungrounded
    (its geometry is not the object, so a metric scale would be meaningless).
    ``l6_json_path`` (the API physics-material stage's ``l6.json``) is staged into
    ``out_dir`` so the L6 layer binds into the package on the generative paths.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if modality == "image" and image is not None and image.is_file():
        return _produce_generative(
            task_id,
            modality,
            prompt,
            image,
            out_dir,
            refine_iters=refine_iters,
            longest_axis_m=longest_axis_m,
            l6_json_path=l6_json_path,
        )
    if modality == "text" and prompt.strip():
        # Opt-in multi-view path (text → MV-Adapter consistent views → 3DGS recon):
        # real back/sides + intricate detail. Default OFF (single-image distillation
        # stays the zero-risk default) until validated on the eval corpus.
        if os.environ.get("ASTEL_T2MV") == "1":
            return _produce_text_multiview(
                task_id,
                modality,
                prompt,
                out_dir,
                longest_axis_m=longest_axis_m,
                l6_json_path=l6_json_path,
            )
        return _produce_text_generative(
            task_id,
            modality,
            prompt,
            out_dir,
            refine_iters=refine_iters,
            longest_axis_m=longest_axis_m,
            l6_json_path=l6_json_path,
        )
    if modality == "video":
        return _produce_video(
            task_id,
            modality,
            prompt,
            out_dir,
            iters=iters,
            image=image,
            refine_iters=refine_iters,
            longest_axis_m=longest_axis_m,
            l6_json_path=l6_json_path,
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
    parser.add_argument("--refine-iters", type=int, default=DEFAULT_REFINE_ITERS)
    parser.add_argument(
        "--longest-axis-m",
        type=float,
        default=None,
        help=(
            "Metric size estimate (longest axis, metres) from the Generation "
            "Spec; grounds the L5/L6 mass + package scale on the generative paths."
        ),
    )
    parser.add_argument(
        "--l6-json",
        type=Path,
        default=None,
        help=(
            "Path to a pre-computed l6.json (physics-material layer); staged into "
            "--out so the L6 layer binds into the .astel package."
        ),
    )
    args = parser.parse_args()

    result = produce(
        args.task_id,
        args.modality,
        args.prompt,
        args.out,
        args.iters,
        image=args.image,
        refine_iters=args.refine_iters,
        longest_axis_m=args.longest_axis_m,
        l6_json_path=args.l6_json,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
