"""Text -> reference image: the canonicalized text-to-image conditioning stage.

This is the conditioning stage for the text->3D path (CLAUDE.md §4 "Text"
pipeline): a user prompt is wrapped into a TripoSplat-friendly template (single
object, centered, plain studio background) and rendered with a local diffusers
text-to-image model. The resulting PNG is fed into the EXISTING
image->TripoSplat L2 -> 2DGS L3 path (:func:`astel_gpu.generative.run_l2_to_l3`);
TripoSplat performs its own background removal, so no separate segmentation
stage is needed here.

MODEL CHOICE (license-clean, local, no custom CUDA build):
- **Default** ``stabilityai/stable-diffusion-xl-base-1.0`` -- OPEN access (not
  gated), CreativeML-OpenRAIL++-M (commercial use permitted), ~7 GB, runs
  comfortably on a 24 GB GPU, no Hugging Face login required. Works out of the
  box on Box A.
- **Opt-in upgrade** ``black-forest-labs/FLUX.1-schnell`` -- Apache-2.0 (the
  cleanest license) but HF-GATED (needs a free `hf auth login` + accepting the
  model terms). Select it with ``ASTEL_T2I_MODEL=black-forest-labs/FLUX.1-schnell``.

Both load through :class:`diffusers.AutoPipelineForText2Image`, which selects the
right pipeline from the model config -- so swapping models is purely an env var.

HONESTY: the generated image (and therefore the downstream asset) is not
conditioned on any real object -- ``generated_ratio`` stays ``1.0`` and
``geometric_error``/``scale`` stay ``None`` through the whole chain, exactly as
the image-modality generative path already reports.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .image_qa import ImageQAConfig, ImageScore, score_image_file

#: Default number of candidate images generated per prompt; the critic
#: (:mod:`astel_gpu.image_qa`) picks the best. ``1`` disables best-of-N (single
#: draw, original behaviour). Override via ``ASTEL_T2I_BEST_OF``.
DEFAULT_BEST_OF_N = 4
_BEST_OF_ENV = "ASTEL_T2I_BEST_OF"

#: Appended to every user prompt to push the model toward a single, centered,
#: TripoSplat-friendly product shot: one object, full-frame, neutral
#: background, even lighting -- the conditions TripoSplat's L2 reconstruction
#: was designed for.
_T2I_PROMPT_SUFFIX = (
    "single object, centered, full object in frame, plain neutral studio "
    "background, soft even lighting, product photograph, high detail"
)

#: Non-gated, commercially-licensed default (no HF login needed). Override with
#: the ``ASTEL_T2I_MODEL`` env var (e.g. the Apache FLUX.1-schnell upgrade).
DEFAULT_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

_T2I_MODEL_ENV = "ASTEL_T2I_MODEL"


def active_model_id() -> str:
    """The text-to-image model id in effect (env override or the default)."""
    return os.environ.get(_T2I_MODEL_ENV, DEFAULT_MODEL_ID).strip() or DEFAULT_MODEL_ID


def build_t2i_prompt(user_prompt: str) -> str:
    """Wrap ``user_prompt`` into a single-object, TripoSplat-friendly template.

    Pure and CPU-testable: just string handling. Strips surrounding whitespace
    and raises ``ValueError`` if the prompt is empty after stripping (an empty
    prompt has no subject to render).
    """
    cleaned = user_prompt.strip()
    if not cleaned:
        raise ValueError("build_t2i_prompt: user_prompt must be non-empty")
    return f"{cleaned}, {_T2I_PROMPT_SUFFIX}"


#: Backwards-compatible alias (the stage was FLUX-only before the model became
#: configurable). Callers may use either name.
build_flux_prompt = build_t2i_prompt


def _model_params(model_id: str, *, steps: int | None) -> dict[str, Any]:
    """Per-family generation parameters.

    FLUX.1-schnell is a guidance-distilled few-step model (guidance 0, ~4 steps,
    needs ``max_sequence_length``); SDXL-class models use classifier-free
    guidance and ~30 steps. Returns the dtype name + call kwargs.
    """
    is_flux = "flux" in model_id.lower()
    if is_flux:
        return {
            "dtype": "bfloat16",
            "guidance_scale": 0.0,
            "num_inference_steps": 4 if steps is None else steps,
            "extra_call_kwargs": {"max_sequence_length": 256},
            "extra_load_kwargs": {},
        }
    return {
        "dtype": "float16",
        "guidance_scale": 5.0,
        "num_inference_steps": 30 if steps is None else steps,
        "extra_call_kwargs": {},
        # SDXL emits an invisible watermark by default (needs an extra dep);
        # disable it so we have no hidden requirement.
        "extra_load_kwargs": {"add_watermarker": False},
    }


def _run_one(
    pipe: Any,
    prompt: str,
    params: dict[str, Any],
    *,
    seed: int,
    size: int,
    out_path: Path,
    device: str,
) -> dict[str, Any]:
    """Run one diffusion sample on an already-loaded ``pipe`` and save the image."""
    import torch

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    result = pipe(
        prompt,
        guidance_scale=params["guidance_scale"],
        num_inference_steps=params["num_inference_steps"],
        height=size,
        width=size,
        generator=torch.Generator("cpu").manual_seed(seed),
        **params["extra_call_kwargs"],
    )
    wall_time_s = time.perf_counter() - start
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.images[0].save(out_path)
    peak_vram_gb = (
        torch.cuda.max_memory_allocated() / (1024**3) if device == "cuda" else 0.0
    )
    return {
        "model": active_model_id(),
        "steps": params["num_inference_steps"],
        "size": size,
        "seed": seed,
        "wall_time_s": wall_time_s,
        "peak_vram_gb": peak_vram_gb,
        "prompt_used": prompt,
        "success": True,
    }


def generate_image(
    prompt: str,
    out_path: Path,
    *,
    seed: int,
    steps: int | None = None,
    size: int = 1024,
    device: str | None = None,
) -> dict[str, Any]:
    """Generate a reference image with the active T2I model and save it.

    The model is :func:`active_model_id` (``ASTEL_T2I_MODEL`` or the non-gated
    SDXL default). Loaded via ``AutoPipelineForText2Image`` with CPU offloading
    enabled (fits alongside the downstream TripoSplat load on a single 24 GB
    GPU). When ``steps`` is ``None`` a model-appropriate default is used
    (FLUX few-step vs. SDXL ~30). After generation the pipeline is deleted and
    CUDA's cache freed so the subsequent TripoSplat L2 stage has the full VRAM
    budget.

    Returns a metrics dict: ``model``, ``steps``, ``size``, ``seed``,
    ``wall_time_s``, ``peak_vram_gb``, ``prompt_used``, ``success``.
    """
    import gc

    import torch
    from diffusers import AutoPipelineForText2Image

    model_id = active_model_id()
    params = _model_params(model_id, steps=steps)
    dtype = getattr(torch, params["dtype"])
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    pipe = AutoPipelineForText2Image.from_pretrained(  # type: ignore[no-untyped-call]
        model_id,
        torch_dtype=dtype,
        use_safetensors=True,
        **params["extra_load_kwargs"],
    )
    pipe.enable_model_cpu_offload()
    try:
        return _run_one(
            pipe, prompt, params, seed=seed, size=size, out_path=out_path,
            device=device,
        )
    finally:
        # Free VRAM before the downstream TripoSplat L2 stage loads its weights.
        del pipe
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()


@dataclass(frozen=True)
class ImageCandidate:
    """One generated candidate image plus its generation metrics."""

    seed: int
    path: Path
    metrics: dict[str, Any]


@dataclass(frozen=True)
class BestOfNResult:
    """Outcome of :func:`generate_image_best_of_n` — the chosen image + scorecards."""

    chosen_seed: int
    chosen_score: ImageScore
    chosen_metrics: dict[str, Any]
    candidates: list[dict[str, Any]]

    def to_sidecar(self) -> dict[str, Any]:
        """JSON-serialisable summary for a ``text2img-candidates.json`` sidecar."""
        return {
            "n": len(self.candidates),
            "chosen_seed": self.chosen_seed,
            "chosen_score": self.chosen_score.to_dict(),
            "chosen_metrics": self.chosen_metrics,
            "candidates": self.candidates,
        }


#: A candidate generator: ``(prompt, out_dir, seeds, steps, size, device) ->
#: list[ImageCandidate]``. Injectable so CPU tests can fake the diffusion stage.
CandidateGenerator = Callable[..., list[ImageCandidate]]


def _generate_candidates_diffusers(
    prompt: str,
    out_dir: Path,
    *,
    seeds: list[int],
    steps: int | None,
    size: int,
    device: str | None,
) -> list[ImageCandidate]:
    """Generate one candidate image per seed, loading the pipeline ONCE.

    Best-of-N would otherwise reload ~7 GB of weights per draw; here a single
    load services every seed, then VRAM is freed before TripoSplat runs.
    """
    import gc

    import torch
    from diffusers import AutoPipelineForText2Image

    model_id = active_model_id()
    params = _model_params(model_id, steps=steps)
    dtype = getattr(torch, params["dtype"])
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir.mkdir(parents=True, exist_ok=True)

    pipe = AutoPipelineForText2Image.from_pretrained(  # type: ignore[no-untyped-call]
        model_id,
        torch_dtype=dtype,
        use_safetensors=True,
        **params["extra_load_kwargs"],
    )
    pipe.enable_model_cpu_offload()
    candidates: list[ImageCandidate] = []
    try:
        for i, seed in enumerate(seeds):
            path = out_dir / f"cand_{i}_seed{seed}.png"
            metrics = _run_one(
                pipe, prompt, params, seed=seed, size=size, out_path=path,
                device=device,
            )
            candidates.append(ImageCandidate(seed=seed, path=path, metrics=metrics))
    finally:
        del pipe
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    return candidates


def active_best_of_n(default: int = DEFAULT_BEST_OF_N) -> int:
    """Best-of-N count in effect (``ASTEL_T2I_BEST_OF`` env override or default)."""
    raw = os.environ.get(_BEST_OF_ENV)
    if raw is None or raw.strip() == "":
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def generate_image_best_of_n(
    prompt: str,
    out_path: Path,
    *,
    base_seed: int,
    n: int | None = None,
    steps: int | None = None,
    size: int = 1024,
    device: str | None = None,
    config: ImageQAConfig | None = None,
    candidate_generator: CandidateGenerator | None = None,
    score_fn: Callable[[Path], ImageScore] = score_image_file,
) -> BestOfNResult:
    """Generate ``n`` candidate images and keep the one the critic ranks highest.

    This is the reliability fix for "same prompt, sometimes wrong": instead of
    betting the whole asset on a single text-to-image draw, we draw ``n`` images
    (seeds ``base_seed .. base_seed + n - 1``), score each with
    :mod:`astel_gpu.image_qa`, copy the best to ``out_path``, and delete the rest.
    The chosen image's score + every candidate's scorecard are returned so the
    caller can persist them (Truth Meter input) and so a fully-rejected batch is
    visible rather than silently shipped.

    ``candidate_generator`` defaults to the real diffusers stage but is injectable
    for CPU tests. ``n`` defaults to :func:`active_best_of_n` (env-configurable).
    """
    n = n if n is not None else active_best_of_n()
    gen = candidate_generator or _generate_candidates_diffusers
    seeds = [base_seed + i for i in range(max(1, n))]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    candidates = gen(
        prompt, out_path.parent, seeds=seeds, steps=steps, size=size, device=device
    )
    if not candidates:
        raise RuntimeError("generate_image_best_of_n: generator produced no images")

    scored = [(c, score_fn(c.path)) for c in candidates]
    best_candidate, best_score = max(scored, key=lambda cs: cs[1].overall)

    shutil.copyfile(best_candidate.path, out_path)
    # The chosen image now lives at out_path; drop the candidate temp files.
    for cand, _ in scored:
        with contextlib.suppress(OSError):
            cand.path.unlink()

    candidate_dicts = [
        {"seed": c.seed, "score": s.to_dict(), "metrics": c.metrics}
        for c, s in scored
    ]
    return BestOfNResult(
        chosen_seed=best_candidate.seed,
        chosen_score=best_score,
        chosen_metrics=best_candidate.metrics,
        candidates=candidate_dicts,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Inference steps (default: model-appropriate — FLUX 4 / SDXL 30).",
    )
    parser.add_argument("--size", type=int, default=1024)
    args = parser.parse_args()

    t2i_prompt = build_t2i_prompt(args.prompt)
    metrics = generate_image(
        t2i_prompt, args.out, seed=args.seed, steps=args.steps, size=args.size
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
