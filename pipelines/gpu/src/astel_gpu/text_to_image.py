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
import json
import os
import time
from pathlib import Path
from typing import Any

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
    num_steps = params["num_inference_steps"]

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    pipe = AutoPipelineForText2Image.from_pretrained(  # type: ignore[no-untyped-call]
        model_id,
        torch_dtype=dtype,
        use_safetensors=True,
        **params["extra_load_kwargs"],
    )
    pipe.enable_model_cpu_offload()

    start = time.perf_counter()
    result = pipe(
        prompt,
        guidance_scale=params["guidance_scale"],
        num_inference_steps=num_steps,
        height=size,
        width=size,
        generator=torch.Generator("cpu").manual_seed(seed),
        **params["extra_call_kwargs"],
    )
    wall_time_s = time.perf_counter() - start

    image = result.images[0]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)

    peak_vram_gb = (
        torch.cuda.max_memory_allocated() / (1024**3) if device == "cuda" else 0.0
    )

    # Free VRAM before the downstream TripoSplat L2 stage loads its weights.
    del pipe
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return {
        "model": model_id,
        "steps": num_steps,
        "size": size,
        "seed": seed,
        "wall_time_s": wall_time_s,
        "peak_vram_gb": peak_vram_gb,
        "prompt_used": prompt,
        "success": True,
    }


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
