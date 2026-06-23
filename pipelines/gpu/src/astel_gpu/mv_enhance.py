"""Multi-view target enhancer — the detail source that lets the refine exceed L2.

The Tier-1 finding (verified on Box A): the densified refine
(:func:`astel_gpu.refine.refine_with_densification`) only improves an asset when its
``external_targets`` carry MORE information than the L2 cloud being refined.
Supervising against the cloud's own renders is bounded by — and measurably worse
than — the frozen distillation.

This module manufactures higher-detail targets WITHOUT a new heavy model (TRELLIS.2
is mesh-output + Linux-only; this runs on the already-installed SDXL on native
Windows). For each camera in the train rig:

1. render the L2 cloud (geometry/identity anchor),
2. run SDXL **img2img** (SDEdit) at LOW strength on that render, conditioned on the
   prompt — diffusion adds texture/sharpness while the low strength keeps the result
   anchored to the rendered geometry (so the views stay roughly consistent),
3. re-apply the render's foreground mask so the background stays black and the
   refine never chases hallucinated background.

The densified refine then chases these enhanced, sharper images and grows splats to
match — injecting detail the feed-forward L2 never had. This is the DreamGaussian /
SDEdit lineage.

HONESTY / LIMITS: per-view img2img is only *approximately* multi-view consistent —
high strength would let each view invent incompatible detail and the refine would
average them to mush, so strength is kept low and is the key knob. Self-consistency
PSNR vs the L2 self-renders is NO LONGER the right metric once targets are enhanced
(the targets deliberately diverge from L2); judge by rendered sharpness +
:mod:`astel_gpu.geometry_qa` soundness. A truly view-consistent generator is the
later upgrade behind this same seam.

The orchestration + masking are pure ``torch`` (CPU-tested with an injected
enhancer); all PIL/SDXL lives in :func:`_sdxl_img2img_enhance` (GPU).
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import Any

import torch

#: An enhancer: ``(images, *, prompt, strength, steps, guidance_scale, seed,
#: device) -> images``; ``images`` is ``(V, H, W, 3)`` in ``[0, 1]``. Injectable so
#: CPU tests can fake the diffusion stage.
EnhancerFn = Callable[..., torch.Tensor]

#: Appended to the prompt for the img2img pass — pushes toward crisp material detail
#: rather than re-composition.
_ENHANCE_SUFFIX = "sharp focus, fine surface detail, high resolution, photorealistic"


def build_enhance_prompt(user_prompt: str) -> str:
    """Prompt for the img2img enhancement pass (falls back to a generic detail cue)."""
    cleaned = (user_prompt or "").strip()
    if not cleaned:
        return _ENHANCE_SUFFIX
    return f"{cleaned}, {_ENHANCE_SUFFIX}"


def _box_blur(images: torch.Tensor, kernel: int) -> torch.Tensor:
    """Separable box blur of a ``(V, H, W, 3)`` tensor (pure / CPU-testable)."""
    import torch.nn.functional as f

    x = images.permute(0, 3, 1, 2)
    x = f.avg_pool2d(x, kernel_size=kernel, stride=1, padding=kernel // 2)
    return x.permute(0, 2, 3, 1)


def detail_transfer(
    base: torch.Tensor,
    enhanced: torch.Tensor,
    *,
    gain: float = 1.0,
    blur_kernel: int = 9,
) -> torch.Tensor:
    """Add only the HIGH-FREQUENCY detail of ``enhanced`` onto ``base``.

    ``target = base + gain * (enhanced - lowpass(enhanced))``. The low-frequency
    content (exposure, base colour, silhouette) — exactly where independent per-view
    img2img is INCONSISTENT across views and so collapses the refine into dark mush —
    is taken from the consistent ``base`` render; only the sharp structure (edges,
    numerals, texture) comes from the diffusion enhancement. Pure / CPU-testable.
    """
    high = enhanced - _box_blur(enhanced, blur_kernel)
    return base + gain * high


def foreground_mask(images: torch.Tensor, threshold: float = 0.02) -> torch.Tensor:
    """``(V, H, W, 1)`` float mask of object pixels in black-background renders.

    gsplat renders the object on a black background, so a pixel is foreground where
    its brightest channel exceeds ``threshold``. Pure / CPU-testable.
    """
    bright = images.amax(dim=-1, keepdim=True)
    return (bright > threshold).to(images.dtype)


def apply_black_background(
    images: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Zero out (black) every non-foreground pixel. Pure / CPU-testable."""
    return images * mask


def enhance_views(
    base_images: torch.Tensor,
    *,
    prompt: str,
    strength: float = 0.3,
    steps: int = 30,
    guidance_scale: float = 5.0,
    seed: int = 0,
    device: str | None = None,
    mask_background: bool = True,
    combine: str = "detail",
    detail_gain: float = 1.0,
    enhancer: EnhancerFn | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Enhance rendered orbit views into higher-detail refine targets.

    Returns ``(targets, metrics)``; ``targets`` is ``(V, H, W, 3)`` in ``[0, 1]`` on
    the same rig as ``base_images``. ``strength <= 0`` short-circuits to the (masked)
    base renders — a no-op enhancement, used to A/B the seam without paying for
    diffusion. ``enhancer`` defaults to the real SDXL img2img stage; inject a fake
    for CPU tests.

    ``combine`` controls how the diffusion output is merged with the base render:
    ``"detail"`` (default, robust) transfers only the enhancement's high-frequency
    structure onto the consistent base via :func:`detail_transfer` — this avoids the
    dark/mush collapse that ``"replace"`` (use the raw img2img output) suffers when
    independent per-view enhancements disagree on exposure/colour.
    """
    enhancer = enhancer or _sdxl_img2img_enhance
    mask = foreground_mask(base_images) if mask_background else None
    enhance_prompt = build_enhance_prompt(prompt)

    start = time.perf_counter()
    if strength <= 0.0:
        target = base_images
    else:
        enhanced = enhancer(
            base_images,
            prompt=enhance_prompt,
            strength=strength,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
            device=device,
        )
        if combine == "detail":
            target = detail_transfer(base_images, enhanced, gain=detail_gain)
        elif combine == "replace":
            target = enhanced
        else:
            raise ValueError(f"enhance_views: unknown combine mode {combine!r}")
    if mask is not None:
        target = apply_black_background(target, mask)
    target = target.clamp(0.0, 1.0)

    metrics: dict[str, Any] = {
        "n_views": int(base_images.shape[0]),
        "strength": strength,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "combine": combine,
        "detail_gain": detail_gain,
        "prompt_used": enhance_prompt,
        "masked_background": mask_background,
        "wall_time_s": time.perf_counter() - start,
    }
    return target, metrics


def _sdxl_img2img_enhance(
    images: torch.Tensor,
    *,
    prompt: str,
    strength: float,
    steps: int,
    guidance_scale: float,
    seed: int,
    device: str | None,
) -> torch.Tensor:
    """Real SDXL img2img over a batch of views (GPU). Loads the pipeline ONCE.

    Tensor ``(V, H, W, 3)`` in ``[0, 1]`` -> PIL -> SDXL img2img (same model id as
    :mod:`astel_gpu.text_to_image`) -> tensor on the input device. VRAM is freed
    afterwards so the downstream refine has the full budget.
    """
    import gc

    from diffusers import AutoPipelineForImage2Image
    from PIL import Image

    from .text_to_image import active_model_id

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_id = active_model_id()
    dtype = torch.float16 if dev == "cuda" else torch.float32

    pipe = AutoPipelineForImage2Image.from_pretrained(  # type: ignore[no-untyped-call]
        model_id, torch_dtype=dtype, use_safetensors=True, add_watermarker=False
    )
    pipe.enable_model_cpu_offload()

    pil_views = [
        Image.fromarray(
            (v.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy(), mode="RGB"
        )
        for v in images
    ]
    # Process views in small chunks: decoding all orbit views at once OOMs the VAE
    # at higher resolution / view counts (the float32 VAE upcast is the peak).
    chunk = int(os.environ.get("ASTEL_MV_ENHANCE_CHUNK", "4"))
    generator = torch.Generator("cpu").manual_seed(seed)
    try:
        out_imgs: list[Any] = []
        for i in range(0, len(pil_views), max(1, chunk)):
            batch = pil_views[i : i + max(1, chunk)]
            result = pipe(
                prompt=[prompt] * len(batch),
                image=batch,
                strength=strength,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )
            out_imgs.extend(result.images)
        out = torch.stack(
            [
                torch.from_numpy(_to_float_array(img)).to(images.device)
                for img in out_imgs
            ]
        )
    finally:
        del pipe
        gc.collect()
        if dev == "cuda":
            torch.cuda.empty_cache()
    return out


def _to_float_array(img: Any) -> Any:
    """PIL.Image -> ``(H, W, 3)`` float32 numpy in ``[0, 1]`` (RGB)."""
    import numpy as np

    return np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
