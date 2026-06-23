"""Image -> multi-view-consistent images via MV-Adapter i2mv (the back-fixing keystone).

The single-image generative default (TripoSplat L2) sees ONE view and hallucinates
the back/sides as low-confidence, low-opacity "glassy" splats — the failure the
founder flagged (2026-06-24). MV-Adapter's i2mv adapter (``huanngzh/mv-adapter``,
``mvadapter_i2mv_sdxl.safetensors``, Apache-2.0) instead turns SDXL into an
*image-conditioned* multi-view generator: one reference image -> N view-consistent
images around the object, faithful to the input's identity. Those views carry a real
back/sides + per-view detail that the densified from-scratch reconstruction
(:mod:`astel_gpu.mv_reconstruct`) fits into a geometrically-coherent splat.

This is the image-modality twin of :mod:`astel_gpu.text_to_multiview` (which does the
same for text prompts) and reuses its pure camera spec (:class:`MultiViewSpec`,
:func:`default_spec`) and result type (:class:`MultiViewResult`). The recipe mirrors
upstream ``external/MV-Adapter/scripts/inference_i2mv_sdxl.py``: reference image
(matted, centred, composited on neutral grey) + plücker camera control, ShiftSNR
DDPM scheduler, i2mv adapter on SDXL.

The heavy diffusion stack is imported lazily inside :func:`generate_multiview_from_image`,
so the pure preprocessing seam (:func:`preprocess_reference_array`) is CPU-testable
without diffusers / a GPU.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .text_to_multiview import (
    DEFAULT_ADAPTER_REPO,
    DEFAULT_BASE_MODEL,
    MultiViewResult,
    MultiViewSpec,
    _ensure_mv_adapter_on_path,
    default_spec,
)

#: i2mv adapter weight in the (shared) ``huanngzh/mv-adapter`` repo.
DEFAULT_I2MV_WEIGHT = "mvadapter_i2mv_sdxl.safetensors"
#: i2mv's lower CFG (upstream default 3.0) — the reference image carries identity, so
#: high guidance over-saturates / fights the conditioning.
DEFAULT_GUIDANCE_SCALE = 3.0
#: Upstream i2mv negative prompt (differs from t2mv's).
_DEFAULT_NEGATIVE = "watermark, ugly, deformed, noisy, blurry, low contrast"
#: Neutral-grey background MV-Adapter's i2mv was trained to condition on.
_BG_GREY = 0.5


def preprocess_reference_array(
    rgba: NDArray[np.uint8], height: int, width: int, *, margin: float = 0.9
) -> NDArray[np.uint8]:
    """Centre + pad a matted RGBA reference onto a neutral-grey ``height×width`` RGB.

    Vendored from upstream ``inference_i2mv_sdxl.preprocess_image``: crop to the alpha
    bounding box, resize the longer side to ``margin * size``, centre-pad, then
    composite over grey (``_BG_GREY``) using the alpha. Pure NumPy — no diffusers /
    GPU — so it is the CPU-testable seam for this module.
    """
    alpha = rgba[..., 3] > 0
    if not alpha.any():  # no foreground detected: fall back to a plain resize
        from PIL import Image  # noqa: PLC0415

        rgb = rgba[..., :3]
        return np.asarray(Image.fromarray(rgb).resize((width, height)), dtype=np.uint8)

    from PIL import Image  # noqa: PLC0415

    ys, xs = np.where(alpha)
    h_img, w_img = alpha.shape
    y0, y1 = max(int(ys.min()) - 1, 0), min(int(ys.max()) + 1, h_img)
    x0, x1 = max(int(xs.min()) - 1, 0), min(int(xs.max()) + 1, w_img)
    crop = rgba[y0:y1, x0:x1]

    ch, cw, _ = crop.shape
    if ch > cw:
        new_w = int(cw * (height * margin) / ch)
        new_h = int(height * margin)
    else:
        new_h = int(ch * (width * margin) / cw)
        new_w = int(width * margin)
    crop = np.asarray(Image.fromarray(crop).resize((new_w, new_h)), dtype=np.uint8)

    canvas = np.zeros((height, width, 4), dtype=np.uint8)
    sh, sw = (height - new_h) // 2, (width - new_w) // 2
    canvas[sh : sh + new_h, sw : sw + new_w] = crop

    f = canvas.astype(np.float32) / 255.0
    a = f[..., 3:4]
    composited = f[..., :3] * a + (1.0 - a) * _BG_GREY
    return (composited * 255.0).clip(0, 255).astype(np.uint8)


def _matte_reference(image: Any, height: int, width: int, *, remove_bg: bool) -> Any:
    """Load/normalise a reference PIL image to the grey-composited RGB i2mv expects.

    With ``remove_bg`` (default) the background is stripped via rembg (the same
    matter :mod:`astel_gpu.mv_reconstruct` uses), then centred/padded. An already-RGBA
    image is preprocessed as-is; a plain RGB with ``remove_bg=False`` is passed through.
    """
    from PIL import Image  # noqa: PLC0415

    if remove_bg:
        from rembg import new_session, remove  # noqa: PLC0415

        rgba = remove(image.convert("RGB"), session=new_session("isnet-general-use"))
        arr = preprocess_reference_array(np.asarray(rgba, dtype=np.uint8), height, width)
        return Image.fromarray(arr, mode="RGB")
    if image.mode == "RGBA":
        arr = preprocess_reference_array(
            np.asarray(image, dtype=np.uint8), height, width
        )
        return Image.fromarray(arr, mode="RGB")
    return image.convert("RGB")


def generate_multiview_from_image(
    image_path: str | Path,
    *,
    prompt: str = "high quality",
    spec: MultiViewSpec | None = None,
    height: int = 768,
    width: int = 768,
    steps: int = 50,
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
    reference_conditioning_scale: float = 1.0,
    seed: int = 0,
    device: str = "cuda",
    base_model: str = DEFAULT_BASE_MODEL,
    adapter_repo: str = DEFAULT_ADAPTER_REPO,
    adapter_weight: str = DEFAULT_I2MV_WEIGHT,
    negative_prompt: str = _DEFAULT_NEGATIVE,
    remove_bg: bool = True,
) -> MultiViewResult:
    """Reference image -> N view-consistent images (GPU; MV-Adapter SDXL i2mv).

    Loads SDXL + the i2mv adapter, mattes + centres the reference image, builds the
    orthographic plücker control from ``spec`` (the SAME camera ring
    :mod:`astel_gpu.mv_reconstruct` reconstructs from), generates ``spec.num_views``
    consistent views conditioned on the reference, and frees the pipeline VRAM before
    returning so the downstream reconstruction gets the full budget.
    """
    import torch  # noqa: PLC0415 (GPU-heavy; keep the pure seam importable on CPU)

    _ensure_mv_adapter_on_path()
    from diffusers import DDPMScheduler  # noqa: PLC0415
    from mvadapter.pipelines.pipeline_mvadapter_i2mv_sdxl import (  # type: ignore[import-not-found]  # noqa: PLC0415,E501
        MVAdapterI2MVSDXLPipeline,
    )
    from mvadapter.schedulers.scheduling_shift_snr import (  # type: ignore[import-not-found]  # noqa: PLC0415,E501
        ShiftSNRScheduler,
    )
    from mvadapter.utils.geometry import (  # type: ignore[import-not-found]  # noqa: PLC0415,E501
        get_plucker_embeds_from_cameras_ortho,
    )
    from mvadapter.utils.mesh_utils import (  # type: ignore[import-not-found]  # noqa: PLC0415,E501
        get_orthogonal_camera,
    )
    from PIL import Image  # noqa: PLC0415

    spec = spec or default_spec()
    n = spec.num_views
    dtype = torch.float16 if device.startswith("cuda") else torch.float32

    start = time.perf_counter()
    pipe = MVAdapterI2MVSDXLPipeline.from_pretrained(base_model)
    pipe.scheduler = ShiftSNRScheduler.from_scheduler(
        pipe.scheduler, shift_mode="interpolated", shift_scale=8.0,
        scheduler_class=DDPMScheduler,
    )
    pipe.init_custom_adapter(num_views=n)
    pipe.load_custom_adapter(adapter_repo, weight_name=adapter_weight)
    pipe.to(device=device, dtype=dtype)
    pipe.cond_encoder.to(device=device, dtype=dtype)
    pipe.enable_vae_slicing()

    cameras = get_orthogonal_camera(
        elevation_deg=[spec.elevation_deg] * n, distance=[spec.distance] * n,
        left=-spec.frustum / 2, right=spec.frustum / 2,
        bottom=-spec.frustum / 2, top=spec.frustum / 2,
        azimuth_deg=[a - 90 for a in spec.azimuth_deg], device=device,
    )
    plucker = get_plucker_embeds_from_cameras_ortho(cameras.c2w, [1.1] * n, width)
    control_images = ((plucker + 1.0) / 2.0).clamp(0, 1)

    reference = _matte_reference(
        Image.open(image_path), height, width, remove_bg=remove_bg
    )

    images = pipe(
        prompt, height=height, width=width, num_inference_steps=steps,
        guidance_scale=guidance_scale, num_images_per_prompt=n,
        control_image=control_images, control_conditioning_scale=1.0,
        reference_image=reference, reference_conditioning_scale=reference_conditioning_scale,
        negative_prompt=negative_prompt,
        generator=torch.Generator(device).manual_seed(seed),
    ).images

    del pipe
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return MultiViewResult(
        images=list(images), spec=spec, prompt=prompt, seed=seed,
        wall_time_s=time.perf_counter() - start,
    )
