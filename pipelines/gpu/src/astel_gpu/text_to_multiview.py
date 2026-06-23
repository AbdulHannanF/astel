"""Text -> multi-view-consistent images via MV-Adapter (the detail keystone).

MV-Adapter (``huanngzh/MV-Adapter``, Apache-2.0) turns SDXL into a multi-view
generator: one text prompt -> N view-consistent images around the object. Unlike a
single image (TripoSplat's input), these carry the real back/sides + per-view detail
that a from-scratch gaussian reconstruction (:mod:`astel_gpu.mv_reconstruct`) fits
into a geometrically-coherent, intricately-detailed splat.

The heavy diffusion stack — ``diffusers`` + the vendored MV-Adapter at
``pipelines/gpu/external/MV-Adapter`` (patched to make its nvdiffrast *texturing*
import lazy; the t2mv path never needs it — see ``docs/research/12``) — is imported
lazily inside :func:`generate_multiview`, so the pure camera spec
(:class:`MultiViewSpec`, :func:`default_spec`) is CPU-testable without it.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: MV-Adapter's default 6 trained azimuths (degrees); front at 0.
DEFAULT_AZIMUTHS: tuple[int, ...] = (0, 45, 90, 180, 270, 315)
#: Vendored, patched MV-Adapter checkout (gitignored; see scripts/setup-gpu-env).
MV_ADAPTER_DIR = Path(__file__).resolve().parents[2] / "external" / "MV-Adapter"
DEFAULT_BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
DEFAULT_ADAPTER_REPO = "huanngzh/mv-adapter"
DEFAULT_ADAPTER_WEIGHT = "mvadapter_t2mv_sdxl.safetensors"
_DEFAULT_NEGATIVE = "watermark, text, low quality, blurry, deformed, extra objects"


@dataclass(frozen=True)
class MultiViewSpec:
    """The camera ring MV-Adapter renders: azimuths (deg), one elevation, ortho.

    The reconstruction (:mod:`astel_gpu.mv_reconstruct`) builds matching gsplat
    orthographic cameras from exactly these fields, so this is the single source of
    truth tying the generated views to their viewpoints.
    """

    azimuth_deg: tuple[int, ...]
    elevation_deg: float = 0.0
    distance: float = 1.8
    frustum: float = 1.1

    @property
    def num_views(self) -> int:
        return len(self.azimuth_deg)


@dataclass
class MultiViewResult:
    """Generated views + the spec that positions them + provenance."""

    images: list[Any]  # list[PIL.Image.Image]
    spec: MultiViewSpec
    prompt: str
    seed: int
    wall_time_s: float


def default_spec(num_views: int = 6) -> MultiViewSpec:
    """Camera spec for ``num_views`` views (MV-Adapter's 6 trained azimuths, else even).

    Pure / CPU-testable. The 6-view default uses MV-Adapter's trained azimuths
    (best consistency); other counts fall back to evenly-spaced azimuths.
    """
    if num_views == len(DEFAULT_AZIMUTHS):
        return MultiViewSpec(DEFAULT_AZIMUTHS)
    step = 360.0 / max(1, num_views)
    return MultiViewSpec(tuple(round(i * step) for i in range(num_views)))


def _ensure_mv_adapter_on_path() -> None:
    if not MV_ADAPTER_DIR.is_dir():
        raise RuntimeError(
            f"MV-Adapter not found at {MV_ADAPTER_DIR}. Clone "
            "https://github.com/huanngzh/MV-Adapter there and apply the "
            "nvdiffrast-lazy patch (docs/research/12)."
        )
    p = str(MV_ADAPTER_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)


def generate_multiview(
    prompt: str,
    *,
    spec: MultiViewSpec | None = None,
    height: int = 768,
    width: int = 768,
    steps: int = 40,
    guidance_scale: float = 7.0,
    seed: int = 0,
    device: str = "cuda",
    base_model: str = DEFAULT_BASE_MODEL,
    adapter_repo: str = DEFAULT_ADAPTER_REPO,
    adapter_weight: str = DEFAULT_ADAPTER_WEIGHT,
    negative_prompt: str = _DEFAULT_NEGATIVE,
) -> MultiViewResult:
    """Prompt -> N view-consistent images (GPU; MV-Adapter SDXL t2mv).

    Loads SDXL + the MV-Adapter t2mv adapter, builds the orthographic camera control
    from ``spec``, generates ``spec.num_views`` consistent views, and frees the
    pipeline VRAM before returning so a downstream reconstruction has the full budget.
    """
    import torch  # noqa: PLC0415 (GPU-heavy; keep the pure seam importable on CPU)

    _ensure_mv_adapter_on_path()
    from diffusers import DDPMScheduler  # noqa: PLC0415
    from mvadapter.pipelines.pipeline_mvadapter_t2mv_sdxl import (  # type: ignore[import-not-found]  # noqa: PLC0415,E501
        MVAdapterT2MVSDXLPipeline,
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

    spec = spec or default_spec()
    n = spec.num_views
    dtype = torch.float16 if device.startswith("cuda") else torch.float32

    start = time.perf_counter()
    pipe = MVAdapterT2MVSDXLPipeline.from_pretrained(base_model)
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

    images = pipe(
        prompt, height=height, width=width, num_inference_steps=steps,
        guidance_scale=guidance_scale, num_images_per_prompt=n,
        control_image=control_images, control_conditioning_scale=1.0,
        negative_prompt=negative_prompt, max_sequence_length=214,
        generator=torch.Generator(device).manual_seed(seed),
    ).images

    del pipe
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return MultiViewResult(
        images=list(images), spec=spec, prompt=prompt, seed=seed,
        wall_time_s=time.perf_counter() - start,
    )
