"""Production L2 wrapper around the vendored TripoSplat pipeline.

Graduates ``triposplat_spike.py`` (M3 step 2) into a typed module and fixes
the inf-opacity defect in upstream's ``Gaussian.save_ply``: upstream writes
``_inverse_opacity_activation(get_opacity)`` = ``log(x / (1 - x))``, which is
``inf`` whenever ``get_opacity`` is exactly ``1.0`` (fp16 saturation hits this
for a non-trivial fraction of gaussians).

We avoid that field entirely. ``Gaussian._get_ply_data(transform)`` already
applies upstream's correct coordinate transform to ``xyz``, ``f_dc``,
``scale_log`` (log-scale) and ``rotation`` (wxyz quats) — we take those
unchanged. For opacity we instead take the SIGMOID-ACTIVATED
``gaussian.get_opacity`` (in ``[0, 1]``), clamp it to
``[OPACITY_EPS, 1 - OPACITY_EPS]``, and recompute the logit ourselves, exactly
as ``astel_gpu.export.to_splat_cloud`` does for our other L-stages. The result
is an :class:`astel_splat_io.cloud.SplatCloud` with finite opacity logits for
every splat.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from astel_splat_io.cloud import SplatCloud
from astel_splat_io.ply import write_ply
from numpy.typing import ArrayLike, NDArray

OPACITY_EPS = 1e-6

# Matches upstream's ``Gaussian._DEFAULT_TRANSFORM``: our PLY orientation
# should match what upstream's own exporters intend.
TRIPOSPLAT_DEFAULT_TRANSFORM: list[list[float]] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]

_GPU_ROOT = Path(__file__).resolve().parents[2]
_TRIPOSPLAT_ROOT = _GPU_ROOT / "external" / "TripoSplat"
_MODELS_ROOT = _GPU_ROOT / "models" / "triposplat"
_DEFAULT_IMAGE = (
    _TRIPOSPLAT_ROOT / "static" / "example_inputs" / "building_stone_house.webp"
)
_OUT_DIR = _GPU_ROOT / "out_triposplat"


def splat_cloud_from_fields(
    *,
    xyz: ArrayLike,
    f_dc: ArrayLike,
    opacity_activated: ArrayLike,
    log_scales: ArrayLike,
    quats: ArrayLike,
) -> SplatCloud:
    """Build a :class:`SplatCloud` from raw per-splat fields.

    Pure, GPU-free, weights-free. ``opacity_activated`` is the
    sigmoid-activated opacity in ``[0, 1]`` (e.g. ``Gaussian.get_opacity``),
    NOT a logit. It is clamped to ``[OPACITY_EPS, 1 - OPACITY_EPS]`` and
    converted to a logit here, which is the fix for upstream's inf-opacity
    defect.
    """
    positions = np.asarray(xyz, dtype=np.float32)
    n = positions.shape[0]

    colors_dc = np.asarray(f_dc, dtype=np.float32).reshape(n, 3)
    scales = np.asarray(log_scales, dtype=np.float32).reshape(n, 3)
    rotations = np.asarray(quats, dtype=np.float32).reshape(n, 4)

    alpha = np.asarray(opacity_activated, dtype=np.float32).reshape(n)
    alpha = np.clip(alpha, OPACITY_EPS, 1.0 - OPACITY_EPS)
    opacity_logit = np.log(alpha / (1.0 - alpha)).astype(np.float32)

    return SplatCloud(
        positions=positions,
        colors_dc=colors_dc,
        opacity=opacity_logit,
        log_scales=scales,
        quats=rotations,
    )


def gaussian_to_splat_cloud(
    gaussian: Any, transform: list[list[float]] = TRIPOSPLAT_DEFAULT_TRANSFORM
) -> SplatCloud:
    """Convert a vendored ``triposplat.Gaussian`` into a :class:`SplatCloud`.

    Uses ``gaussian._get_ply_data(transform)`` for ``xyz``, ``f_dc``,
    ``scale_log`` and ``rotation`` (upstream's transform is applied there),
    but ignores the ``opacity_logit`` field it returns (inf-prone) in favor of
    ``gaussian.get_opacity`` (activated, in ``[0, 1]``), routed through
    :func:`splat_cloud_from_fields`'s clamp.
    """
    xyz, _normals, f_dc, _unsafe_logit, scale_log, rotation = gaussian._get_ply_data(
        transform
    )
    opacity_activated: NDArray[np.float32] = (
        gaussian.get_opacity.detach().cpu().numpy()
    )
    return splat_cloud_from_fields(
        xyz=xyz,
        f_dc=f_dc,
        opacity_activated=opacity_activated,
        log_scales=scale_log,
        quats=rotation,
    )


def load_pipeline(device: str) -> Any:
    """Construct the vendored ``TripoSplatPipeline`` with our local checkpoints."""
    sys.path.insert(0, str(_TRIPOSPLAT_ROOT))
    from triposplat import TripoSplatPipeline  # noqa: PLC0415 (vendored, path-injected)

    diffusion = _MODELS_ROOT / "diffusion_models"
    vae = _MODELS_ROOT / "vae"
    return TripoSplatPipeline(
        ckpt_path=str(diffusion / "triposplat_fp16.safetensors"),
        decoder_path=str(vae / "triposplat_vae_decoder_fp16.safetensors"),
        dinov3_path=str(_MODELS_ROOT / "clip_vision" / "dino_v3_vit_h.safetensors"),
        flux2_vae_encoder_path=str(vae / "flux2-vae.safetensors"),
        rmbg_path=str(_MODELS_ROOT / "background_removal" / "birefnet.safetensors"),
        device=device,
    )


@dataclass
class L2Result:
    """Output of :func:`run_l2`: the converted cloud plus run metrics."""

    cloud: SplatCloud
    metrics: dict[str, Any]


def run_l2(
    image_path: str | Path,
    *,
    num_gaussians: int = 65536,
    steps: int = 20,
    seed: int = 0,
    device: str | None = None,
) -> L2Result:
    """Run the TripoSplat L2 pipeline on a single image and convert the result."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = load_pipeline(device)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    gaussian, _prepared = pipe.run(
        str(image_path),
        seed=seed,
        steps=steps,
        num_gaussians=num_gaussians,
        show_progress=True,
    )
    wall_time_s = time.perf_counter() - start

    cloud = gaussian_to_splat_cloud(gaussian)

    peak_vram_gb = (
        torch.cuda.max_memory_allocated() / (1024**3) if device == "cuda" else 0.0
    )

    metrics: dict[str, Any] = {
        "gaussian_count": cloud.count,
        "wall_time_s": wall_time_s,
        "peak_vram_gb": peak_vram_gb,
        "n_nonfinite_opacity_logit": int(np.sum(~np.isfinite(cloud.opacity))),
        "n_nonfinite_xyz": int(np.sum(~np.isfinite(cloud.positions))),
        "steps": steps,
        "num_gaussians": num_gaussians,
        "seed": seed,
        "device": device,
        "image_used": str(image_path),
        "success": True,
    }

    return L2Result(cloud=cloud, metrics=metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=_DEFAULT_IMAGE)
    parser.add_argument("--num-gaussians", type=int, default=65536)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=_OUT_DIR)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    result = run_l2(
        args.image,
        num_gaussians=args.num_gaussians,
        steps=args.steps,
        seed=args.seed,
    )

    write_ply(result.cloud, args.out / "l2.ply")

    metrics_path = args.out / "l2-metrics.json"
    metrics_path.write_text(json.dumps(result.metrics, indent=2))
    print(json.dumps(result.metrics, indent=2))


if __name__ == "__main__":
    main()
