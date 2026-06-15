"""GPU+weights-gated test for FLUX.1-schnell text-to-image generation.

Skips cleanly (via ``requires_flux_runtime``) unless CUDA is available,
``diffusers`` is importable, AND ``black-forest-labs/FLUX.1-schnell`` is
already cached locally — this test must never trigger a ~24 GB download.
"""

from __future__ import annotations

from pathlib import Path

from astel_gpu.text_to_image import build_flux_prompt, generate_image


def test_generate_image_writes_png(
    requires_flux_runtime: None, tmp_path: Path
) -> None:
    prompt = build_flux_prompt("a small red toy boat")
    out_path = tmp_path / "text-reference.png"

    metrics = generate_image(prompt, out_path, seed=0, steps=2, size=256)

    assert out_path.is_file()
    assert metrics["success"] is True
    assert metrics["model"] == "black-forest-labs/FLUX.1-schnell"
    assert metrics["steps"] == 2
    assert metrics["size"] == 256
    assert metrics["wall_time_s"] > 0
