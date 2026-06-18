"""Producer dispatcher: stub (default) vs. GPU pipeline (subprocess, opt-in).

The default code path (no ``ASTEL_PRODUCER`` env var, or any value other than
``"gpu"``) calls :func:`astel_api.producer.produce_artifacts` exactly as
before -- byte-for-byte unchanged behaviour.

When ``ASTEL_PRODUCER=gpu`` is set, :func:`produce_artifacts_dispatch` instead
invokes the ``pipelines/gpu`` ``astel_gpu.produce`` CLI as a SUBPROCESS (its
own uv-managed venv, with torch/gsplat), writing artifacts to a temp directory
and then copying them into ``store``. This keeps torch/gsplat OUT of the API's
own import graph and dependency set entirely -- they are never imported here,
only invoked via the ``run-python.cmd`` launcher (which provides the MSVC
build env that torch 2.11's gsplat JIT loader requires on every import) in a
separate process and working directory.

For the **image** modality, when a ``capture_id`` is supplied the uploaded
source image is resolved from ``store`` and passed to the GPU CLI via
``--image``, which selects the real generative path (single image -> TripoSplat
L2 -> 2DGS L3). Local-filesystem stores expose a real path directly; this is the
documented seam where an S3-backed store would download the capture to a temp
file first.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .producer import produce_artifacts
from .storage import ArtifactStore

logger = logging.getLogger(__name__)

# Repo-relative path to the standalone GPU pipeline project.
_GPU_PIPELINE_DIR = Path(__file__).resolve().parents[4] / "pipelines" / "gpu"


def _resolve_capture_image(
    capture_id: str | None, store: ArtifactStore
) -> Path | None:
    """Return an absolute path to the uploaded source image, or ``None``.

    Captures are stored under the ``capture_id`` namespace as ``source<ext>``
    (see ``main.create_capture``). We pick the first ``source*`` member. Returns
    ``None`` if there is no capture or it cannot be resolved to a local path
    (e.g. a future S3 store -- that case will download to a temp file instead).
    """
    if not capture_id:
        return None
    for name in store.list_names(capture_id):
        if name.startswith("source"):
            path = store.path_for(capture_id, name)
            if path is not None:
                return path.resolve()
    return None


def _run_gpu_producer(
    task_id: str,
    modality: str,
    prompt: str,
    store: ArtifactStore,
    capture_id: str | None = None,
    longest_axis_m: float | None = None,
    l6_json_path: Path | None = None,
) -> dict[str, Any]:
    """Invoke ``astel_gpu.produce`` via ``run-python.cmd`` in ``pipelines/gpu``.

    Writes artifacts to a temp directory, then copies each file into
    ``store``. Raises on subprocess failure -- callers already wrap
    ``produce_artifacts``-family calls in a broad except (M1 contract:
    production failure must not fail the submit).
    """
    image_path = _resolve_capture_image(capture_id, store)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        cmd = [
            "cmd",
            "/c",
            str(_GPU_PIPELINE_DIR / "run-python.cmd"),
            "-m",
            "astel_gpu.produce",
            "--task-id",
            task_id,
            "--modality",
            modality,
            "--prompt",
            prompt,
            "--out",
            str(tmp_path),
        ]
        if modality == "image" and image_path is not None:
            cmd += ["--image", str(image_path)]
        if longest_axis_m is not None:
            cmd += ["--longest-axis-m", str(longest_axis_m)]
        if l6_json_path is not None:
            cmd += ["--l6-json", str(l6_json_path)]
        result = subprocess.run(
            cmd,
            cwd=_GPU_PIPELINE_DIR,
            capture_output=True,
            text=True,
            check=True,
        )
        logger.info("gpu producer stdout for %s: %s", task_id, result.stdout)

        artifacts: list[str] = []
        for f in sorted(tmp_path.iterdir()):
            if f.is_file():
                store.put(task_id, f.name, f.read_bytes())
                artifacts.append(f.name)

        splats = 0
        report_path = tmp_path / "quality-report.json"
        if report_path.is_file():
            report = json.loads(report_path.read_text())
            splats = int(report.get("splats", 0))

        return {
            "splats": splats,
            "seed_splats": splats // 24,
            "artifacts": artifacts,
            "conditioning": _gpu_conditioning(modality, prompt, image_path),
        }


def _gpu_conditioning(
    modality: str, prompt: str, image_path: Path | None
) -> str:
    """Best-effort label of what input the GPU path's geometry reflects.

    Both generative cells run prompt/capture-conditioned generation today:
    Image GPU with a resolved capture runs single image -> TripoSplat L2 ->
    2DGS L3 (``"image"``); Text GPU with a non-empty prompt runs prompt ->
    SDXL/FLUX reference image -> TripoSplat L2 -> 2DGS L3 (``"prompt"``) -- so
    the geometry IS conditioned on the prompt via the generated image (wired in
    session 22; the older smoke-refit text path is gone). ``"none"`` covers the
    cases whose geometry is genuinely unconditioned: video (no generator wired
    yet -> smoke-refit fallback) and an empty prompt.
    """
    if modality == "image" and image_path is not None:
        return "image"
    if modality == "text" and prompt.strip():
        return "prompt"
    return "none"


def produce_artifacts_dispatch(
    task_id: str,
    modality: str,
    prompt: str,
    store: ArtifactStore,
    capture_id: str | None = None,
    longest_axis_m: float | None = None,
    l6_json_path: Path | None = None,
) -> dict[str, Any]:
    """Select the stub (default) or GPU (``ASTEL_PRODUCER=gpu``) producer.

    The default path is byte-for-byte :func:`produce_artifacts` plus a
    ``conditioning: "none"`` field -- this dispatcher adds no other behaviour
    unless the env var is set. ``capture_id`` is only consumed by the GPU path
    (to resolve the source image for the image modality); the stub ignores it,
    exactly as before. ``longest_axis_m`` (the Generation Spec's metric size
    estimate) is likewise GPU-only: it grounds the L5/L6 mass + package scale; the
    stub's procedural geometry has no meaningful metric scale, so it is ignored.

    Logs at INFO which producer path is taken (audit §2.3/rec #6), and at
    WARNING if ``ASTEL_PRODUCER`` is set to a non-empty value other than
    exactly ``"gpu"`` -- that value silently falls back to the stub, which is
    an easy-to-miss misconfiguration in a deploy that intended GPU production.
    """
    producer_env = os.environ.get("ASTEL_PRODUCER", "")
    if producer_env == "gpu":
        logger.info("producer dispatch for %s: gpu (ASTEL_PRODUCER=gpu)", task_id)
        return _run_gpu_producer(
            task_id, modality, prompt, store, capture_id, longest_axis_m, l6_json_path
        )

    if producer_env:
        logger.warning(
            "ASTEL_PRODUCER=%r is set but is not exactly 'gpu'; "
            "using the CPU stub producer for %s -- if GPU production was "
            "intended, this is a misconfiguration (audit §2.3)",
            producer_env,
            task_id,
        )
    else:
        logger.info("producer dispatch for %s: stub (ASTEL_PRODUCER unset)", task_id)

    result = produce_artifacts(task_id, modality, prompt, store)
    result["conditioning"] = "none"
    return result
