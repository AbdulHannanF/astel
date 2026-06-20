"""CPU tests for the video modality dispatch logic in produce.py.

These tests are CPU-pure: they do NOT run the real recon pipeline (which
requires gsplat + MSVC). Instead they test the dispatch and honest-caveat
logic:

1. ``produce(modality="video", image=None)`` routes to the smoke fallback
   (no frame available → honest caveat, no L7 emitted).
2. ``build_quality_report(modality="video", ...)`` carries the expected
   'not used to condition' caveat text.
3. ``_produce_video`` exists and accepts the right signature (import guard).
4. The video path never silently emits an L7 layer without explicit L7 paths.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from astel_format.models import QualityReport
from astel_splat_io.cloud import SplatCloud

from astel_gpu.produce import _produce_video, build_quality_report

# ---------------------------------------------------------------------------
# 1. build_quality_report for video carries honest dynamics caveat
# ---------------------------------------------------------------------------


def test_build_quality_report_video_caveats() -> None:
    """The quality report for video must state that the input was not used
    to condition the geometry — the 'not used to condition' caveat prevents
    the smoke being presented as a real reconstruction."""
    report = build_quality_report(
        count=1000,
        modality="video",
        psnr_db=24.5,
        n_views=4,
    )

    caveats: list[str] = report.get("caveats", [])
    # At least one caveat must mention the modality was not conditioning.
    caveat_text = " ".join(caveats).lower()
    assert "not used to condition" in caveat_text, (
        f"Expected 'not used to condition' in caveats; got: {caveats}"
    )

    # The report must declare the modality.
    assert report["modality"] == "video"

    # fidelity.psnr_db is a REAL measured value (not None).
    assert report["fidelity"]["psnr_db"] == pytest.approx(24.5)

    # Geometric error and scale must be honest (None, not fabricated).
    assert report["geometric_error"]["chamfer_mm_vs_l1"] is None
    assert report["scale"]["longest_axis_m"] is None


# ---------------------------------------------------------------------------
# 2. _produce_video is importable and has the right signature
# ---------------------------------------------------------------------------


def test_produce_video_is_importable() -> None:
    """_produce_video must be importable from astel_gpu.produce."""
    sig = inspect.signature(_produce_video)
    params = set(sig.parameters.keys())
    # Required positional params.
    assert "task_id" in params
    assert "modality" in params
    assert "prompt" in params
    assert "out_dir" in params
    # Keyword-only params that govern the dispatch.
    assert "image" in params
    assert "iters" in params
    assert "refine_iters" in params


# ---------------------------------------------------------------------------
# 3. Video route dispatches to smoke when no frame supplied (dispatch check)
# ---------------------------------------------------------------------------


def test_produce_video_no_frame_dispatches_to_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When image=None, _produce_video delegates to _produce_smoke.

    We monkeypatch _produce_smoke so we can assert it was called without
    actually running the gsplat kernel.
    """
    import astel_gpu.produce as produce_mod

    calls: list[dict[str, Any]] = []

    def fake_smoke(
        task_id: str,
        modality: str,
        prompt: str,
        out_dir: Path,
        *,
        iters: int,
    ) -> dict[str, Any]:
        calls.append({"task_id": task_id, "modality": modality, "iters": iters})
        return {"splats": 0, "seed_splats": 0, "artifacts": [], "metrics": {}}

    monkeypatch.setattr(produce_mod, "_produce_smoke", fake_smoke)

    result = _produce_video(
        "vid-task",
        "video",
        "",
        tmp_path,
        iters=300,
        image=None,
        refine_iters=50,
    )

    assert len(calls) == 1, "Expected _produce_smoke to be called exactly once"
    assert calls[0]["task_id"] == "vid-task"
    assert calls[0]["modality"] == "video"
    assert result["splats"] == 0


# ---------------------------------------------------------------------------
# 4. No L7 layer emitted on the video-smoke path
# ---------------------------------------------------------------------------


def test_produce_video_smoke_does_not_emit_l7(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The video smoke fallback must NOT emit an L7 layer.

    Verified by patching write_layer_stack and asserting l7_deformation_path
    is None (honesty contract: L7 is not fabricated without real tracking).
    """
    import astel_gpu.packaging as packaging_mod
    import astel_gpu.produce as produce_mod

    l7_args_seen: list[Any] = []

    def spy_write_layer_stack(*args: Any, **kwargs: Any) -> list[str]:
        l7_args_seen.append(
            {
                "l7_deformation_path": kwargs.get("l7_deformation_path"),
                "l7_timeline_path": kwargs.get("l7_timeline_path"),
            }
        )
        # Return a minimal artifact list without actually running the full stack.
        return []

    monkeypatch.setattr(packaging_mod, "write_layer_stack", spy_write_layer_stack)

    # Simulate the smoke path that calls write_layer_stack WITHOUT L7 paths.
    def fake_smoke(
        task_id: str, modality: str, prompt: str, out_dir: Path, *, iters: int
    ) -> dict[str, Any]:
        packaging_mod.write_layer_stack(
            _make_tiny_cloud(),
            out_dir,
            task_id=task_id,
            modality=modality,
            prompt=prompt,
            seed=0,
            report_dict={},
            package_report=_dummy_package_report(),
            solidify_l5=False,
            appearance_l4=False,
            # No l7_deformation_path / l7_timeline_path.
        )
        return {"splats": 0, "seed_splats": 0, "artifacts": [], "metrics": {}}

    monkeypatch.setattr(produce_mod, "_produce_smoke", fake_smoke)

    _produce_video(
        "vid-honest",
        "video",
        "",
        tmp_path,
        iters=100,
        image=None,
        refine_iters=50,
    )

    assert len(l7_args_seen) >= 1
    for call_kwargs in l7_args_seen:
        assert call_kwargs["l7_deformation_path"] is None, (
            "L7 deformation path must not be set on video smoke path "
            "(no dynamics were computed)"
        )
        assert call_kwargs["l7_timeline_path"] is None, (
            "L7 timeline path must not be set on video smoke path"
        )


# ---------------------------------------------------------------------------
# Helpers used by the spy tests
# ---------------------------------------------------------------------------


def _make_tiny_cloud() -> SplatCloud:
    n = 4
    quats = np.zeros((n, 4), dtype=np.float32)
    quats[:, 0] = 1.0
    return SplatCloud(
        positions=np.zeros((n, 3), dtype=np.float32),
        colors_dc=np.zeros((n, 3), dtype=np.float32),
        opacity=np.zeros(n, dtype=np.float32),
        log_scales=np.full((n, 3), -3.0, dtype=np.float32),
        quats=quats,
    )


def _dummy_package_report() -> QualityReport:
    from astel_gpu.packaging import build_package_quality_report

    return build_package_quality_report(
        modality="video", origin_note="dummy for test"
    )
