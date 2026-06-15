"""Tests for the producer dispatcher (stub default vs. ``ASTEL_PRODUCER=gpu``).

These tests must pass WITHOUT torch/gsplat installed in the API's env: the
GPU path is only ever invoked via subprocess, and these tests stub that
subprocess call out entirely. The default (no env var) path is asserted to
delegate to the existing stub :func:`produce_artifacts` unchanged.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

os.environ.setdefault(
    "ASTEL_DATABASE_URL", "sqlite+aiosqlite:///./astel_test_gpu_producer.db"
)

from astel_api.gpu_producer import produce_artifacts_dispatch  # noqa: E402
from astel_api.storage import LocalArtifactStore  # noqa: E402


@pytest.fixture
def store(tmp_path: Path) -> LocalArtifactStore:
    return LocalArtifactStore(tmp_path)


def test_default_dispatch_calls_stub_producer(
    store: LocalArtifactStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no ASTEL_PRODUCER set, dispatch delegates to produce_artifacts."""
    monkeypatch.delenv("ASTEL_PRODUCER", raising=False)

    sentinel: dict[str, Any] = {"splats": 123, "seed_splats": 5, "artifacts": ["x"]}
    with patch(
        "astel_api.gpu_producer.produce_artifacts", return_value=sentinel
    ) as mock_stub:
        result = produce_artifacts_dispatch("task-default", "text", "a mug", store)

    mock_stub.assert_called_once_with("task-default", "text", "a mug", store)
    assert result == sentinel


def test_unrecognized_producer_value_falls_back_to_stub(
    store: LocalArtifactStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Any value other than exactly "gpu" uses the stub path."""
    monkeypatch.setenv("ASTEL_PRODUCER", "something-else")

    sentinel: dict[str, Any] = {"splats": 1, "seed_splats": 0, "artifacts": []}
    with patch(
        "astel_api.gpu_producer.produce_artifacts", return_value=sentinel
    ) as mock_stub:
        result = produce_artifacts_dispatch("task-x", "image", "", store)

    mock_stub.assert_called_once()
    assert result == sentinel


def test_gpu_dispatch_invokes_subprocess_and_collects_artifacts(
    store: LocalArtifactStore, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With ASTEL_PRODUCER=gpu, dispatch runs the GPU CLI via subprocess and
    copies its output files into the artifact store."""
    monkeypatch.setenv("ASTEL_PRODUCER", "gpu")

    def fake_run(
        cmd: list[str], cwd: Any, capture_output: bool, text: bool, check: bool
    ) -> Any:
        # Simulate the GPU CLI writing files into the temp --out dir.
        out_dir = Path(cmd[cmd.index("--out") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "l3.ply").write_bytes(b"fake-ply")
        (out_dir / "quality-report.json").write_text('{"splats": 8000}')

        class _Result:
            stdout = "ok"
            stderr = ""
            returncode = 0

        return _Result()

    with patch("astel_api.gpu_producer.subprocess.run", side_effect=fake_run):
        result = produce_artifacts_dispatch("task-gpu", "text", "a torus", store)

    assert result["splats"] == 8000
    assert result["seed_splats"] == 8000 // 24
    assert set(result["artifacts"]) == {"l3.ply", "quality-report.json"}
    assert store.path_for("task-gpu", "l3.ply") is not None
    assert store.path_for("task-gpu", "quality-report.json") is not None


def test_gpu_image_modality_threads_capture_image(
    store: LocalArtifactStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Image modality + a capture resolves the source file and passes --image."""
    monkeypatch.setenv("ASTEL_PRODUCER", "gpu")

    # Simulate an uploaded capture stored under its capture_id namespace.
    capture_id = "capture-xyz"
    store.put(capture_id, "source.png", b"\x89PNG-fake-bytes")
    expected_image = store.path_for(capture_id, "source.png")
    assert expected_image is not None

    seen: dict[str, list[str]] = {}

    def fake_run(
        cmd: list[str], cwd: Any, capture_output: bool, text: bool, check: bool
    ) -> Any:
        seen["cmd"] = cmd
        out_dir = Path(cmd[cmd.index("--out") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "l3.ply").write_bytes(b"fake-ply")
        (out_dir / "l2.ply").write_bytes(b"fake-l2")
        (out_dir / "quality-report.json").write_text('{"splats": 65536}')

        class _Result:
            stdout = "ok"
            stderr = ""
            returncode = 0

        return _Result()

    with patch("astel_api.gpu_producer.subprocess.run", side_effect=fake_run):
        result = produce_artifacts_dispatch(
            "task-img", "image", "", store, capture_id=capture_id
        )

    cmd = seen["cmd"]
    assert "--image" in cmd
    assert cmd[cmd.index("--image") + 1] == str(expected_image.resolve())
    assert result["splats"] == 65536
    assert "l2.ply" in result["artifacts"]


def test_gpu_image_modality_without_capture_omits_image_flag(
    store: LocalArtifactStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Image modality with no capture falls through without --image (smoke path)."""
    monkeypatch.setenv("ASTEL_PRODUCER", "gpu")
    seen: dict[str, list[str]] = {}

    def fake_run(
        cmd: list[str], cwd: Any, capture_output: bool, text: bool, check: bool
    ) -> Any:
        seen["cmd"] = cmd
        out_dir = Path(cmd[cmd.index("--out") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "quality-report.json").write_text('{"splats": 1}')

        class _Result:
            stdout = "ok"
            stderr = ""
            returncode = 0

        return _Result()

    with patch("astel_api.gpu_producer.subprocess.run", side_effect=fake_run):
        produce_artifacts_dispatch("task-noimg", "image", "", store, capture_id=None)

    assert "--image" not in seen["cmd"]
