"""CPU unit tests for the COLMAP command builders.

These pin the exact COLMAP 4.1 flag surface (verified on Box A) so a future
COLMAP upgrade that renames a flag fails here loudly rather than silently
producing an empty reconstruction. The subprocess-executing :func:`run_sfm` is
validated on real images (DTU), not here.
"""

from __future__ import annotations

from pathlib import Path

from astel_gpu.colmap_runner import (
    exhaustive_matcher_cmd,
    feature_extractor_cmd,
    image_undistorter_cmd,
    mapper_cmd,
)

_COLMAP = Path("colmap.exe")


def _pair(cmd: list[str], flag: str) -> str:
    """Return the value following ``flag`` in a flat ``[--flag, value, ...]`` list."""
    return cmd[cmd.index(flag) + 1]


def test_feature_extractor_cmd() -> None:
    cmd = feature_extractor_cmd(
        _COLMAP, Path("db.db"), Path("imgs"), camera_model="OPENCV"
    )
    assert cmd[1] == "feature_extractor"
    assert _pair(cmd, "--database_path") == "db.db"
    assert _pair(cmd, "--image_path") == "imgs"
    assert _pair(cmd, "--ImageReader.camera_model") == "OPENCV"
    assert _pair(cmd, "--ImageReader.single_camera") == "1"
    # GPU toggle lives under FeatureExtraction in 4.1, not SiftExtraction.
    assert _pair(cmd, "--FeatureExtraction.use_gpu") == "1"
    assert "--SiftExtraction.use_gpu" not in cmd


def test_feature_extractor_cmd_cpu_single_camera_off() -> None:
    cmd = feature_extractor_cmd(
        _COLMAP, Path("db.db"), Path("imgs"), single_camera=False, use_gpu=False
    )
    assert _pair(cmd, "--ImageReader.single_camera") == "0"
    assert _pair(cmd, "--FeatureExtraction.use_gpu") == "0"


def test_exhaustive_matcher_cmd() -> None:
    cmd = exhaustive_matcher_cmd(_COLMAP, Path("db.db"))
    assert cmd[1] == "exhaustive_matcher"
    assert _pair(cmd, "--database_path") == "db.db"
    assert _pair(cmd, "--FeatureMatching.use_gpu") == "1"
    assert "--SiftMatching.use_gpu" not in cmd


def test_mapper_cmd() -> None:
    cmd = mapper_cmd(_COLMAP, Path("db.db"), Path("imgs"), Path("sparse"))
    assert cmd[1] == "mapper"
    assert _pair(cmd, "--database_path") == "db.db"
    assert _pair(cmd, "--image_path") == "imgs"
    assert _pair(cmd, "--output_path") == "sparse"


def test_image_undistorter_cmd() -> None:
    cmd = image_undistorter_cmd(
        _COLMAP, Path("imgs"), Path("sparse/0"), Path("undist")
    )
    assert cmd[1] == "image_undistorter"
    assert _pair(cmd, "--image_path") == "imgs"
    assert _pair(cmd, "--input_path") == str(Path("sparse/0"))
    assert _pair(cmd, "--output_path") == "undist"
    assert _pair(cmd, "--output_type") == "COLMAP"
