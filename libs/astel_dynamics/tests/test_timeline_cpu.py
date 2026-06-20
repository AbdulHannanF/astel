"""Timeline construction and JSON round-trip tests."""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import pytest

from astel_dynamics.timeline import Timeline, read_timeline_json, write_timeline_json


def test_valid_construction() -> None:
    tl = Timeline(fps=24.0, frame_count=48, duration_s=2.0, loop=False)
    assert tl.fps == 24.0
    assert tl.frame_count == 48
    assert math.isclose(tl.duration_s, 2.0)
    assert tl.loop is False
    assert tl.keyframes == ()


def test_valid_with_keyframes() -> None:
    kf = ({"frame": 0, "label": "start"}, {"frame": 24, "label": "end"})
    tl = Timeline(fps=24.0, frame_count=24, duration_s=1.0, loop=True, keyframes=kf)
    assert len(tl.keyframes) == 2
    assert tl.loop is True


def test_fps_zero_raises() -> None:
    with pytest.raises(ValueError, match="fps"):
        Timeline(fps=0.0, frame_count=1, duration_s=1.0, loop=False)


def test_fps_negative_raises() -> None:
    with pytest.raises(ValueError, match="fps"):
        Timeline(fps=-1.0, frame_count=1, duration_s=1.0, loop=False)


def test_frame_count_zero_raises() -> None:
    with pytest.raises(ValueError, match="frame_count"):
        Timeline(fps=24.0, frame_count=0, duration_s=1.0, loop=False)


def test_duration_zero_raises() -> None:
    with pytest.raises(ValueError, match="duration_s"):
        Timeline(fps=24.0, frame_count=24, duration_s=0.0, loop=False)


def test_duration_mismatch_raises() -> None:
    # 48 frames at 24 fps = 2.0 s; 3.0 s is inconsistent
    with pytest.raises(ValueError, match="inconsistent"):
        Timeline(fps=24.0, frame_count=48, duration_s=3.0, loop=False)


def test_duration_mismatch_just_outside_tolerance() -> None:
    # 24 frames / 24 fps = 1.0 s; 1.0 * (1 + 2e-3) should fail
    bad_duration = 1.0 * (1 + 2e-3)
    with pytest.raises(ValueError, match="inconsistent"):
        Timeline(fps=24.0, frame_count=24, duration_s=bad_duration, loop=False)


def test_duration_within_tolerance() -> None:
    # 1.0 * (1 + 5e-4) should be fine
    ok_duration = 1.0 * (1 + 5e-4)
    tl = Timeline(fps=24.0, frame_count=24, duration_s=ok_duration, loop=False)
    assert math.isclose(tl.duration_s, ok_duration)


def test_json_round_trip_no_keyframes() -> None:
    tl = Timeline(fps=30.0, frame_count=90, duration_s=3.0, loop=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir) / "timeline.json"
        write_timeline_json(tl, p)
        tl2 = read_timeline_json(p)

    assert tl2.fps == tl.fps
    assert tl2.frame_count == tl.frame_count
    assert math.isclose(tl2.duration_s, tl.duration_s)
    assert tl2.loop == tl.loop
    assert tl2.keyframes == ()


def test_json_round_trip_with_keyframes() -> None:
    kf = ({"frame": 0}, {"frame": 10, "label": "peak"})
    tl = Timeline(fps=10.0, frame_count=10, duration_s=1.0, loop=False, keyframes=kf)
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir) / "timeline.json"
        write_timeline_json(tl, p)
        tl2 = read_timeline_json(p)

    assert len(tl2.keyframes) == 2
    assert tl2.keyframes[1]["label"] == "peak"


def test_json_shape_matches_manifest() -> None:
    """The serialised JSON must have the exact keys the manifest expects."""
    tl = Timeline(fps=24.0, frame_count=24, duration_s=1.0, loop=False)
    d = tl.to_dict()
    assert set(d.keys()) == {"fps", "frame_count", "duration_s", "loop"}


def test_json_keyframes_omitted_when_empty() -> None:
    tl = Timeline(fps=24.0, frame_count=24, duration_s=1.0, loop=False)
    d = tl.to_dict()
    assert "keyframes" not in d


def test_json_file_is_valid_json() -> None:
    tl = Timeline(fps=25.0, frame_count=25, duration_s=1.0, loop=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir) / "timeline.json"
        write_timeline_json(tl, p)
        raw = p.read_text(encoding="utf-8")
        parsed = json.loads(raw)
    assert parsed["fps"] == 25.0
