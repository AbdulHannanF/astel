"""Timeline metadata for a deformation sequence.

Stores fps, frame count, duration, loop flag, and optional keyframe markers.
Serializes to/from JSON matching the .astel manifest's ``timeline.json`` shape.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Timeline:
    """Metadata describing a deformation timeline.

    Parameters
    ----------
    fps:
        Frames per second; must be > 0.
    frame_count:
        Total number of frames; must be >= 1.
    duration_s:
        Duration in seconds; must be > 0 and approximately equal to
        ``frame_count / fps`` within a relative tolerance of 1e-3.
    loop:
        Whether the animation loops seamlessly.
    keyframes:
        Optional list of keyframe dicts (passed through opaquely).
    """

    fps: float
    frame_count: int
    duration_s: float
    loop: bool
    keyframes: tuple[dict[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError(f"fps must be > 0, got {self.fps}")
        if self.frame_count < 1:
            raise ValueError(f"frame_count must be >= 1, got {self.frame_count}")
        if self.duration_s <= 0:
            raise ValueError(f"duration_s must be > 0, got {self.duration_s}")
        expected = self.frame_count / self.fps
        rel_err = abs(self.duration_s - expected) / max(abs(expected), 1e-12)
        if rel_err > 1e-3:
            raise ValueError(
                f"duration_s={self.duration_s!r} inconsistent with "
                f"frame_count/fps={expected!r} (rel err {rel_err:.2e} > 1e-3)"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to the manifest's ``timeline.json`` shape."""
        d: dict[str, Any] = {
            "fps": self.fps,
            "frame_count": self.frame_count,
            "duration_s": self.duration_s,
            "loop": self.loop,
        }
        if self.keyframes:
            d["keyframes"] = list(self.keyframes)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Timeline:
        """Deserialise from the manifest's ``timeline.json`` shape."""
        keyframes: tuple[dict[str, Any], ...] = tuple(d.get("keyframes", []))
        return cls(
            fps=float(d["fps"]),
            frame_count=int(d["frame_count"]),
            duration_s=float(d["duration_s"]),
            loop=bool(d["loop"]),
            keyframes=keyframes,
        )


def write_timeline_json(timeline: Timeline, path: str | Path) -> None:
    """Write a :class:`Timeline` to a JSON file."""
    Path(path).write_text(json.dumps(timeline.to_dict(), indent=2), encoding="utf-8")


def read_timeline_json(path: str | Path) -> Timeline:
    """Read a :class:`Timeline` from a JSON file."""
    data: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    return Timeline.from_dict(data)
