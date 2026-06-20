"""astel-scene — scene seeds / scene composition (CLAUDE.md §8).

Composes multiple single-object Gaussian-splat clouds into a small
multi-object scene with ground-contact placement and no-overlap resolution.

Torch-free, CPU-only.  Operates on raw numpy arrays; no SplatCloud import.
"""

from __future__ import annotations

from .compose import compose_scene
from .contacts import aabb, ground_drop, resolve_no_overlap
from .layout import Placement, SceneLayout, SceneObject
from .llm_stage import build_scene_layout
from .splats import ObjectSplats
from .transform import apply_placement, quat_from_yaw, quat_multiply

__all__ = [
    "ObjectSplats",
    "Placement",
    "SceneLayout",
    "SceneObject",
    "aabb",
    "apply_placement",
    "build_scene_layout",
    "compose_scene",
    "ground_drop",
    "quat_from_yaw",
    "quat_multiply",
    "resolve_no_overlap",
]
