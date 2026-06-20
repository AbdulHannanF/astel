"""astel-lod — LOD streaming & splat budget management (CLAUDE.md §8.6).

Ranks every Gaussian by perceptual importance and returns index arrays for
tier-capped subsamples.  The caller subsamples its own cloud; this library
never copies the cloud itself.

Torch-free, numpy-only — runs in CI and in the packaging worker without any
GPU context.
"""

from __future__ import annotations

from .budgets import (
    PLATFORM_BUDGETS,
    TIER_BUDGETS,
    auto_target,
    tier_target,
)
from .descriptor import (
    build_lod_descriptor,
    read_descriptor,
    write_descriptor,
)
from .importance import splat_importance
from .lod import generate_lod_indices, select_lod_indices

__all__ = [
    "PLATFORM_BUDGETS",
    "TIER_BUDGETS",
    "auto_target",
    "build_lod_descriptor",
    "generate_lod_indices",
    "read_descriptor",
    "select_lod_indices",
    "splat_importance",
    "tier_target",
    "write_descriptor",
]
