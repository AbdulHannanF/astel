"""Shared fixtures for astel_splat_io tests."""

from __future__ import annotations

import numpy as np
import pytest

from astel_splat_io.cloud import SplatCloud

SEED = 20260613


@pytest.fixture
def small_cloud() -> SplatCloud:
    """A small, deterministic SplatCloud for golden/round-trip tests."""
    rng = np.random.default_rng(SEED)
    n = 64

    positions = rng.uniform(-2.0, 2.0, size=(n, 3)).astype(np.float32)
    colors_dc = rng.uniform(-1.0, 1.0, size=(n, 3)).astype(np.float32)
    opacity = rng.normal(0.0, 2.0, size=n).astype(np.float32)
    log_scales = rng.uniform(-4.0, -1.0, size=(n, 3)).astype(np.float32)

    quats = rng.normal(size=(n, 4)).astype(np.float32)
    norms = np.linalg.norm(quats, axis=1, keepdims=True)
    quats = (quats / norms).astype(np.float32)

    return SplatCloud(
        positions=positions,
        colors_dc=colors_dc,
        opacity=opacity,
        log_scales=log_scales,
        quats=quats,
    )
