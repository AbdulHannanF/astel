"""Tests for importance.py — finite, monotone, and correctly ordered."""

from __future__ import annotations

import numpy as np
import pytest
from _clouds import (
    dominant_cloud,
    monotone_opacity_cloud,
    monotone_scale_cloud,
    single_splat_pair,
)

from astel_lod.importance import splat_importance


def test_importance_is_finite_for_finite_inputs() -> None:
    opacity, log_scales = dominant_cloud(n_dominant=5, n_background=20)
    scores = splat_importance(opacity, log_scales)
    assert np.all(np.isfinite(scores)), "All importance scores must be finite"


def test_importance_dtype_is_float64() -> None:
    opacity, log_scales = dominant_cloud()
    scores = splat_importance(opacity, log_scales)
    assert scores.dtype == np.float64


def test_importance_shape_matches_n() -> None:
    opacity, log_scales = dominant_cloud(n_dominant=3, n_background=7)
    scores = splat_importance(opacity, log_scales)
    assert scores.shape == (10,)


def test_high_opacity_large_splat_beats_low_opacity_small_splat() -> None:
    """A dominant Gaussian (high opacity, large scale) must outscore background."""
    opacity, log_scales = single_splat_pair()
    scores = splat_importance(opacity, log_scales)
    assert scores[0] > scores[1], (
        f"Expected dominant splat (idx 0, score {scores[0]:.4f}) > "
        f"background splat (idx 1, score {scores[1]:.4f})"
    )


def test_dominant_block_beats_all_background() -> None:
    n_dom = 5
    opacity, log_scales = dominant_cloud(n_dominant=n_dom, n_background=50)
    scores = splat_importance(opacity, log_scales)
    min_dominant = scores[:n_dom].min()
    max_background = scores[n_dom:].max()
    assert min_dominant > max_background, (
        f"Every dominant score must exceed every background score; "
        f"min_dominant={min_dominant:.4f}, max_background={max_background:.4f}"
    )


def test_importance_monotone_in_opacity() -> None:
    """With fixed isotropic scale (footprint=1), importance == opacity."""
    opacity, log_scales = monotone_opacity_cloud(n=10)
    scores = splat_importance(opacity, log_scales)
    # Scores should be strictly increasing (same order as opacity).
    diffs = np.diff(scores)
    assert np.all(diffs > 0), (
        f"Importance must be strictly increasing with opacity; got diffs {diffs}"
    )


def test_importance_monotone_in_scale() -> None:
    """With fixed opacity, importance == opacity * footprint, so it grows with scale."""
    opacity, log_scales = monotone_scale_cloud(n=10)
    scores = splat_importance(opacity, log_scales)
    diffs = np.diff(scores)
    assert np.all(diffs > 0), (
        f"Importance must be strictly increasing with scale; got diffs {diffs}"
    )


def test_importance_zero_opacity_gives_zero() -> None:
    opacity = np.array([0.0, 0.5], dtype=np.float64)
    log_scales = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=np.float64)
    scores = splat_importance(opacity, log_scales)
    assert scores[0] == pytest.approx(0.0)


def test_importance_single_splat() -> None:
    opacity = np.array([0.8], dtype=np.float64)
    log_scales = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    scores = splat_importance(opacity, log_scales)
    # footprint = exp(0)^3 = 1 → importance = 0.8 * 1 = 0.8
    assert scores[0] == pytest.approx(0.8)


def test_importance_known_formula() -> None:
    """Exact numeric check: importance = opacity * exp(sum(log_scales))."""
    opacity = np.array([0.5, 0.3], dtype=np.float64)
    log_scales = np.array([[1.0, 2.0, 0.5], [0.0, -1.0, 1.0]], dtype=np.float64)
    scores = splat_importance(opacity, log_scales)

    expected_0 = 0.5 * np.exp(1.0 + 2.0 + 0.5)
    expected_1 = 0.3 * np.exp(0.0 + (-1.0) + 1.0)
    assert scores[0] == pytest.approx(expected_0, rel=1e-12)
    assert scores[1] == pytest.approx(expected_1, rel=1e-12)
