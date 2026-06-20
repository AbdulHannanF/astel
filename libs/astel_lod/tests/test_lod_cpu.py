"""Tests for lod.py — correct top-k, sorted indices, and nested subset property."""

from __future__ import annotations

import numpy as np
import pytest
from _clouds import dominant_cloud

from astel_lod.importance import splat_importance
from astel_lod.lod import generate_lod_indices, select_lod_indices

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ground_truth_topk(importance: np.ndarray, k: int) -> set[int]:
    """Brute-force top-k by full sort (reference for argpartition-based impl)."""
    n = len(importance)
    k = min(k, n)
    return set(np.argsort(-importance)[:k].tolist())


# ---------------------------------------------------------------------------
# select_lod_indices
# ---------------------------------------------------------------------------


def test_select_returns_exact_count() -> None:
    opacity, log_scales = dominant_cloud(n_dominant=5, n_background=20)
    imp = splat_importance(opacity, log_scales)
    for k in [1, 3, 5, 10, 25]:
        idx = select_lod_indices(imp, k)
        assert len(idx) == k, f"Expected {k} indices, got {len(idx)}"


def test_select_indices_are_sorted_ascending() -> None:
    opacity, log_scales = dominant_cloud(n_dominant=5, n_background=20)
    imp = splat_importance(opacity, log_scales)
    idx = select_lod_indices(imp, 10)
    assert np.all(np.diff(idx) > 0), "Output indices must be strictly ascending"


def test_select_topk_matches_ground_truth() -> None:
    """The returned set must equal the brute-force top-k set."""
    opacity, log_scales = dominant_cloud(n_dominant=5, n_background=20)
    imp = splat_importance(opacity, log_scales)
    for k in [1, 5, 10, 20, 25]:
        idx = select_lod_indices(imp, k)
        expected = _ground_truth_topk(imp, k)
        assert set(idx.tolist()) == expected, (
            f"k={k}: returned {set(idx.tolist())} != expected {expected}"
        )


def test_select_dominant_block_always_in_top5() -> None:
    """The 5 dominant Gaussians (indices 0-4) must all appear in top-5."""
    opacity, log_scales = dominant_cloud(n_dominant=5, n_background=20)
    imp = splat_importance(opacity, log_scales)
    idx = select_lod_indices(imp, 5)
    assert set(idx.tolist()) == {0, 1, 2, 3, 4}


def test_select_target_gte_n_returns_all() -> None:
    opacity, log_scales = dominant_cloud(n_dominant=3, n_background=7)
    imp = splat_importance(opacity, log_scales)
    n = len(imp)
    for k in [n, n + 1, n + 100]:
        idx = select_lod_indices(imp, k)
        assert len(idx) == n
        assert set(idx.tolist()) == set(range(n))


def test_select_target_zero_raises() -> None:
    imp = np.array([0.5, 0.3, 0.8], dtype=np.float64)
    with pytest.raises(ValueError, match="target_count must be > 0"):
        select_lod_indices(imp, 0)


def test_select_target_negative_raises() -> None:
    imp = np.array([0.5, 0.3, 0.8], dtype=np.float64)
    with pytest.raises(ValueError, match="target_count must be > 0"):
        select_lod_indices(imp, -5)


def test_select_single_element_cloud() -> None:
    imp = np.array([0.99], dtype=np.float64)
    idx = select_lod_indices(imp, 1)
    assert list(idx) == [0]

    idx_large = select_lod_indices(imp, 100)
    assert list(idx_large) == [0]


def test_select_output_dtype_is_integer() -> None:
    imp = np.array([0.1, 0.9, 0.5], dtype=np.float64)
    idx = select_lod_indices(imp, 2)
    assert np.issubdtype(idx.dtype, np.integer)


# ---------------------------------------------------------------------------
# generate_lod_indices — counts, subset, monotonicity
# ---------------------------------------------------------------------------


def test_generate_returns_correct_tier_sizes() -> None:
    opacity, log_scales = dominant_cloud(n_dominant=5, n_background=95)
    targets = [10, 50, 100]
    tiers = generate_lod_indices(opacity, log_scales, targets)
    assert len(tiers) == len(targets)
    for t, idx in zip(targets, tiers, strict=True):
        assert len(idx) == t, f"Expected {t} indices, got {len(idx)}"


def test_generate_tier_sizes_capped_at_n() -> None:
    opacity, log_scales = dominant_cloud(n_dominant=3, n_background=7)
    n = 10
    targets = [5, 15, 100]
    tiers = generate_lod_indices(opacity, log_scales, targets)
    assert len(tiers[0]) == 5
    assert len(tiers[1]) == n  # capped
    assert len(tiers[2]) == n  # capped


def test_generate_nested_subset_property() -> None:
    """CRITICAL: smaller tier's selected set must be a subset of every larger tier.

    This is the fundamental guarantee for progressive LOD streaming: a client
    holding tier-k never re-downloads splats when upgrading to tier-K > k.
    """
    opacity, log_scales = dominant_cloud(n_dominant=10, n_background=90)
    targets = [5, 20, 50, 100]
    tiers = generate_lod_indices(opacity, log_scales, targets)

    for i, smaller in enumerate(tiers):
        for j, larger in enumerate(tiers):
            if len(smaller) <= len(larger):
                s_set = set(smaller.tolist())
                l_set = set(larger.tolist())
                assert s_set <= l_set, (
                    f"Nested subset violated: tier[{i}] (size {len(smaller)}) "
                    f"is NOT a subset of tier[{j}] (size {len(larger)}). "
                    f"Extra elements: {s_set - l_set}"
                )


def test_generate_each_tier_is_sorted_ascending() -> None:
    opacity, log_scales = dominant_cloud(n_dominant=5, n_background=45)
    targets = [5, 20, 50]
    tiers = generate_lod_indices(opacity, log_scales, targets)
    for t, idx in zip(targets, tiers, strict=True):
        assert np.all(np.diff(idx) > 0) or len(idx) <= 1, (
            f"Tier {t}: indices must be strictly ascending"
        )


def test_generate_matches_select_lod_indices() -> None:
    """generate_lod_indices must agree with individual select_lod_indices calls."""
    opacity, log_scales = dominant_cloud(n_dominant=5, n_background=45)
    targets = [5, 15, 30]
    tiers = generate_lod_indices(opacity, log_scales, targets)
    imp = splat_importance(opacity, log_scales)
    for t, idx in zip(targets, tiers, strict=True):
        expected = select_lod_indices(imp, t)
        assert np.array_equal(idx, expected), (
            f"generate_lod_indices[{t}] != select_lod_indices({t})"
        )


def test_generate_zero_target_raises() -> None:
    opacity, log_scales = dominant_cloud()
    with pytest.raises(ValueError, match="target_count must be > 0"):
        generate_lod_indices(opacity, log_scales, [10, 0, 50])


def test_generate_preserves_input_order_in_return() -> None:
    """Tiers are returned in the same order as target_counts (not sorted by size)."""
    opacity, log_scales = dominant_cloud(n_dominant=5, n_background=45)
    targets = [30, 5, 20]  # deliberately unordered
    tiers = generate_lod_indices(opacity, log_scales, targets)
    assert len(tiers[0]) == 30
    assert len(tiers[1]) == 5
    assert len(tiers[2]) == 20
