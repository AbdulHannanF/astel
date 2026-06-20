"""Tests for budgets.py — capping, known values, and error messages."""

from __future__ import annotations

import pytest

from astel_lod.budgets import (
    PLATFORM_BUDGETS,
    TIER_BUDGETS,
    auto_target,
    tier_target,
)

# ---------------------------------------------------------------------------
# Constants sanity
# ---------------------------------------------------------------------------


def test_tier_budgets_keys() -> None:
    assert set(TIER_BUDGETS) == {"lowpoly", "standard", "cinematic"}


def test_tier_budgets_values_are_positive_integers() -> None:
    for name, v in TIER_BUDGETS.items():
        assert isinstance(v, int) and v > 0, f"TIER_BUDGETS[{name!r}] = {v}"


def test_tier_budgets_strictly_increasing() -> None:
    """lowpoly < standard < cinematic."""
    lp, std, cine = (
        TIER_BUDGETS["lowpoly"],
        TIER_BUDGETS["standard"],
        TIER_BUDGETS["cinematic"],
    )
    assert lp < std < cine


def test_platform_budgets_keys() -> None:
    assert set(PLATFORM_BUDGETS) == {"mobile", "web", "console", "cinematic"}


def test_platform_budgets_values_are_positive_integers() -> None:
    for name, v in PLATFORM_BUDGETS.items():
        assert isinstance(v, int) and v > 0, f"PLATFORM_BUDGETS[{name!r}] = {v}"


# ---------------------------------------------------------------------------
# auto_target
# ---------------------------------------------------------------------------


def test_auto_target_caps_at_platform_budget() -> None:
    # Large cloud: should return the platform budget.
    for platform, budget in PLATFORM_BUDGETS.items():
        result = auto_target(budget * 10, platform)
        assert result == budget, f"auto_target cap failed for {platform!r}"


def test_auto_target_never_exceeds_n_splats() -> None:
    # Small cloud: should return n_splats, not the budget.
    for platform in PLATFORM_BUDGETS:
        n = 10
        result = auto_target(n, platform)
        assert result == n, f"auto_target({n}, {platform!r}) = {result}, expected {n}"


def test_auto_target_exact_boundary() -> None:
    """When n_splats == budget, result == budget."""
    for platform, budget in PLATFORM_BUDGETS.items():
        assert auto_target(budget, platform) == budget


def test_auto_target_unknown_platform_raises_with_helpful_message() -> None:
    with pytest.raises(ValueError, match="Unknown platform") as exc_info:
        auto_target(1_000_000, "toaster")
    msg = str(exc_info.value)
    # The error must list valid platforms.
    for p in PLATFORM_BUDGETS:
        assert p in msg, f"Valid platform {p!r} not in error message: {msg!r}"


def test_auto_target_known_values() -> None:
    assert auto_target(50_000, "mobile") == 50_000  # cloud smaller than budget
    assert auto_target(200_000, "mobile") == 100_000  # capped at mobile budget
    assert auto_target(300_000, "web") == 300_000  # cloud smaller than web budget
    assert auto_target(1_000_000, "web") == 500_000  # capped at web budget


# ---------------------------------------------------------------------------
# tier_target
# ---------------------------------------------------------------------------


def test_tier_target_caps_at_tier_budget() -> None:
    for tier, budget in TIER_BUDGETS.items():
        result = tier_target(budget * 10, tier)
        assert result == budget, f"tier_target cap failed for {tier!r}"


def test_tier_target_never_exceeds_n_splats() -> None:
    for tier in TIER_BUDGETS:
        n = 50
        result = tier_target(n, tier)
        assert result == n, f"tier_target({n}, {tier!r}) = {result}, expected {n}"


def test_tier_target_exact_boundary() -> None:
    for tier, budget in TIER_BUDGETS.items():
        assert tier_target(budget, tier) == budget


def test_tier_target_unknown_tier_raises_with_helpful_message() -> None:
    with pytest.raises(ValueError, match="Unknown tier") as exc_info:
        tier_target(500_000, "ultramax")
    msg = str(exc_info.value)
    for t in TIER_BUDGETS:
        assert t in msg, f"Valid tier {t!r} not in error message: {msg!r}"


def test_tier_target_known_values() -> None:
    assert tier_target(50_000, "lowpoly") == 50_000  # smaller than budget
    assert tier_target(500_000, "lowpoly") == 100_000  # capped
    assert tier_target(500_000, "standard") == 500_000  # smaller than budget
    assert tier_target(2_000_000, "standard") == 1_000_000  # capped
    assert tier_target(10_000_000, "cinematic") == 5_000_000  # capped
