"""Tests for token-cost accounting and ledger entries."""

from __future__ import annotations

import math

import pytest

from astel_llm.adapter import TokenUsage
from astel_llm.pricing import estimate_cost_usd, ledger_entry


def test_haiku_basic_cost() -> None:
    # 10k in / 5k out on Haiku ($1/$5 per 1M): 0.01 + 0.025 = 0.035
    usage = TokenUsage(input_tokens=10_000, output_tokens=5_000)
    cost = estimate_cost_usd("claude-haiku-4-5", usage)
    assert math.isclose(cost, 0.035, rel_tol=1e-9)


def test_cache_discounts_applied() -> None:
    # cache read bills 0.1x input, cache write 1.25x input.
    usage = TokenUsage(
        input_tokens=0,
        output_tokens=0,
        cache_read_input_tokens=10_000,       # 10k * $1/1M * 0.1 = 0.001
        cache_creation_input_tokens=10_000,   # 10k * $1/1M * 1.25 = 0.0125
    )
    assert math.isclose(
        estimate_cost_usd("claude-haiku-4-5", usage), 0.0135, rel_tol=1e-9
    )


def test_unknown_model_raises() -> None:
    with pytest.raises(KeyError):
        estimate_cost_usd("gpt-4", TokenUsage(input_tokens=1, output_tokens=1))


def test_ledger_entry_shape() -> None:
    usage = TokenUsage(input_tokens=1000, output_tokens=200)
    entry = ledger_entry(stage="generation_spec", model="claude-haiku-4-5", usage=usage)
    assert entry["stage"] == "generation_spec"
    assert entry["model"] == "claude-haiku-4-5"
    assert entry["input_tokens"] == 1000
    assert entry["cost_usd"] > 0.0
