"""Token-cost accounting for the credit ledger (CLAUDE.md §5).

Rates verified live 2026-06-15 (per 1M tokens). Cache reads bill ~0.1x input,
cache writes ~1.25x input (5-min TTL) — see the prompt-caching reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .adapter import TokenUsage

CACHE_READ_MULTIPLIER = 0.1
CACHE_WRITE_MULTIPLIER = 1.25


@dataclass(frozen=True)
class ModelRate:
    """USD per 1M tokens, input and output."""

    input_per_mtok: float
    output_per_mtok: float


#: Verified 2026-06-15 via the claude-api reference. Defaults to Haiku for the
#: Generation Spec stage (constrained extraction, not deep reasoning).
RATES: dict[str, ModelRate] = {
    "claude-haiku-4-5": ModelRate(1.0, 5.0),
    "claude-sonnet-4-6": ModelRate(3.0, 15.0),
    "claude-opus-4-8": ModelRate(5.0, 25.0),
}


def estimate_cost_usd(model: str, usage: TokenUsage) -> float:
    """USD cost of one call, accounting for cache read/write discounts."""
    try:
        rate = RATES[model]
    except KeyError as exc:
        raise KeyError(
            f"no pricing for model {model!r}; known: {sorted(RATES)}"
        ) from exc
    inp = rate.input_per_mtok / 1e6
    out = rate.output_per_mtok / 1e6
    return (
        usage.input_tokens * inp
        + usage.output_tokens * out
        + usage.cache_read_input_tokens * inp * CACHE_READ_MULTIPLIER
        + usage.cache_creation_input_tokens * inp * CACHE_WRITE_MULTIPLIER
    )


def ledger_entry(*, stage: str, model: str, usage: TokenUsage) -> dict[str, Any]:
    """A credit-ledger row for one LLM call (logged per task, CLAUDE.md §5)."""
    return {
        "stage": stage,
        "model": model,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_read_input_tokens": usage.cache_read_input_tokens,
        "cache_creation_input_tokens": usage.cache_creation_input_tokens,
        "cost_usd": round(estimate_cost_usd(model, usage), 6),
    }
