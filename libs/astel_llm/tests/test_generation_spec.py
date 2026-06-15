"""End-to-end Generation Spec stage test — offline, via FixtureAdapter.

No API key, no network, no spend (the founder-gate rule). We record a fixture
for the exact (model, system, user) the stage will request, then run the stage
against it — exercising the full path the live adapter will take later.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from astel_llm.adapter import (
    FixtureAdapter,
    FixtureMissingError,
    StructuredResult,
    TokenUsage,
)
from astel_llm.generation_spec import (
    DEFAULT_MODEL,
    SYSTEM_PROMPT,
    build_generation_spec,
)

PROMPT = "a small brass steampunk pocket watch on a chain"

CANNED = {
    "object_class": "pocket watch",
    "summary": "A small brass steampunk pocket watch on a chain.",
    "parts": [
        {"name": "case", "material": "brass"},
        {"name": "chain", "material": "brass"},
        {"name": "crystal", "material": "glass"},
    ],
    "materials": ["brass", "glass"],
    "style": "steampunk",
    "target_scale": {
        "longest_axis_m": 0.05,
        "confidence": 0.6,
        "low_m": 0.04,
        "high_m": 0.07,
    },
    "symmetry": "radial",
}


def _adapter_with_canned(tmp_path: Path) -> FixtureAdapter:
    adapter = FixtureAdapter(tmp_path)
    adapter.record(
        model=DEFAULT_MODEL,
        system=SYSTEM_PROMPT,
        user=PROMPT,
        result=StructuredResult(
            data=CANNED,
            usage=TokenUsage(
                input_tokens=900,
                output_tokens=180,
                cache_read_input_tokens=850,
            ),
            model=DEFAULT_MODEL,
        ),
    )
    return adapter


def test_build_generation_spec_offline(tmp_path: Path) -> None:
    adapter = _adapter_with_canned(tmp_path)

    result = build_generation_spec(PROMPT, adapter)

    assert result.spec.object_class == "pocket watch"
    assert result.spec.symmetry == "radial"
    assert {m for m in result.spec.materials} == {"brass", "glass"}
    # confidence band is preserved and user-overridable
    assert result.spec.target_scale.confidence == 0.6
    # ledger row carries a real, non-zero cost
    assert result.ledger["stage"] == "generation_spec"
    assert result.ledger["model"] == DEFAULT_MODEL
    assert result.ledger["cost_usd"] > 0.0


def test_missing_fixture_raises(tmp_path: Path) -> None:
    adapter = FixtureAdapter(tmp_path)
    with pytest.raises(FixtureMissingError):
        build_generation_spec("an unrecorded prompt", adapter)


def test_empty_prompt_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        build_generation_spec("   ", FixtureAdapter(tmp_path))
