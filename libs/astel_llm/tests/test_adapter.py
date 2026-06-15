"""Tests for the adapter layer (fixture key stability + record/replay)."""

from __future__ import annotations

from pathlib import Path

from astel_llm.adapter import FixtureAdapter, StructuredResult, TokenUsage, fixture_key


def test_fixture_key_is_stable_and_sensitive() -> None:
    a = fixture_key("m", "sys", "user")
    assert a == fixture_key("m", "sys", "user")
    assert a != fixture_key("m", "sys", "USER")
    assert a != fixture_key("other", "sys", "user")


def test_record_then_replay(tmp_path: Path) -> None:
    adapter = FixtureAdapter(tmp_path)
    result = StructuredResult(
        data={"k": "v"},
        usage=TokenUsage(input_tokens=5, output_tokens=2),
        model="claude-haiku-4-5",
    )
    adapter.record(model="claude-haiku-4-5", system="s", user="u", result=result)

    got = adapter.complete_structured(
        system="s", user="u", schema={}, model="claude-haiku-4-5"
    )
    assert got.data == {"k": "v"}
    assert got.usage.input_tokens == 5
    assert got.model == "claude-haiku-4-5"
