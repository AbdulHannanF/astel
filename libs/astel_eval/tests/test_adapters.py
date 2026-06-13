"""Stub adapters must be unmistakably non-real (CLAUDE.md §1.3 honesty)."""

from __future__ import annotations

from astel_eval.adapters import (
    AstelAdapter,
    MeshyAdapter,
    Trellis2Adapter,
    TripoAdapter,
    all_adapters,
)
from astel_eval.corpus import load_corpus


def test_all_adapters_return_unavailable() -> None:
    case = load_corpus()[0]
    for adapter in all_adapters():
        artifact = adapter.generate(case)
        assert artifact.available is False
        assert "STUB" in artifact.unavailable_reason
        assert artifact.output_paths == ()
        assert artifact.case_id == case.id
        assert artifact.system == adapter.name


def test_adapter_names_are_distinct() -> None:
    names = {a.name for a in all_adapters()}
    assert names == {"astel", "trellis2", "meshy_free", "tripo_free"}


def test_factories_match_all_adapters() -> None:
    assert AstelAdapter().name == "astel"
    assert Trellis2Adapter().name == "trellis2"
    assert MeshyAdapter().name == "meshy_free"
    assert TripoAdapter().name == "tripo_free"
