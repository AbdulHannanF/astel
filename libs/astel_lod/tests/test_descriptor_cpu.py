"""Tests for descriptor.py — schema, sorting, validation, JSON round-trip."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from astel_lod.descriptor import (
    build_lod_descriptor,
    read_descriptor,
    write_descriptor,
)

# ---------------------------------------------------------------------------
# build_lod_descriptor
# ---------------------------------------------------------------------------


def test_build_returns_correct_schema() -> None:
    tiers = [
        {"name": "lowpoly", "count": 100_000, "file": "lod_lowpoly.ply"},
        {"name": "standard", "count": 1_000_000, "file": "lod_standard.ply"},
    ]
    desc = build_lod_descriptor(tiers)
    assert desc["schema"] == "astel.lod/v0"


def test_build_sorts_tiers_ascending_by_count() -> None:
    """Input order must not matter; output is always ascending by count."""
    tiers = [
        {"name": "cinematic", "count": 5_000_000, "file": "c.ply"},
        {"name": "lowpoly", "count": 100_000, "file": "l.ply"},
        {"name": "standard", "count": 1_000_000, "file": "s.ply"},
    ]
    desc = build_lod_descriptor(tiers)
    counts = [t["count"] for t in desc["tiers"]]
    assert counts == sorted(counts), f"Tiers not sorted ascending: {counts}"
    assert counts == [100_000, 1_000_000, 5_000_000]


def test_build_tiers_field_has_correct_keys() -> None:
    tiers = [
        {"name": "lowpoly", "count": 100_000, "file": "lod_lowpoly.ply"},
    ]
    desc = build_lod_descriptor(tiers)
    for tier in desc["tiers"]:
        assert "name" in tier
        assert "count" in tier
        assert "file" in tier


def test_build_rejects_duplicate_counts() -> None:
    tiers = [
        {"name": "a", "count": 100_000, "file": "a.ply"},
        {"name": "b", "count": 100_000, "file": "b.ply"},
    ]
    with pytest.raises(ValueError, match="strictly increasing"):
        build_lod_descriptor(tiers)


def test_build_rejects_decreasing_counts() -> None:
    """Even if the input is sorted descending the validation must catch it.

    Note: build_lod_descriptor sorts ascending internally, so [1_000_000, 100_000]
    becomes [100_000, 1_000_000] which is valid.  To test the rejection path we
    supply two tiers with the SAME count (duplicate), which remains invalid after
    sorting.
    """
    tiers_equal = [
        {"name": "x", "count": 500_000, "file": "x.ply"},
        {"name": "y", "count": 500_000, "file": "y.ply"},
    ]
    with pytest.raises(ValueError, match="strictly increasing"):
        build_lod_descriptor(tiers_equal)


def test_build_single_tier_ok() -> None:
    tiers = [{"name": "standard", "count": 1_000_000, "file": "s.ply"}]
    desc = build_lod_descriptor(tiers)
    assert len(desc["tiers"]) == 1


def test_build_preserves_names_and_files() -> None:
    tiers = [
        {"name": "lp", "count": 100_000, "file": "lp.spz"},
        {"name": "std", "count": 1_000_000, "file": "std.spz"},
    ]
    desc = build_lod_descriptor(tiers)
    names = [t["name"] for t in desc["tiers"]]
    files = [t["file"] for t in desc["tiers"]]
    assert names == ["lp", "std"]
    assert files == ["lp.spz", "std.spz"]


# ---------------------------------------------------------------------------
# JSON round-trip via write_descriptor / read_descriptor
# ---------------------------------------------------------------------------


def _sample_desc() -> dict[str, object]:
    return build_lod_descriptor(
        [
            {"name": "lowpoly", "count": 100_000, "file": "l.ply"},
            {"name": "standard", "count": 1_000_000, "file": "s.ply"},
            {"name": "cinematic", "count": 5_000_000, "file": "c.ply"},
        ]
    )


def test_write_and_read_roundtrip() -> None:
    desc = _sample_desc()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "lod.json"
        write_descriptor(desc, path)
        recovered = read_descriptor(path)
    assert recovered == desc


def test_write_produces_valid_json() -> None:
    desc = _sample_desc()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "lod.json"
        write_descriptor(desc, path)
        raw = path.read_text(encoding="utf-8")
    parsed = json.loads(raw)
    assert parsed["schema"] == "astel.lod/v0"


def test_read_rejects_wrong_schema() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bad.json"
        path.write_text(json.dumps({"schema": "something.else/v1", "tiers": []}))
        with pytest.raises(ValueError, match="Expected schema"):
            read_descriptor(path)


def test_read_rejects_missing_schema() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bad.json"
        path.write_text(json.dumps({"tiers": []}))
        with pytest.raises(ValueError, match="Expected schema"):
            read_descriptor(path)


def test_roundtrip_preserves_tier_order() -> None:
    """After a write/read cycle, tier order (ascending count) must be preserved."""
    desc = _sample_desc()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "lod.json"
        write_descriptor(desc, path)
        recovered = read_descriptor(path)
    counts = [t["count"] for t in recovered["tiers"]]
    assert counts == sorted(counts)
