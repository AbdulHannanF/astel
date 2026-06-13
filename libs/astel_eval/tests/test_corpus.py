"""Verify corpus_v1.json against docs/eval/CORPUS.md (the frozen source of truth).

CORPUS.md itself is not parsed in full (its markdown tables are not meant to be
machine-parsed -- see astel_eval.corpus module docstring), but we do a
lightweight cross-check: every case ID transcribed into corpus_v1.json must
appear in CORPUS.md, and the thin-structure anchor trio + counts must match.
"""

from __future__ import annotations

import re
from pathlib import Path

from astel_eval.corpus import (
    THIN_STRUCTURE_ANCHOR_TRIO,
    cases_by_modality,
    load_corpus,
)


def _corpus_md_path() -> Path:
    # tests/ -> astel_eval/ -> libs/ -> repo root -> docs/eval/CORPUS.md
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "docs" / "eval" / "CORPUS.md"


def test_corpus_md_exists() -> None:
    assert _corpus_md_path().is_file()


def test_load_corpus_returns_50_cases() -> None:
    cases = load_corpus()
    assert len(cases) == 50


def test_modality_counts() -> None:
    cases = load_corpus()
    assert len(cases_by_modality(cases, "text")) == 20
    assert len(cases_by_modality(cases, "image")) == 20
    assert len(cases_by_modality(cases, "capture")) == 10


def test_ids_match_expected_ranges() -> None:
    cases = load_corpus()
    text_ids = [c.id for c in cases_by_modality(cases, "text")]
    image_ids = [c.id for c in cases_by_modality(cases, "image")]
    capture_ids = [c.id for c in cases_by_modality(cases, "capture")]

    assert text_ids == [f"T{i:02d}" for i in range(1, 21)]
    assert image_ids == [f"I{i:02d}" for i in range(1, 21)]
    assert capture_ids == [f"C{i:02d}" for i in range(1, 11)]


def test_all_ids_unique() -> None:
    cases = load_corpus()
    ids = [c.id for c in cases]
    assert len(ids) == len(set(ids))


def test_thin_structure_anchor_trio_tagged() -> None:
    cases = load_corpus()
    by_id = {c.id: c for c in cases}
    for case_id in THIN_STRUCTURE_ANCHOR_TRIO:
        assert by_id[case_id].thin_structure is True, (
            f"{case_id} must be tagged thin_structure=True (anchor trio)"
        )


def test_thin_structure_tags_match_corpus_md_section_4_3() -> None:
    # CORPUS.md §4.3 axis 3 enumerates exactly these thin-structure cases.
    expected = {
        "T02",
        "T03",
        "T13",
        "T14",
        "T15",
        "T18",
        "T20",
        "C03",
        "C05",
        "C08",
    }
    cases = load_corpus()
    actual = {c.id for c in cases if c.thin_structure}
    assert actual == expected


def test_ground_truth_scale_cases_match_corpus_md_section_4_3() -> None:
    # CORPUS.md §4.3 axis 5 names T09, I12, I14, I15, C07 as having ground truth.
    expected = {"T09", "I12", "I14", "I15", "C07"}
    cases = load_corpus()
    actual = {c.id for c in cases if c.ground_truth_scale is not None}
    assert actual == expected


def test_case_ids_appear_in_corpus_md() -> None:
    text = _corpus_md_path().read_text(encoding="utf-8")
    cases = load_corpus()
    for case in cases:
        # Each ID should appear as a markdown table cell, e.g. "| T01 |".
        pattern = re.compile(rf"\|\s*{re.escape(case.id)}\s*\|")
        assert pattern.search(text), f"{case.id} not found in CORPUS.md"


def test_t09_ground_truth_value() -> None:
    cases = load_corpus()
    by_id = {c.id: c for c in cases}
    t09 = by_id["T09"]
    assert t09.ground_truth_scale is not None
    assert t09.ground_truth_scale.value_mm == 18.0


def test_text_prompts_are_verbatim_quoted_strings() -> None:
    # Spot-check a couple of frozen prompt strings match CORPUS.md exactly.
    cases = load_corpus()
    by_id = {c.id: c for c in cases}
    assert by_id["T01"].prompt_or_spec == (
        "A worn cast-iron skillet with a long handle, matte black surface, "
        "slight rust spots near the rim"
    )
    assert by_id["T20"].prompt_or_spec == (
        "A pair of scissors, open at a 45-degree angle, with metal blades and "
        "orange plastic handles, a visible pivot screw"
    )
