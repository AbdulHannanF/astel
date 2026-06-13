"""Typed loader for the frozen Astel blind-eval corpus v1.

The corpus is specified in ``docs/eval/CORPUS.md`` (repo root) and is FROZEN
per that document's §5 Freeze Policy: 20 text prompts (T01-T20), 20 image case
specs (I01-I20), and 10 capture scenarios (C01-C10) = 50 cases total.

``CORPUS.md``'s markdown tables are the authoritative source of truth, but
parsing markdown tables robustly (embedded commas, quotes, pipes-in-prose) is
brittle. Per the task spec, this module instead loads a checked-in transcription
(``corpus_v1.json``, shipped alongside this module) whose count/IDs/tags are
asserted by ``tests/test_corpus.py`` to match ``CORPUS.md``. If ``CORPUS.md``
is ever revised (it should not be -- it is frozen), that test will fail loudly
and ``corpus_v1.json`` must be re-transcribed and re-verified.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from typing import Literal

Modality = Literal["text", "image", "capture"]

#: The three thin-structure "anchor trio" cases. Per CORPUS.md §1 coverage
#: check: if Astel cannot beat raw TRELLIS.2 on these three, that is the
#: headline finding regardless of aggregate score.
THIN_STRUCTURE_ANCHOR_TRIO: tuple[str, str, str] = ("T02", "T13", "T20")


@dataclass(frozen=True, slots=True)
class GroundTruthScale:
    """Independently-measured ground-truth scale for metric-scale scoring.

    Per CORPUS.md §4.3 axis 5: only cases with a measured ground truth get a
    metric-scale score; all others are N/A for that axis (excluded from its
    aggregate, never scored as 1).
    """

    description: str
    """Human-readable description of the measured quantity, e.g.
    'ring inner diameter ~18mm'."""

    value_mm: float
    """The measured value, in millimetres, used as the denominator for the
    relative-error bucketing in CORPUS.md §4.3 axis 5."""


@dataclass(frozen=True, slots=True)
class Case:
    """A single corpus case (one text prompt, image spec, or capture scenario).

    Fields map directly onto the corresponding CORPUS.md table row.
    """

    id: str
    """Case ID, e.g. ``"T01"``, ``"I07"``, ``"C10"``."""

    modality: Modality
    """One of ``"text"``, ``"image"``, ``"capture"``."""

    prompt_or_spec: str
    """For text cases: the verbatim prompt string (frozen, CORPUS.md §1).
    For image cases: the image spec / sourcing description (CORPUS.md §2).
    For capture cases: the object/scene + script description (CORPUS.md §3)."""

    stresses: tuple[str, ...]
    """Free-text stress descriptors transcribed from the corpus table's
    'Stresses' column, kept as a tuple of individual phrases."""

    expected_failure_modes: str
    """Verbatim (or near-verbatim) 'Expected failure modes' text. Empty string
    for image/capture cases, which CORPUS.md does not give this column for."""

    thin_structure: bool = False
    """True if this case is tagged 'thin structures' per CORPUS.md §4.3 axis 3
    (the cases entering the thin-structure-survival aggregate):
    T02, T03, T13, T14, T15, T18, T20, C03, C05, C08."""

    ground_truth_scale: GroundTruthScale | None = None
    """Present only for cases with an independently measured ground truth used
    for CORPUS.md §4.3 axis 5 (metric-scale accuracy): T09, I12, I14, I15, C07.
    ``None`` means this case is N/A for the metric-scale axis."""

    notes: str = ""
    """Extra context transcribed from the corpus (e.g. environment/lighting/
    motion-pattern details for capture cases, or sourcing notes for image
    cases) that doesn't fit the other fields."""


def _case_from_dict(raw: dict[str, object]) -> Case:
    gt_raw = raw.get("ground_truth_scale")
    gt: GroundTruthScale | None = None
    if gt_raw is not None:
        assert isinstance(gt_raw, dict)
        gt = GroundTruthScale(
            description=str(gt_raw["description"]),
            value_mm=float(gt_raw["value_mm"]),
        )
    stresses_raw = raw["stresses"]
    assert isinstance(stresses_raw, list)
    modality = raw["modality"]
    assert modality in ("text", "image", "capture")
    return Case(
        id=str(raw["id"]),
        modality=modality,
        prompt_or_spec=str(raw["prompt_or_spec"]),
        stresses=tuple(str(s) for s in stresses_raw),
        expected_failure_modes=str(raw.get("expected_failure_modes", "")),
        thin_structure=bool(raw.get("thin_structure", False)),
        ground_truth_scale=gt,
        notes=str(raw.get("notes", "")),
    )


def load_corpus() -> tuple[Case, ...]:
    """Load and return all 50 frozen v1 corpus cases, in CORPUS.md order.

    Cases are returned T01-T20, then I01-I20, then C01-C10.
    """
    data = (
        resources.files("astel_eval")
        .joinpath("corpus_v1.json")
        .read_text(encoding="utf-8")
    )
    raw_cases = json.loads(data)["cases"]
    return tuple(_case_from_dict(c) for c in raw_cases)


def cases_by_modality(cases: tuple[Case, ...], modality: Modality) -> tuple[Case, ...]:
    """Filter ``cases`` to a single modality, preserving order."""
    return tuple(c for c in cases if c.modality == modality)


def case_by_id(cases: tuple[Case, ...], case_id: str) -> Case:
    """Look up a single case by its ID. Raises ``KeyError`` if not found."""
    for c in cases:
        if c.id == case_id:
            return c
    raise KeyError(f"no case with id {case_id!r}")
