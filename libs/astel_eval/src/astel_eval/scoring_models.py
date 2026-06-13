"""Data models for the CORPUS.md §4.3 5-axis rubric and §4.2 pairwise prefs.

These are plain, JSON-friendly dataclasses meant to round-trip through the
CSV/JSON ratings files described in CORPUS.md §4.6. They intentionally do not
encode any aggregation logic -- see ``bradley_terry`` and ``gate`` for that.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

Axis = Literal[
    "geometry_fidelity",
    "texture_appearance",
    "thin_structure_survival",
    "printability",
    "metric_scale_accuracy",
    "overall",
]

#: All five per-output rubric axes plus "overall" (used for pairwise prefs and
#: the Bradley-Terry/gate aggregation), per CORPUS.md §4.3-4.4.
ALL_AXES: tuple[Axis, ...] = (
    "geometry_fidelity",
    "texture_appearance",
    "thin_structure_survival",
    "printability",
    "metric_scale_accuracy",
    "overall",
)

#: Axes that are scored per-output on a 1-5 Likert scale (CORPUS.md §4.3).
LIKERT_AXES: tuple[Axis, ...] = (
    "geometry_fidelity",
    "texture_appearance",
    "thin_structure_survival",
    "printability",
    "metric_scale_accuracy",
)


class NotApplicable(Enum):
    """Sentinel for axes that are N/A for a given case/output.

    Per CORPUS.md §4.3 axis 5 (metric-scale accuracy): cases without an
    independently measured ground truth are marked N/A and EXCLUDED from that
    axis's aggregate -- never scored as 1. Likewise, thin-structure-survival is
    only meaningful for thin-structure-tagged cases.
    """

    NOT_APPLICABLE = "N/A"


#: A Likert score is either an int 1-5 or the explicit N/A sentinel.
LikertScore = int | NotApplicable


def is_valid_likert(value: LikertScore) -> bool:
    """True if ``value`` is N/A or an integer in [1, 5]."""
    if isinstance(value, NotApplicable):
        return True
    return 1 <= value <= 5


@dataclass(frozen=True, slots=True)
class LikertRating:
    """One rater's per-axis Likert score for one (case, system) output."""

    case_id: str
    system: str
    rater_id: str
    axis: Axis
    score: LikertScore

    def __post_init__(self) -> None:
        if not is_valid_likert(self.score):
            raise ValueError(
                f"score must be N/A or an int in [1, 5], got {self.score!r}"
            )
        if self.axis not in LIKERT_AXES:
            raise ValueError(
                f"axis {self.axis!r} is not a Likert axis (must be one of "
                f"{LIKERT_AXES!r})"
            )


@dataclass(frozen=True, slots=True)
class PairwisePreference:
    """One rater's pairwise A/B preference for one axis on one case.

    Per CORPUS.md §4.2/§4.4: raters compare two anonymized system outputs for
    the same case and pick a winner (or declare a tie) per axis, including
    "overall".

    ``winner`` is ``None`` for an explicit tie -- ties contribute no signal to
    Bradley-Terry strength differences but are still recorded for
    transparency/auditability (CORPUS.md §4.6 raw-CSV requirement).
    """

    case_id: str
    rater_id: str
    axis: Axis
    system_a: str
    system_b: str
    winner: str | None
    """One of ``system_a``, ``system_b``, or ``None`` for a tie."""

    def __post_init__(self) -> None:
        if self.system_a == self.system_b:
            raise ValueError("system_a and system_b must differ")
        if self.winner is not None and self.winner not in (
            self.system_a,
            self.system_b,
        ):
            raise ValueError(
                f"winner {self.winner!r} must be system_a, system_b, or None"
            )
