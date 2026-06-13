"""Load/save rating files (CSV or JSON) per CORPUS.md §4.6.

Two flat record types are supported, distinguished by file content:

- **Likert ratings**: one row per (case, system, rater, axis) with a 1-5
  score or ``"N/A"``.
- **Pairwise preferences**: one row per (case, rater, axis, system_a,
  system_b) with a ``winner`` column (``system_a``, ``system_b``, or empty
  for a tie).

CSV columns:

Likert: ``case_id,system,rater_id,axis,score``
Pairwise: ``case_id,rater_id,axis,system_a,system_b,winner``

JSON: either format is a top-level list of objects with the same field names
(``{"likert": [...], "pairwise": [...]}`` is also accepted for a combined
file).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, cast

from astel_eval.scoring_models import (
    Axis,
    LikertRating,
    NotApplicable,
    PairwisePreference,
)


def _parse_score(raw: str | int) -> int | NotApplicable:
    if isinstance(raw, int):
        return raw
    text = str(raw).strip()
    if text.upper() in ("N/A", "NA", ""):
        return NotApplicable.NOT_APPLICABLE
    return int(text)


def _likert_from_row(row: dict[str, Any]) -> LikertRating:
    return LikertRating(
        case_id=str(row["case_id"]),
        system=str(row["system"]),
        rater_id=str(row["rater_id"]),
        axis=cast(Axis, str(row["axis"])),
        score=_parse_score(row["score"]),
    )


def _pairwise_from_row(row: dict[str, Any]) -> PairwisePreference:
    winner_raw = row.get("winner")
    winner: str | None
    if winner_raw is None or str(winner_raw).strip() == "":
        winner = None
    else:
        winner = str(winner_raw)
    return PairwisePreference(
        case_id=str(row["case_id"]),
        rater_id=str(row["rater_id"]),
        axis=cast(Axis, str(row["axis"])),
        system_a=str(row["system_a"]),
        system_b=str(row["system_b"]),
        winner=winner,
    )


def load_likert_csv(path: Path) -> list[LikertRating]:
    """Load Likert ratings from a CSV with columns
    ``case_id,system,rater_id,axis,score``."""
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [_likert_from_row(row) for row in reader]


def load_pairwise_csv(path: Path) -> list[PairwisePreference]:
    """Load pairwise preferences from a CSV with columns
    ``case_id,rater_id,axis,system_a,system_b,winner``."""
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [_pairwise_from_row(row) for row in reader]


def load_ratings_json(
    path: Path,
) -> tuple[list[LikertRating], list[PairwisePreference]]:
    """Load a combined ratings JSON file: ``{"likert": [...], "pairwise": [...]}``.

    Either key may be absent (treated as an empty list).
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    likert = [_likert_from_row(row) for row in data.get("likert", [])]
    pairwise = [_pairwise_from_row(row) for row in data.get("pairwise", [])]
    return likert, pairwise


def load_pairwise_json(path: Path) -> list[PairwisePreference]:
    """Load a JSON file that is a top-level list of pairwise-preference rows."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return [_pairwise_from_row(row) for row in data.get("pairwise", [])]
    return [_pairwise_from_row(row) for row in data]


def load_pairwise(path: Path) -> list[PairwisePreference]:
    """Load pairwise preferences from either a ``.csv`` or ``.json`` file,
    dispatching on the file extension."""
    if path.suffix.lower() == ".csv":
        return load_pairwise_csv(path)
    return load_pairwise_json(path)
