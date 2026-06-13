"""Aggregation helpers for per-axis Likert scores (CORPUS.md §4.3/§4.4).

Per CORPUS.md §4.3 axis 5 (metric-scale accuracy) and the thin-structure axis:
cases marked N/A for an axis are EXCLUDED from that axis's aggregate, never
scored as 1. This module enforces that by filtering ``NotApplicable`` entries
out before computing medians/IQR.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from astel_eval.scoring_models import Axis, LikertRating, NotApplicable


@dataclass(frozen=True, slots=True)
class AxisSummary:
    """Median + IQR summary of Likert scores for one (system, axis) group."""

    system: str
    axis: Axis
    n: int
    """Number of non-N/A scores included."""

    n_excluded_na: int
    """Number of N/A scores excluded from this summary."""

    median: float | None
    """``None`` if ``n == 0`` (no scoreable observations)."""

    q1: float | None
    q3: float | None


def summarize_by_system_axis(
    ratings: list[LikertRating],
) -> dict[tuple[str, Axis], AxisSummary]:
    """Group ``ratings`` by (system, axis) and compute median/IQR, excluding N/A."""
    groups: dict[tuple[str, Axis], list[int]] = defaultdict(list)
    na_counts: dict[tuple[str, Axis], int] = defaultdict(int)

    for r in ratings:
        key = (r.system, r.axis)
        if isinstance(r.score, NotApplicable):
            na_counts[key] += 1
        else:
            groups[key].append(r.score)

    out: dict[tuple[str, Axis], AxisSummary] = {}
    all_keys = set(groups) | set(na_counts)
    for key in all_keys:
        system, axis = key
        scores = groups.get(key, [])
        n_na = na_counts.get(key, 0)
        median: float | None
        q1: float | None
        q3: float | None
        if scores:
            arr = np.array(scores, dtype=np.float64)
            median = float(np.median(arr))
            q1 = float(np.percentile(arr, 25))
            q3 = float(np.percentile(arr, 75))
        else:
            median = q1 = q3 = None
        out[key] = AxisSummary(
            system=system,
            axis=axis,
            n=len(scores),
            n_excluded_na=n_na,
            median=median,
            q1=q1,
            q3=q3,
        )
    return out
