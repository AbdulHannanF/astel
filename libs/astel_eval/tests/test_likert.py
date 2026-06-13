"""Likert aggregation: N/A handling per CORPUS.md §4.3 axis 5."""

from __future__ import annotations

from astel_eval.likert import summarize_by_system_axis
from astel_eval.scoring_models import Axis, LikertRating, NotApplicable


def test_na_scores_excluded_not_zeroed() -> None:
    ratings = [
        LikertRating(
            case_id="T01",
            system="astel",
            rater_id="r0",
            axis="metric_scale_accuracy",
            score=NotApplicable.NOT_APPLICABLE,
        ),
        LikertRating(
            case_id="T09",
            system="astel",
            rater_id="r0",
            axis="metric_scale_accuracy",
            score=5,
        ),
        LikertRating(
            case_id="T09",
            system="astel",
            rater_id="r1",
            axis="metric_scale_accuracy",
            score=4,
        ),
    ]
    summary = summarize_by_system_axis(ratings)
    key: tuple[str, Axis] = ("astel", "metric_scale_accuracy")
    assert summary[key].n == 2  # only the two real scores
    assert summary[key].n_excluded_na == 1
    assert summary[key].median == 4.5


def test_empty_after_na_exclusion_gives_none_median() -> None:
    ratings = [
        LikertRating(
            case_id="T01",
            system="astel",
            rater_id="r0",
            axis="metric_scale_accuracy",
            score=NotApplicable.NOT_APPLICABLE,
        ),
    ]
    summary = summarize_by_system_axis(ratings)
    key: tuple[str, Axis] = ("astel", "metric_scale_accuracy")
    assert summary[key].n == 0
    assert summary[key].n_excluded_na == 1
    assert summary[key].median is None


def test_iqr_computed_for_normal_axis() -> None:
    ratings = [
        LikertRating(
            case_id="T01",
            system="astel",
            rater_id=f"r{i}",
            axis="geometry_fidelity",
            score=score,
        )
        for i, score in enumerate([1, 2, 3, 4, 5])
    ]
    summary = summarize_by_system_axis(ratings)
    key: tuple[str, Axis] = ("astel", "geometry_fidelity")
    assert summary[key].median == 3.0
    assert summary[key].n == 5
    assert summary[key].n_excluded_na == 0
