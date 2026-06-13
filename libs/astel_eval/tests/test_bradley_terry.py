"""Bradley-Terry fitting tests: ordering, CI finiteness, symmetry."""

from __future__ import annotations

from numpy.random import default_rng

from astel_eval.bradley_terry import fit_bradley_terry
from astel_eval.scoring_models import PairwisePreference


def _pref(
    case_id: str, rater: str, a: str, b: str, winner: str | None
) -> PairwisePreference:
    return PairwisePreference(
        case_id=case_id,
        rater_id=rater,
        axis="overall",
        system_a=a,
        system_b=b,
        winner=winner,
    )


def test_a_beats_b_beats_c_ordering() -> None:
    prefs: list[PairwisePreference] = []
    # A beats B, B beats C, A beats C -- repeated for signal.
    for i in range(20):
        prefs.append(_pref(f"case{i}", f"r{i}", "A", "B", "A"))
        prefs.append(_pref(f"case{i}", f"r{i}", "B", "C", "B"))
        prefs.append(_pref(f"case{i}", f"r{i}", "A", "C", "A"))

    result = fit_bradley_terry(prefs, n_bootstrap=200, rng=default_rng(42))

    strength = dict(zip(result.systems, result.strengths, strict=True))
    assert strength["A"] > strength["B"] > strength["C"]


def test_confidence_intervals_finite() -> None:
    prefs = [
        _pref("case0", "r0", "A", "B", "A"),
        _pref("case1", "r1", "A", "B", "B"),
        _pref("case2", "r2", "A", "B", "A"),
    ]
    result = fit_bradley_terry(prefs, n_bootstrap=100, rng=default_rng(0))
    for lo, hi in zip(result.ci_low, result.ci_high, strict=True):
        assert lo == lo  # not NaN
        assert hi == hi
        assert lo <= hi


def test_symmetric_record_yields_equal_strengths() -> None:
    # Each system beats the other equally often -> strengths should be ~equal.
    prefs = []
    for i in range(10):
        prefs.append(_pref(f"c{i}a", f"r{i}", "A", "B", "A"))
        prefs.append(_pref(f"c{i}b", f"r{i}", "A", "B", "B"))

    result = fit_bradley_terry(prefs, n_bootstrap=100, rng=default_rng(1))
    strength = dict(zip(result.systems, result.strengths, strict=True))
    assert abs(strength["A"] - strength["B"]) < 0.05


def test_no_comparisons_returns_empty_result() -> None:
    result = fit_bradley_terry([], n_bootstrap=10)
    assert result.systems == ()
    assert result.strengths == ()
    assert result.n_comparisons == 0


def test_ties_split_evenly() -> None:
    prefs = [_pref("c0", "r0", "A", "B", None) for _ in range(10)]
    result = fit_bradley_terry(prefs, n_bootstrap=50, rng=default_rng(2))
    strength = dict(zip(result.systems, result.strengths, strict=True))
    assert abs(strength["A"] - strength["B"]) < 0.05
