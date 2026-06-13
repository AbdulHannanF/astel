"""M3-gate evaluator tests (CORPUS.md §4.4)."""

from __future__ import annotations

from astel_eval.corpus import load_corpus
from astel_eval.gate import evaluate_gate
from astel_eval.scoring_models import PairwisePreference


def _all_prefs_astel_wins_everything() -> list[PairwisePreference]:
    cases = load_corpus()
    prefs: list[PairwisePreference] = []
    for i, case in enumerate(cases):
        prefs.append(
            PairwisePreference(
                case_id=case.id,
                rater_id=f"r{i}",
                axis="overall",
                system_a="astel",
                system_b="trellis2",
                winner="astel",
            )
        )
        # Astel beats Meshy on all 50 here too (we'll flip a few in the FAIL test).
        prefs.append(
            PairwisePreference(
                case_id=case.id,
                rater_id=f"r{i}",
                axis="overall",
                system_a="astel",
                system_b="meshy_free",
                winner="astel",
            )
        )
    return prefs


def test_gate_passes_when_astel_dominates() -> None:
    cases = load_corpus()
    prefs = _all_prefs_astel_wins_everything()

    result = evaluate_gate(cases, prefs, n_bootstrap=20)

    assert result.trellis2_wins == 50
    assert result.trellis2_total == 50
    assert result.trellis2_pass is True

    assert result.meshy_wins == 50
    assert result.meshy_pass is True

    assert result.overall_pass is True
    assert result.trellis2_losses == ()
    assert result.meshy_losses == ()


def test_gate_fails_when_trellis2_loses_one_case() -> None:
    cases = load_corpus()
    prefs = _all_prefs_astel_wins_everything()

    # Flip the very first case's TRELLIS.2 comparison to a TRELLIS.2 win.
    first_case_id = cases[0].id
    prefs = [
        p
        if not (
            p.case_id == first_case_id
            and p.system_b == "trellis2"
            and p.axis == "overall"
        )
        else PairwisePreference(
            case_id=p.case_id,
            rater_id=p.rater_id,
            axis=p.axis,
            system_a=p.system_a,
            system_b=p.system_b,
            winner="trellis2",
        )
        for p in prefs
    ]

    result = evaluate_gate(cases, prefs, n_bootstrap=20)

    assert result.trellis2_pass is False
    assert first_case_id in result.trellis2_losses
    # Meshy gate is unaffected.
    assert result.meshy_pass is True


def test_gate_fails_when_meshy_majority_not_met() -> None:
    cases = load_corpus()
    prefs = _all_prefs_astel_wins_everything()

    # Flip 25 of the 50 Meshy comparisons to Meshy wins -> Astel wins only 25 (<26).
    flip_ids = {c.id for c in cases[:25]}
    new_prefs: list[PairwisePreference] = []
    for p in prefs:
        if p.system_b == "meshy_free" and p.case_id in flip_ids:
            new_prefs.append(
                PairwisePreference(
                    case_id=p.case_id,
                    rater_id=p.rater_id,
                    axis=p.axis,
                    system_a=p.system_a,
                    system_b=p.system_b,
                    winner="meshy_free",
                )
            )
        else:
            new_prefs.append(p)

    result = evaluate_gate(cases, new_prefs, n_bootstrap=20)

    assert result.meshy_wins == 25
    assert result.meshy_pass is False
    assert len(result.meshy_losses) == 25
    assert result.overall_pass is False
    # TRELLIS.2 gate is unaffected and still passes.
    assert result.trellis2_pass is True


def test_missing_cases_reported_not_silently_passed() -> None:
    cases = load_corpus()
    # Only provide pairwise data for the first case.
    prefs = [
        PairwisePreference(
            case_id=cases[0].id,
            rater_id="r0",
            axis="overall",
            system_a="astel",
            system_b="trellis2",
            winner="astel",
        ),
        PairwisePreference(
            case_id=cases[0].id,
            rater_id="r0",
            axis="overall",
            system_a="astel",
            system_b="meshy_free",
            winner="astel",
        ),
    ]
    result = evaluate_gate(cases, prefs, n_bootstrap=10)

    assert len(result.missing_cases) == 49
    assert result.trellis2_total == 1
    assert result.trellis2_pass is False  # 1 != 50
    assert result.meshy_total == 1
