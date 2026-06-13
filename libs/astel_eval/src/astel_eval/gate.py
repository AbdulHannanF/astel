"""M3-gate evaluator, per CORPUS.md §4.4.

The M3 gate: for each of the 50 cases, does Astel's per-case Bradley-Terry
"overall" strength exceed:

- raw TRELLIS.2's strength on **all 50** cases (required), and
- Meshy-free's strength on **at least 26 of 50** cases (a simple majority,
  required).

Tripo-free is tracked but not part of the gate (CORPUS.md §4.1).

Honesty (CLAUDE.md §1.3 / CORPUS.md §4.6): the evaluator reports losing cases
explicitly, with no spin, regardless of overall pass/fail.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from astel_eval.bradley_terry import per_case_strengths
from astel_eval.corpus import Case
from astel_eval.scoring_models import PairwisePreference

ASTEL = "astel"
TRELLIS2 = "trellis2"
MESHY_FREE = "meshy_free"

#: Cases required for the all-50 TRELLIS.2 comparison and the >=26 Meshy
#: comparison, per CORPUS.md §4.4.
MESHY_MAJORITY_THRESHOLD = 26


@dataclass(frozen=True, slots=True)
class CaseComparison:
    """Per-case outcome of Astel vs. one baseline system."""

    case_id: str
    baseline: str
    astel_strength: float
    baseline_strength: float
    astel_wins: bool
    """True iff Astel's overall strength strictly exceeds the baseline's for
    this case."""


@dataclass(frozen=True, slots=True)
class GateResult:
    """Full M3-gate evaluation result."""

    trellis2_comparisons: tuple[CaseComparison, ...]
    meshy_comparisons: tuple[CaseComparison, ...]

    trellis2_wins: int
    trellis2_total: int
    meshy_wins: int
    meshy_total: int

    trellis2_pass: bool
    """True iff Astel beats TRELLIS.2 on every compared case (all 50)."""

    meshy_pass: bool
    """True iff Astel beats Meshy-free on >= MESHY_MAJORITY_THRESHOLD cases."""

    overall_pass: bool
    """True iff both ``trellis2_pass`` and ``meshy_pass`` hold."""

    trellis2_losses: tuple[str, ...]
    """Case IDs where Astel did not strictly beat TRELLIS.2 (ties count as a
    loss for this gate, per the strict-inequality criterion)."""

    meshy_losses: tuple[str, ...]
    """Case IDs where Astel did not strictly beat Meshy-free."""

    missing_cases: tuple[str, ...]
    """Case IDs from the corpus with no usable pairwise data for one or both
    baselines -- these are NOT counted in totals and are surfaced so the gate
    cannot silently pass on an incomplete dataset."""


def evaluate_gate(
    cases: tuple[Case, ...],
    prefs: list[PairwisePreference],
    *,
    n_bootstrap: int = 200,
) -> GateResult:
    """Evaluate the M3 gate over ``cases`` using pairwise records in ``prefs``.

    For each case, fits a per-case "overall"-axis Bradley-Terry model
    restricted to {Astel, baseline} comparisons for that case (so the strength
    comparison is a direct head-to-head, independent of other systems'
    records).
    """
    case_ids = tuple(c.id for c in cases)
    by_case: dict[str, list[PairwisePreference]] = defaultdict(list)
    for p in prefs:
        by_case[p.case_id].append(p)

    trellis2_comparisons: list[CaseComparison] = []
    meshy_comparisons: list[CaseComparison] = []
    missing: list[str] = []

    for case_id in case_ids:
        case_prefs = by_case.get(case_id, [])

        t2 = _compare_case(case_id, case_prefs, TRELLIS2, n_bootstrap)
        meshy = _compare_case(case_id, case_prefs, MESHY_FREE, n_bootstrap)

        if t2 is None or meshy is None:
            missing.append(case_id)
        if t2 is not None:
            trellis2_comparisons.append(t2)
        if meshy is not None:
            meshy_comparisons.append(meshy)

    trellis2_wins = sum(1 for c in trellis2_comparisons if c.astel_wins)
    meshy_wins = sum(1 for c in meshy_comparisons if c.astel_wins)

    trellis2_total = len(trellis2_comparisons)
    meshy_total = len(meshy_comparisons)

    trellis2_pass = trellis2_total == len(case_ids) and trellis2_wins == trellis2_total
    meshy_pass = meshy_wins >= MESHY_MAJORITY_THRESHOLD

    trellis2_losses = tuple(c.case_id for c in trellis2_comparisons if not c.astel_wins)
    meshy_losses = tuple(c.case_id for c in meshy_comparisons if not c.astel_wins)

    return GateResult(
        trellis2_comparisons=tuple(trellis2_comparisons),
        meshy_comparisons=tuple(meshy_comparisons),
        trellis2_wins=trellis2_wins,
        trellis2_total=trellis2_total,
        meshy_wins=meshy_wins,
        meshy_total=meshy_total,
        trellis2_pass=trellis2_pass,
        meshy_pass=meshy_pass,
        overall_pass=trellis2_pass and meshy_pass,
        trellis2_losses=trellis2_losses,
        meshy_losses=meshy_losses,
        missing_cases=tuple(missing),
    )


def _compare_case(
    case_id: str,
    case_prefs: list[PairwisePreference],
    baseline: str,
    n_bootstrap: int,
) -> CaseComparison | None:
    relevant = [
        p
        for p in case_prefs
        if p.axis == "overall" and {p.system_a, p.system_b} == {ASTEL, baseline}
    ]
    if not relevant:
        return None
    result = per_case_strengths(relevant, case_id, n_bootstrap=n_bootstrap)
    if ASTEL not in result.systems or baseline not in result.systems:
        return None
    astel_strength = result.strength_of(ASTEL)
    baseline_strength = result.strength_of(baseline)
    return CaseComparison(
        case_id=case_id,
        baseline=baseline,
        astel_strength=astel_strength,
        baseline_strength=baseline_strength,
        astel_wins=astel_strength > baseline_strength,
    )
