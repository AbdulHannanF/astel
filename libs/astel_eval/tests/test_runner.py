"""Runner: enumerates (case x system) jobs, writes incremental JSON results."""

from __future__ import annotations

from pathlib import Path

from astel_eval.corpus import load_corpus
from astel_eval.runner import RunPlan, default_plan, load_results, run_plan


def test_default_plan_job_count() -> None:
    plan = default_plan()
    assert len(plan.cases) == 50
    assert len(plan.systems) == 4
    assert plan.job_count == 200


def test_run_plan_writes_results(tmp_path: Path) -> None:
    cases = load_corpus()[:2]
    plan = RunPlan(cases=cases)
    results = run_plan(plan, tmp_path)

    assert len(results) == len(cases) * len(plan.systems)
    for r in results:
        assert r.status == "ok"
        assert r.available is False
        assert r.wall_seconds >= 0.0

    written = list(tmp_path.glob("*.json"))
    assert len(written) == len(cases) * len(plan.systems)


def test_run_plan_resumable(tmp_path: Path) -> None:
    cases = load_corpus()[:1]
    plan = RunPlan(cases=cases)
    run_plan(plan, tmp_path)

    files_before = {p: p.stat().st_mtime_ns for p in tmp_path.glob("*.json")}

    # Re-run without overwrite: result files should not be rewritten.
    run_plan(plan, tmp_path, overwrite=False)
    files_after = {p: p.stat().st_mtime_ns for p in tmp_path.glob("*.json")}
    assert files_before == files_after


def test_load_results_roundtrip(tmp_path: Path) -> None:
    cases = load_corpus()[:1]
    plan = RunPlan(cases=cases)
    run_plan(plan, tmp_path)

    loaded = load_results(tmp_path)
    assert len(loaded) == len(plan.systems)
    for r in loaded:
        assert r.case_id == cases[0].id
