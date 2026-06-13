"""Smoke tests for the python -m astel_eval CLI."""

from __future__ import annotations

from pathlib import Path

from astel_eval.__main__ import main


def test_list_command(capsys: object) -> None:
    rc = main(["list"])
    assert rc == 0


def test_run_command(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    rc = main(["run", "--results-dir", str(results_dir)])
    assert rc == 0
    assert len(list(results_dir.glob("*.json"))) == 200


def test_score_and_gate_commands(tmp_path: Path) -> None:
    pairwise_path = tmp_path / "pairwise.json"
    pairwise_path.write_text(
        '{"pairwise": ['
        '{"case_id": "T01", "rater_id": "r0", "axis": "overall", '
        '"system_a": "astel", "system_b": "trellis2", "winner": "astel"}'
        "]}",
        encoding="utf-8",
    )
    rc_score = main(["score", "--pairwise", str(pairwise_path), "--bootstrap", "10"])
    assert rc_score == 0

    rc_gate = main(["gate", "--pairwise", str(pairwise_path), "--bootstrap", "5"])
    # Gate fails (only 1/50 cases covered) -> non-zero exit, but command itself
    # must run successfully (no crash).
    assert rc_gate == 1
