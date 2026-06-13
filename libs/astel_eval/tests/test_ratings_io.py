"""CSV/JSON ratings I/O round-trip tests."""

from __future__ import annotations

from pathlib import Path

from astel_eval.ratings_io import load_likert_csv, load_pairwise, load_pairwise_csv
from astel_eval.scoring_models import NotApplicable


def test_load_likert_csv_with_na(tmp_path: Path) -> None:
    csv_path = tmp_path / "likert.csv"
    csv_path.write_text(
        "case_id,system,rater_id,axis,score\n"
        "T01,astel,r0,geometry_fidelity,4\n"
        "T01,astel,r0,metric_scale_accuracy,N/A\n",
        encoding="utf-8",
    )
    ratings = load_likert_csv(csv_path)
    assert len(ratings) == 2
    assert ratings[0].score == 4
    assert ratings[1].score is NotApplicable.NOT_APPLICABLE


def test_load_pairwise_csv_with_tie(tmp_path: Path) -> None:
    csv_path = tmp_path / "pairwise.csv"
    csv_path.write_text(
        "case_id,rater_id,axis,system_a,system_b,winner\n"
        "T01,r0,overall,astel,trellis2,astel\n"
        "T02,r0,overall,astel,meshy_free,\n",
        encoding="utf-8",
    )
    prefs = load_pairwise_csv(csv_path)
    assert len(prefs) == 2
    assert prefs[0].winner == "astel"
    assert prefs[1].winner is None


def test_load_pairwise_dispatches_on_extension(tmp_path: Path) -> None:
    json_path = tmp_path / "pairwise.json"
    json_path.write_text(
        '{"pairwise": [{"case_id": "T01", "rater_id": "r0", "axis": "overall", '
        '"system_a": "astel", "system_b": "trellis2", "winner": "astel"}]}',
        encoding="utf-8",
    )
    prefs = load_pairwise(json_path)
    assert len(prefs) == 1
    assert prefs[0].case_id == "T01"
