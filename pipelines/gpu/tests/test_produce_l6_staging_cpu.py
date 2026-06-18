"""CPU tests for the producer's l6.json staging seam (no gsplat/CUDA)."""

from __future__ import annotations

from pathlib import Path

from astel_gpu.produce import _stage_l6_json


def test_stage_l6_json_copies_into_out_dir(tmp_path: Path) -> None:
    src = tmp_path / "store" / "l6.json"
    src.parent.mkdir(parents=True)
    src.write_text('{"spec": {"regions": []}}')
    out = tmp_path / "out"
    out.mkdir()

    _stage_l6_json(src, out)

    staged = out / "l6.json"
    assert staged.is_file()
    assert staged.read_text() == '{"spec": {"regions": []}}'


def test_stage_l6_json_noop_when_none(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    _stage_l6_json(None, out)
    assert not (out / "l6.json").exists()


def test_stage_l6_json_noop_when_missing(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    _stage_l6_json(tmp_path / "does-not-exist.json", out)
    assert not (out / "l6.json").exists()
