"""Tests for the API-side Generation Spec stage (offline, founder-gated live).

All offline: the live AnthropicAdapter is never constructed. Covers (a) non-text
modality is a no-op, (b) a fixture cache-miss degrades to an honest "skipped"
note, (c) a recorded fixture yields a stored spec AND patches the quality report
with the LLM size estimate.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "ASTEL_DATABASE_URL", "sqlite+aiosqlite:///./astel_test_spec_stage.db"
)

from astel_llm import (  # noqa: E402
    DEFAULT_MODEL,
    SYSTEM_PROMPT,
    FixtureAdapter,
    StructuredResult,
    TokenUsage,
)

from astel_api.config import Settings  # noqa: E402
from astel_api.generation_spec_stage import (  # noqa: E402
    apply_spec_scale_to_report,
    run_generation_spec_stage,
)
from astel_api.storage import LocalArtifactStore  # noqa: E402

_SPEC_DATA = {
    "object_class": "teapot",
    "summary": "A small ceramic teapot with a curved spout.",
    "parts": [
        {"name": "body", "material": "ceramic"},
        {"name": "handle", "material": "ceramic"},
    ],
    "materials": ["ceramic"],
    "style": "modern",
    "target_scale": {
        "longest_axis_m": 0.22,
        "confidence": 0.6,
        "low_m": 0.15,
        "high_m": 0.30,
    },
    "symmetry": "bilateral",
}


def _settings(fixtures_dir: Path) -> Settings:
    return Settings(llm_fixtures_dir=fixtures_dir, llm_live=False)


def _read_json(store: LocalArtifactStore, task_id: str, name: str) -> dict[str, Any]:
    path = store.path_for(task_id, name)
    assert path is not None, f"missing artifact {name}"
    data: dict[str, Any] = json.loads(path.read_text())
    return data


def _record_fixture(fixtures_dir: Path, prompt: str) -> None:
    adapter = FixtureAdapter(fixtures_dir)
    adapter.record(
        model=DEFAULT_MODEL,
        system=SYSTEM_PROMPT,
        user=prompt.strip(),
        result=StructuredResult(
            data=_SPEC_DATA,
            usage=TokenUsage(input_tokens=350, output_tokens=120),
            model=DEFAULT_MODEL,
        ),
    )


def test_non_text_modality_is_noop(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "store")
    out = run_generation_spec_stage(
        "t1", "image", "a teapot", store, _settings(tmp_path / "fx")
    )
    assert out is None
    assert "generation-spec.json" not in store.list_names("t1")


def test_cache_miss_degrades_to_skipped(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "store")
    out = run_generation_spec_stage(
        "t2", "text", "an unseen prompt", store, _settings(tmp_path / "fx")
    )
    assert out is not None and out["status"] == "skipped"
    assert out["mode"] == "fixture"
    stored = _read_json(store, "t2", "generation-spec.json")
    assert stored["status"] == "skipped"
    assert "R-O2" in stored["reason"]


def test_recorded_fixture_yields_spec_and_ledger(tmp_path: Path) -> None:
    fixtures = tmp_path / "fx"
    prompt = "a small ceramic teapot"
    _record_fixture(fixtures, prompt)
    store = LocalArtifactStore(tmp_path / "store")

    out = run_generation_spec_stage("t3", "text", prompt, store, _settings(fixtures))

    assert out is not None and out["status"] == "ok"
    assert out["spec"]["object_class"] == "teapot"
    assert out["ledger"]["stage"] == "generation_spec"
    assert out["ledger"]["cost_usd"] > 0.0
    stored = _read_json(store, "t3", "generation-spec.json")
    assert stored["spec"]["symmetry"] == "bilateral"


def test_apply_spec_scale_patches_report(tmp_path: Path) -> None:
    fixtures = tmp_path / "fx"
    prompt = "a small ceramic teapot"
    _record_fixture(fixtures, prompt)
    store = LocalArtifactStore(tmp_path / "store")
    # Simulate the producer having written a report with an unknown scale.
    store.put(
        "t4",
        "quality-report.json",
        json.dumps({"schema": "astel.quality-report/v0", "scale": None}).encode(),
    )

    out = run_generation_spec_stage("t4", "text", prompt, store, _settings(fixtures))
    apply_spec_scale_to_report("t4", store, out)

    report = _read_json(store, "t4", "quality-report.json")
    assert report["scale"]["longest_axis_m"] == 0.22
    assert report["scale"]["method"] == "llm-estimate"
    assert report["scale"]["source"] == "generation-spec"
    assert report["scale"]["low_m"] == 0.15


def test_apply_spec_scale_noop_when_skipped(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "store")
    store.put("t5", "quality-report.json", json.dumps({"scale": "orig"}).encode())
    skipped = {"status": "skipped", "mode": "fixture"}
    apply_spec_scale_to_report("t5", store, skipped)
    report = _read_json(store, "t5", "quality-report.json")
    assert report["scale"] == "orig"  # untouched
