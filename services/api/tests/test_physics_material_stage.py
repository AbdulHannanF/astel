"""Tests for the API-side L6 physics-material stage (offline, founder-gated live).

All offline: the live AnthropicAdapter is never constructed. Covers (a) non-text
/ no-spec is a no-op, (b) a fixture cache-miss degrades to a non-billable
"skipped" note, (c) a recorded fixture yields the billable ``l6.json`` layer with
per-region materials + a ledger row.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "ASTEL_DATABASE_URL", "sqlite+aiosqlite:///./astel_test_l6_stage.db"
)

from astel_llm import (  # noqa: E402
    DEFAULT_MODEL,
    FixtureAdapter,
    GenerationSpec,
    StructuredResult,
    TokenUsage,
)
from astel_llm.physics_material import (  # noqa: E402
    SYSTEM_PROMPT,
    _format_user,
)

from astel_api.config import Settings  # noqa: E402
from astel_api.physics_material_stage import run_physics_material_stage  # noqa: E402
from astel_api.storage import LocalArtifactStore  # noqa: E402

_SPEC_DATA = {
    "object_class": "claw hammer",
    "summary": "A steel-headed claw hammer with a wooden handle.",
    "parts": [
        {"name": "head", "material": "steel"},
        {"name": "handle", "material": "wood"},
    ],
    "materials": ["steel", "wood"],
    "style": "utilitarian",
    "target_scale": {
        "longest_axis_m": 0.33,
        "confidence": 0.7,
        "low_m": 0.28,
        "high_m": 0.40,
    },
    "symmetry": "bilateral",
}

_L6_DATA: dict[str, Any] = {
    "regions": [
        {
            "region": "head",
            "material": "mild steel",
            "material_class": "rigid",
            "density_kg_m3": 7850.0,
            "friction": 0.6,
            "restitution": 0.4,
        },
        {
            "region": "handle",
            "material": "oak wood",
            "material_class": "rigid",
            "density_kg_m3": 720.0,
            "friction": 0.5,
            "restitution": 0.3,
        },
    ],
    "articulation": [{"parent": "handle", "child": "head", "joint_type": "fixed"}],
    "notes": "Typical bulk densities; exact alloy/species unknown.",
}


def _ok_spec_payload() -> dict[str, Any]:
    return {"status": "ok", "mode": "fixture", "spec": _SPEC_DATA, "ledger": {}}


def _settings(fixtures_dir: Path) -> Settings:
    return Settings(llm_fixtures_dir=fixtures_dir, llm_live=False)


def _read_json(store: LocalArtifactStore, task_id: str, name: str) -> dict[str, Any]:
    path = store.path_for(task_id, name)
    assert path is not None, f"missing artifact {name}"
    data: dict[str, Any] = json.loads(path.read_text())
    return data


def _record_fixture(fixtures_dir: Path) -> None:
    spec = GenerationSpec.from_dict(_SPEC_DATA)
    adapter = FixtureAdapter(fixtures_dir)
    adapter.record(
        model=DEFAULT_MODEL,
        system=SYSTEM_PROMPT,
        user=_format_user(spec),
        result=StructuredResult(
            data=_L6_DATA,
            usage=TokenUsage(input_tokens=400, output_tokens=160),
            model=DEFAULT_MODEL,
        ),
    )


def test_noop_when_no_spec(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "store")
    out = run_physics_material_stage(
        "t1", "text", None, store, _settings(tmp_path / "fx")
    )
    assert out is None
    assert store.list_names("t1") == []


def test_noop_for_image_modality(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "store")
    out = run_physics_material_stage(
        "t2", "image", _ok_spec_payload(), store, _settings(tmp_path / "fx")
    )
    assert out is None
    assert store.list_names("t2") == []


def test_cache_miss_degrades_to_skipped_nonbillable(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "store")
    out = run_physics_material_stage(
        "t3", "text", _ok_spec_payload(), store, _settings(tmp_path / "fx")
    )
    assert out is not None and out["status"] == "skipped"
    # The skip note must NOT be the billable l6.json (would charge L6 for nothing).
    names = store.list_names("t3")
    assert "physics-material.json" in names
    assert "l6.json" not in names


def test_recorded_fixture_yields_billable_l6(tmp_path: Path) -> None:
    fixtures = tmp_path / "fx"
    _record_fixture(fixtures)
    store = LocalArtifactStore(tmp_path / "store")

    out = run_physics_material_stage(
        "t4", "text", _ok_spec_payload(), store, _settings(fixtures)
    )

    assert out is not None and out["status"] == "ok"
    assert "l6.json" in store.list_names("t4")
    stored = _read_json(store, "t4", "l6.json")
    assert stored["schema"] == "astel.physics-material/v0"
    regions = stored["spec"]["regions"]
    assert len(regions) == 2
    assert regions[0]["region"] == "head"
    # The steel head is far denser than the wooden handle.
    assert regions[0]["density_kg_m3"] > regions[1]["density_kg_m3"]
    assert stored["spec"]["articulation"][0]["joint_type"] == "fixed"
    assert stored["ledger"]["stage"] == "physics_material"
    assert stored["ledger"]["cost_usd"] > 0.0
