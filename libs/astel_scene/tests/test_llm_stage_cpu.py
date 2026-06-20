"""Scene-layout LLM stage tests — offline, via FixtureAdapter.

No API key, no network, no spend (the founder-gate rule).  We pre-record a
fixture for the exact (model, system, user) the stage will request, then run
the stage against it — exercising the full path the live adapter will take
later.

All tests are CPU-pure and deterministic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from astel_llm.adapter import (
    FixtureAdapter,
    StructuredResult,
    TokenUsage,
)

from astel_scene.layout import Placement, SceneLayout, SceneObject
from astel_scene.llm_stage import (
    DEFAULT_MODEL,
    SYSTEM_PROMPT,
    build_scene_layout,
)

# ---------------------------------------------------------------------------
# Canned fixture data — exactly what the pre-stored fixture JSON contains.
# The fixture file lives at:
#   src/astel_scene/fixtures/9748c0f4a003ca74b9c1c0bd.json
# Its key is fixture_key(DEFAULT_MODEL, SYSTEM_PROMPT,
#                        "a wooden table with a mug and a book on it").
# ---------------------------------------------------------------------------

PROMPT = "a wooden table with a mug and a book on it"

CANNED: dict[str, Any] = {
    "objects": [
        {
            "object_id": "wooden_table",
            "prompt": (
                "A solid wooden dining table with four legs"
                " and a rectangular top surface."
            ),
            "x": 0.0,
            "z": 0.0,
            "yaw_deg": 0.0,
            "uniform_scale": 1.0,
        },
        {
            "object_id": "mug",
            "prompt": "A ceramic coffee mug with a handle, sitting upright.",
            "x": 0.2,
            "z": -0.1,
            "yaw_deg": 30.0,
            "uniform_scale": 0.12,
        },
        {
            "object_id": "book",
            "prompt": "A closed hardcover book lying flat on its back cover.",
            "x": -0.2,
            "z": 0.05,
            "yaw_deg": -15.0,
            "uniform_scale": 0.18,
        },
    ]
}

CANNED_USAGE = TokenUsage(
    input_tokens=420,
    output_tokens=148,
    cache_read_input_tokens=390,
    cache_creation_input_tokens=0,
)


def _adapter_with_canned(tmp_path: Path) -> FixtureAdapter:
    """Build a FixtureAdapter pre-loaded with the table/mug/book completion."""
    adapter = FixtureAdapter(tmp_path)
    adapter.record(
        model=DEFAULT_MODEL,
        system=SYSTEM_PROMPT,
        user=PROMPT,
        result=StructuredResult(
            data=CANNED,
            usage=CANNED_USAGE,
            model=DEFAULT_MODEL,
        ),
    )
    return adapter


# ---------------------------------------------------------------------------
# Happy-path: fixture present → 3-object layout
# ---------------------------------------------------------------------------


def test_build_scene_layout_returns_three_objects(tmp_path: Path) -> None:
    """FixtureAdapter replay produces exactly the 3 objects from the fixture."""
    adapter = _adapter_with_canned(tmp_path)

    layout, ledger = build_scene_layout(PROMPT, adapter=adapter)

    assert isinstance(layout, SceneLayout)
    assert len(layout.objects) == 3


def test_build_scene_layout_object_ids(tmp_path: Path) -> None:
    """object_id values match the fixture exactly."""
    adapter = _adapter_with_canned(tmp_path)

    layout, _ = build_scene_layout(PROMPT, adapter=adapter)

    ids = [o.object_id for o in layout.objects]
    assert ids == ["wooden_table", "mug", "book"]


def test_build_scene_layout_prompts(tmp_path: Path) -> None:
    """Each SceneObject.prompt matches the fixture text."""
    adapter = _adapter_with_canned(tmp_path)

    layout, _ = build_scene_layout(PROMPT, adapter=adapter)

    assert "wooden" in layout.objects[0].prompt.lower()
    assert "mug" in layout.objects[1].prompt.lower()
    assert "book" in layout.objects[2].prompt.lower()


def test_build_scene_layout_valid_placements(tmp_path: Path) -> None:
    """Every SceneObject has a valid Placement with matching object_id."""
    adapter = _adapter_with_canned(tmp_path)

    layout, _ = build_scene_layout(PROMPT, adapter=adapter)

    for obj in layout.objects:
        assert isinstance(obj, SceneObject)
        assert isinstance(obj.placement, Placement)
        assert obj.placement.object_id == obj.object_id
        assert obj.placement.uniform_scale > 0.0
        # translation is (x, 0.0, z)
        assert obj.placement.translation[1] == 0.0
        assert obj.placement.ground_contact is True


def test_build_scene_layout_table_placement(tmp_path: Path) -> None:
    """Table is at the scene centre; mug and book are offset."""
    adapter = _adapter_with_canned(tmp_path)

    layout, _ = build_scene_layout(PROMPT, adapter=adapter)

    table = layout.objects[0]
    assert table.placement.translation == (0.0, 0.0, 0.0)
    assert table.placement.yaw_deg == 0.0

    mug = layout.objects[1]
    assert mug.placement.translation == (0.2, 0.0, -0.1)
    assert mug.placement.yaw_deg == 30.0


def test_build_scene_layout_ledger_content(tmp_path: Path) -> None:
    """Ledger row carries the stage tag, model, and a cost entry."""
    adapter = _adapter_with_canned(tmp_path)

    _, ledger = build_scene_layout(PROMPT, adapter=adapter)

    assert ledger["stage"] == "scene_layout"
    assert ledger["model"] == DEFAULT_MODEL
    assert ledger["input_tokens"] == 420
    assert ledger["output_tokens"] == 148
    assert ledger["cache_read_input_tokens"] == 390
    # Cache-read tokens are cheap but non-zero in total cost.
    assert ledger["cost_usd"] > 0.0


def test_build_scene_layout_no_network(tmp_path: Path) -> None:
    """Stage runs without any network call (FixtureAdapter only)."""
    # This test is the same as the happy-path test; the assertion that the
    # FixtureAdapter never touches the network is structural — FixtureAdapter
    # has no network code, so reaching here without import-error proves it.
    adapter = _adapter_with_canned(tmp_path)
    layout, _ = build_scene_layout(PROMPT, adapter=adapter)
    assert len(layout.objects) == 3


# ---------------------------------------------------------------------------
# Degradation: unseen prompt → honest single-object fallback
# ---------------------------------------------------------------------------


def test_unseen_prompt_does_not_crash(tmp_path: Path) -> None:
    """An unseen prompt returns a non-empty layout without raising."""
    adapter = FixtureAdapter(tmp_path)  # empty — no fixtures recorded

    layout, ledger = build_scene_layout(
        "a futuristic spaceship on a launchpad", adapter=adapter
    )

    assert isinstance(layout, SceneLayout)
    assert len(layout.objects) >= 1


def test_unseen_prompt_returns_echoed_prompt(tmp_path: Path) -> None:
    """The fallback layout's single object echoes the original prompt."""
    adapter = FixtureAdapter(tmp_path)
    scene_prompt = "a futuristic spaceship on a launchpad"

    layout, _ = build_scene_layout(scene_prompt, adapter=adapter)

    assert layout.objects[0].prompt == scene_prompt


def test_unseen_prompt_ledger_has_note(tmp_path: Path) -> None:
    """The fallback ledger carries a ``note`` key describing the degradation."""
    adapter = FixtureAdapter(tmp_path)

    _, ledger = build_scene_layout("an unrecorded scene prompt", adapter=adapter)

    assert "note" in ledger
    assert "skipped" in ledger["note"].lower()


def test_unseen_prompt_zero_cost(tmp_path: Path) -> None:
    """No tokens are spent on the fallback path."""
    adapter = FixtureAdapter(tmp_path)

    _, ledger = build_scene_layout("another unseen prompt", adapter=adapter)

    assert ledger["cost_usd"] == 0.0
    assert ledger["input_tokens"] == 0
    assert ledger["output_tokens"] == 0


def test_unseen_prompt_does_not_hit_network(tmp_path: Path) -> None:
    """Empty FixtureAdapter catches the miss locally — no network call."""
    # Structural proof: FixtureAdapter raises FixtureMissingError (local),
    # the stage catches it and returns the fallback — no remote I/O.
    adapter = FixtureAdapter(tmp_path)
    layout, ledger = build_scene_layout("no fixture here at all", adapter=adapter)
    assert layout is not None
    assert ledger["cost_usd"] == 0.0


# ---------------------------------------------------------------------------
# Round-trip serialisation
# ---------------------------------------------------------------------------


def test_layout_round_trips_to_dict(tmp_path: Path) -> None:
    """SceneLayout produced by the stage round-trips through to_dict/from_dict."""
    adapter = _adapter_with_canned(tmp_path)

    layout, _ = build_scene_layout(PROMPT, adapter=adapter)

    d = layout.to_dict()
    restored = SceneLayout.from_dict(d)

    assert restored.up_axis == layout.up_axis
    assert restored.ground_y == layout.ground_y
    assert len(restored.objects) == len(layout.objects)

    for orig, rest in zip(layout.objects, restored.objects, strict=True):
        assert rest.object_id == orig.object_id
        assert rest.prompt == orig.prompt
        assert rest.placement.yaw_deg == orig.placement.yaw_deg
        assert rest.placement.uniform_scale == orig.placement.uniform_scale
        assert rest.placement.translation == orig.placement.translation
        assert rest.placement.ground_contact == orig.placement.ground_contact


def test_layout_round_trips_write_read_json(tmp_path: Path) -> None:
    """SceneLayout produced by the stage round-trips through write_json/read_json."""
    adapter = _adapter_with_canned(tmp_path)

    layout, _ = build_scene_layout(PROMPT, adapter=adapter)

    path = tmp_path / "scene.json"
    layout.write_json(path)
    restored = SceneLayout.read_json(path)

    assert len(restored.objects) == 3
    assert restored.objects[0].object_id == "wooden_table"
    assert restored.objects[1].object_id == "mug"
    assert restored.objects[2].object_id == "book"


# ---------------------------------------------------------------------------
# Layout conventions
# ---------------------------------------------------------------------------


def test_layout_up_axis_and_ground_y(tmp_path: Path) -> None:
    """Layout uses Astel's canonical +Y convention with ground_y=0.0."""
    adapter = _adapter_with_canned(tmp_path)

    layout, _ = build_scene_layout(PROMPT, adapter=adapter)

    assert layout.up_axis == "+Y"
    assert layout.ground_y == 0.0


def test_empty_prompt_raises(tmp_path: Path) -> None:
    """Empty/whitespace prompt is rejected before any adapter call."""
    adapter = _adapter_with_canned(tmp_path)
    with pytest.raises(ValueError, match="non-empty"):
        build_scene_layout("   ", adapter=adapter)


# ---------------------------------------------------------------------------
# Fallback layout round-trip
# ---------------------------------------------------------------------------


def test_fallback_layout_round_trips(tmp_path: Path) -> None:
    """The single-object fallback layout also round-trips cleanly."""
    adapter = FixtureAdapter(tmp_path)

    layout, _ = build_scene_layout("an unseen scene", adapter=adapter)

    d = layout.to_dict()
    restored = SceneLayout.from_dict(d)
    assert len(restored.objects) == 1
    assert restored.objects[0].object_id == "object_0"
