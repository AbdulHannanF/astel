"""Tests for the L6 physics-material stage and its spec parsing/schema.

Offline only — via FixtureAdapter, no API key, no network, no spend (the
founder-gate rule). We record a fixture for the exact (model, system, user) the
stage will request, then run the stage against it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from astel_llm.adapter import (
    FixtureAdapter,
    FixtureMissingError,
    StructuredResult,
    TokenUsage,
)
from astel_llm.generation_spec import DEFAULT_MODEL
from astel_llm.physics_material import (
    SYSTEM_PROMPT,
    PhysicsMaterialSpec,
    _format_user,
    build_physics_material_spec,
)
from astel_llm.spec import GenerationSpec

SPEC = GenerationSpec.from_dict(
    {
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
)

CANNED: dict[str, Any] = {
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
    "articulation": [
        {"parent": "handle", "child": "head", "joint_type": "fixed"},
    ],
    "notes": "Densities are typical bulk values; exact alloy/species unknown.",
}


def _adapter_with_canned(tmp_path: Path) -> FixtureAdapter:
    adapter = FixtureAdapter(tmp_path)
    adapter.record(
        model=DEFAULT_MODEL,
        system=SYSTEM_PROMPT,
        user=_format_user(SPEC),
        result=StructuredResult(
            data=CANNED,
            usage=TokenUsage(
                input_tokens=1100,
                output_tokens=220,
                cache_read_input_tokens=1000,
            ),
            model=DEFAULT_MODEL,
        ),
    )
    return adapter


def test_build_physics_material_offline(tmp_path: Path) -> None:
    adapter = _adapter_with_canned(tmp_path)

    result = build_physics_material_spec(SPEC, adapter)

    assert len(result.spec.regions) == 2
    head = result.spec.regions[0]
    assert head.region == "head"
    assert head.material_class == "rigid"
    assert head.density_kg_m3 == 7850.0
    # The steel head is far denser than the wooden handle (mass differs in-engine).
    assert head.density_kg_m3 > result.spec.regions[1].density_kg_m3
    assert result.spec.articulation[0].joint_type == "fixed"
    # ledger row carries the L6 stage tag + a real, non-zero cost.
    assert result.ledger["stage"] == "physics_material"
    assert result.ledger["model"] == DEFAULT_MODEL
    assert result.ledger["cost_usd"] > 0.0


def test_missing_fixture_raises(tmp_path: Path) -> None:
    adapter = FixtureAdapter(tmp_path)
    with pytest.raises(FixtureMissingError):
        build_physics_material_spec(SPEC, adapter)


def test_rejects_unknown_material_class() -> None:
    bad = {
        **CANNED,
        "regions": [{**CANNED["regions"][0], "material_class": "plasma"}],
    }
    with pytest.raises(ValueError, match="material_class"):
        PhysicsMaterialSpec.from_dict(bad)


def test_rejects_nonpositive_density() -> None:
    bad = {
        **CANNED,
        "regions": [{**CANNED["regions"][0], "density_kg_m3": 0.0}],
    }
    with pytest.raises(ValueError, match="density"):
        PhysicsMaterialSpec.from_dict(bad)


def test_rejects_restitution_out_of_range() -> None:
    bad = {
        **CANNED,
        "regions": [{**CANNED["regions"][0], "restitution": 1.5}],
    }
    with pytest.raises(ValueError, match="restitution"):
        PhysicsMaterialSpec.from_dict(bad)


def test_rejects_negative_friction() -> None:
    bad = {
        **CANNED,
        "regions": [{**CANNED["regions"][0], "friction": -0.1}],
    }
    with pytest.raises(ValueError, match="friction"):
        PhysicsMaterialSpec.from_dict(bad)


def test_rejects_empty_regions() -> None:
    with pytest.raises(ValueError, match="at least one region"):
        PhysicsMaterialSpec.from_dict({"regions": [], "articulation": [], "notes": ""})


def test_rejects_bad_joint_type() -> None:
    bad = {
        **CANNED,
        "articulation": [{"parent": "a", "child": "b", "joint_type": "weld"}],
    }
    with pytest.raises(ValueError, match="joint_type"):
        PhysicsMaterialSpec.from_dict(bad)


def test_articulation_defaults_to_empty() -> None:
    spec = PhysicsMaterialSpec.from_dict(
        {"regions": CANNED["regions"], "notes": "single piece"}
    )
    assert spec.articulation == ()


def test_schema_is_anthropic_structured_output_compatible() -> None:
    """Every object sets additionalProperties:false and uses no numeric/length
    constraints (unsupported by Anthropic structured outputs)."""
    schema = PhysicsMaterialSpec.json_schema()
    banned = {
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
        "multipleOf",
    }

    def walk(node: object) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert node.get("additionalProperties") is False, node
            assert banned.isdisjoint(node.keys()), node.keys() & banned
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(schema)
