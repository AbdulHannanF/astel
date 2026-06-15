"""Tests for GenerationSpec parsing/validation and schema constraints."""

from __future__ import annotations

from typing import Any

import pytest

from astel_llm.spec import GenerationSpec

VALID: dict[str, Any] = {
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


def test_from_dict_roundtrip() -> None:
    spec = GenerationSpec.from_dict(VALID)
    assert spec.object_class == "claw hammer"
    assert len(spec.parts) == 2
    assert spec.parts[1].material == "wood"
    assert spec.target_scale.longest_axis_m == 0.33
    assert spec.symmetry == "bilateral"


def test_rejects_confidence_out_of_range() -> None:
    bad = {**VALID, "target_scale": {**VALID["target_scale"], "confidence": 1.5}}
    with pytest.raises(ValueError, match="confidence"):
        GenerationSpec.from_dict(bad)


def test_rejects_inverted_scale_interval() -> None:
    bad = {
        **VALID,
        "target_scale": {
            "longest_axis_m": 0.5,
            "confidence": 0.5,
            "low_m": 0.6,  # low > longest
            "high_m": 0.7,
        },
    }
    with pytest.raises(ValueError, match="low <= longest <= high"):
        GenerationSpec.from_dict(bad)


def test_rejects_bad_symmetry() -> None:
    with pytest.raises(ValueError, match="symmetry"):
        GenerationSpec.from_dict({**VALID, "symmetry": "spiral"})


def test_schema_is_anthropic_structured_output_compatible() -> None:
    """Every object must set additionalProperties:false and use no numeric/length
    constraints (unsupported by Anthropic structured outputs)."""
    schema = GenerationSpec.json_schema()
    banned = {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
              "minLength", "maxLength", "minItems", "maxItems", "multipleOf"}

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
