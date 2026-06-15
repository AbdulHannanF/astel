"""Astel model-agnostic LLM layer (CLAUDE.md §5).

Public surface: the adapter interface + implementations, the Generation Spec
stage, and the token-cost ledger helpers.
"""

from __future__ import annotations

from .adapter import (
    AnthropicAdapter,
    FixtureAdapter,
    FixtureMissingError,
    LLMAdapter,
    StructuredResult,
    TokenUsage,
    fixture_key,
)
from .generation_spec import (
    DEFAULT_MODEL,
    SYSTEM_PROMPT,
    GenerationSpecResult,
    build_generation_spec,
)
from .physics_material import (
    JOINT_TYPES,
    MATERIAL_CLASSES,
    ArticulationHint,
    PhysicsMaterialResult,
    PhysicsMaterialSpec,
    RegionMaterial,
    build_physics_material_spec,
)
from .pricing import RATES, estimate_cost_usd, ledger_entry
from .spec import GenerationSpec, Part, TargetScale

__all__ = [
    "DEFAULT_MODEL",
    "JOINT_TYPES",
    "MATERIAL_CLASSES",
    "RATES",
    "SYSTEM_PROMPT",
    "AnthropicAdapter",
    "ArticulationHint",
    "FixtureAdapter",
    "FixtureMissingError",
    "GenerationSpec",
    "GenerationSpecResult",
    "LLMAdapter",
    "Part",
    "PhysicsMaterialResult",
    "PhysicsMaterialSpec",
    "RegionMaterial",
    "StructuredResult",
    "TargetScale",
    "TokenUsage",
    "build_generation_spec",
    "build_physics_material_spec",
    "estimate_cost_usd",
    "fixture_key",
    "ledger_entry",
]
