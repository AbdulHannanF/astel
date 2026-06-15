"""The Generation Spec — the structured object the text pipeline conditions on.

CLAUDE.md §4: the prompt parser LLM produces a structured Generation Spec —
object class, parts, materials, style, target scale (with an explicit confidence
the user can override), and symmetry. This module is the typed Python shape plus
the JSON schema we hand to the LLM's structured-output mode.

The schema is written to Anthropic's structured-output constraints (verified
2026-06-15): every object carries ``additionalProperties: false`` and there are
NO numeric/length constraints (``minimum``/``maxLength``/etc. are unsupported) —
we validate ranges in :func:`GenerationSpec.from_dict` instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

SYMMETRY_VALUES = ("none", "bilateral", "radial", "axial")


@dataclass(frozen=True)
class Part:
    """A named sub-part of the object and its dominant material."""

    name: str
    material: str


@dataclass(frozen=True)
class TargetScale:
    """Metric size estimate with an explicit, user-overridable confidence band.

    ``longest_axis_m`` is the point estimate (metres); ``low_m``/``high_m`` bound
    the interval; ``confidence`` in ``[0, 1]`` is how sure the estimator is. This
    is the L1 "scale grounded with explicit confidence interval the user can
    override" from CLAUDE.md §3 — never silently presented as exact.
    """

    longest_axis_m: float
    confidence: float
    low_m: float
    high_m: float


@dataclass(frozen=True)
class GenerationSpec:
    """Structured parse of a text prompt — the L0/L2 conditioning input."""

    object_class: str
    summary: str
    parts: tuple[Part, ...]
    materials: tuple[str, ...]
    style: str
    target_scale: TargetScale
    symmetry: str

    @staticmethod
    def json_schema() -> dict[str, Any]:
        """The structured-output JSON schema (Anthropic-compatible)."""
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "object_class": {"type": "string"},
                "summary": {"type": "string"},
                "parts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "name": {"type": "string"},
                            "material": {"type": "string"},
                        },
                        "required": ["name", "material"],
                    },
                },
                "materials": {"type": "array", "items": {"type": "string"}},
                "style": {"type": "string"},
                "target_scale": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "longest_axis_m": {"type": "number"},
                        "confidence": {"type": "number"},
                        "low_m": {"type": "number"},
                        "high_m": {"type": "number"},
                    },
                    "required": ["longest_axis_m", "confidence", "low_m", "high_m"],
                },
                "symmetry": {"type": "string", "enum": list(SYMMETRY_VALUES)},
            },
            "required": [
                "object_class",
                "summary",
                "parts",
                "materials",
                "style",
                "target_scale",
                "symmetry",
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GenerationSpec:
        """Validate + parse a structured-output payload into a GenerationSpec.

        Enforces the range/enum invariants the JSON schema can't express
        (confidence in ``[0, 1]``, ``low <= longest <= high``, valid symmetry).
        """
        ts = data["target_scale"]
        scale = TargetScale(
            longest_axis_m=float(ts["longest_axis_m"]),
            confidence=float(ts["confidence"]),
            low_m=float(ts["low_m"]),
            high_m=float(ts["high_m"]),
        )
        if not 0.0 <= scale.confidence <= 1.0:
            raise ValueError(f"confidence out of [0,1]: {scale.confidence}")
        if not scale.low_m <= scale.longest_axis_m <= scale.high_m:
            raise ValueError(
                "scale interval must satisfy low <= longest <= high: "
                f"{scale.low_m} <= {scale.longest_axis_m} <= {scale.high_m}"
            )
        symmetry = str(data["symmetry"])
        if symmetry not in SYMMETRY_VALUES:
            raise ValueError(f"symmetry must be one of {SYMMETRY_VALUES}: {symmetry}")
        return cls(
            object_class=str(data["object_class"]),
            summary=str(data["summary"]),
            parts=tuple(
                Part(name=str(p["name"]), material=str(p["material"]))
                for p in data["parts"]
            ),
            materials=tuple(str(m) for m in data["materials"]),
            style=str(data["style"]),
            target_scale=scale,
            symmetry=symmetry,
        )
