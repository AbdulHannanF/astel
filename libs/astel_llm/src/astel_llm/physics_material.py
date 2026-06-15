"""The L6 physics-material & semantic stage (CLAUDE.md §3 L6, §8.2).

Reasons over an object's :class:`~astel_llm.spec.GenerationSpec` (its parts +
materials) and assigns each region a *physical* material: a simulation behaviour
class, a density (kg/m³), and friction/restitution defaults — plus articulation
hints (which parts are separable and how they'd be jointed). This is the layer
that gives an asset **correct mass in engines** (density × the L5 solid volume),
a **PhysGaussian-style MPM material** per region, and **articulation hints** for
rigging.

Same shape as the Generation Spec stage: a typed spec + an Anthropic-compatible
structured-output schema + a stage function that runs through any
:class:`LLMAdapter`. It runs **entirely offline** with :class:`FixtureAdapter`
(no API key, no spend — founder gate R-O2); swap in :class:`AnthropicAdapter`
unchanged once a key is available.

Default model is Haiku 4.5 (this is constrained material lookup, not deep
reasoning); the research note (docs/research/13-m3-readiness §3) reserves Sonnet
4.6 for this stage only if Haiku underperforms — pass ``model=`` to upgrade.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .adapter import LLMAdapter
from .generation_spec import DEFAULT_MODEL
from .pricing import ledger_entry
from .spec import GenerationSpec

#: MPM/sim behaviour classes a region can be assigned (PhysGaussian-adjacent).
MATERIAL_CLASSES = ("rigid", "soft", "cloth", "fluid_adjacent", "granular")

#: Joint kinds for articulation hints between separable parts.
JOINT_TYPES = ("fixed", "hinge", "slider", "ball", "free")


@dataclass(frozen=True)
class RegionMaterial:
    """The physical material assigned to one region (≈ a Generation-Spec part).

    ``density_kg_m3`` × the region's L5 solid volume gives its mass; ``friction``
    is a dynamic friction coefficient (≥ 0, typically ~0.1–1.2) and
    ``restitution`` is bounciness in ``[0, 1]``. ``material_class`` selects the
    simulation behaviour (rigid body vs. MPM soft/cloth/granular/fluid).
    """

    region: str
    material: str
    material_class: str
    density_kg_m3: float
    friction: float
    restitution: float


@dataclass(frozen=True)
class ArticulationHint:
    """A detected separable joint between two regions (rigging hint)."""

    parent: str
    child: str
    joint_type: str


@dataclass(frozen=True)
class PhysicsMaterialSpec:
    """Per-region physics materials + articulation hints for an asset (L6)."""

    regions: tuple[RegionMaterial, ...]
    articulation: tuple[ArticulationHint, ...]
    notes: str

    @staticmethod
    def json_schema() -> dict[str, Any]:
        """The structured-output JSON schema (Anthropic-compatible).

        Like the Generation Spec schema, every object sets
        ``additionalProperties: false`` and uses NO numeric/length constraints
        (unsupported by structured outputs) — ranges are enforced in
        :meth:`from_dict` instead. Enums are allowed.
        """
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "regions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "region": {"type": "string"},
                            "material": {"type": "string"},
                            "material_class": {
                                "type": "string",
                                "enum": list(MATERIAL_CLASSES),
                            },
                            "density_kg_m3": {"type": "number"},
                            "friction": {"type": "number"},
                            "restitution": {"type": "number"},
                        },
                        "required": [
                            "region",
                            "material",
                            "material_class",
                            "density_kg_m3",
                            "friction",
                            "restitution",
                        ],
                    },
                },
                "articulation": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "parent": {"type": "string"},
                            "child": {"type": "string"},
                            "joint_type": {
                                "type": "string",
                                "enum": list(JOINT_TYPES),
                            },
                        },
                        "required": ["parent", "child", "joint_type"],
                    },
                },
                "notes": {"type": "string"},
            },
            "required": ["regions", "articulation", "notes"],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PhysicsMaterialSpec:
        """Validate + parse a structured-output payload.

        Enforces the invariants the JSON schema can't express: a known
        ``material_class``/``joint_type``, a positive density, a non-negative
        friction, and a restitution in ``[0, 1]``.
        """
        regions: list[RegionMaterial] = []
        for r in data["regions"]:
            material_class = str(r["material_class"])
            if material_class not in MATERIAL_CLASSES:
                raise ValueError(
                    f"material_class must be one of {MATERIAL_CLASSES}: "
                    f"{material_class}"
                )
            density = float(r["density_kg_m3"])
            if density <= 0.0:
                raise ValueError(f"density_kg_m3 must be positive: {density}")
            friction = float(r["friction"])
            if friction < 0.0:
                raise ValueError(f"friction must be >= 0: {friction}")
            restitution = float(r["restitution"])
            if not 0.0 <= restitution <= 1.0:
                raise ValueError(f"restitution out of [0,1]: {restitution}")
            regions.append(
                RegionMaterial(
                    region=str(r["region"]),
                    material=str(r["material"]),
                    material_class=material_class,
                    density_kg_m3=density,
                    friction=friction,
                    restitution=restitution,
                )
            )
        if not regions:
            raise ValueError("physics-material spec must have at least one region")

        articulation: list[ArticulationHint] = []
        for j in data.get("articulation", []):
            joint_type = str(j["joint_type"])
            if joint_type not in JOINT_TYPES:
                raise ValueError(
                    f"joint_type must be one of {JOINT_TYPES}: {joint_type}"
                )
            articulation.append(
                ArticulationHint(
                    parent=str(j["parent"]),
                    child=str(j["child"]),
                    joint_type=joint_type,
                )
            )
        return cls(
            regions=tuple(regions),
            articulation=tuple(articulation),
            notes=str(data.get("notes", "")),
        )


@dataclass(frozen=True)
class PhysicsMaterialResult:
    """The parsed L6 spec plus the credit-ledger row for the LLM call."""

    spec: PhysicsMaterialSpec
    ledger: dict[str, Any]


#: Frozen so the cached prefix stays byte-identical across generations (the
#: prompt-caching contract — only the per-object user turn varies).
SYSTEM_PROMPT = (
    "You are Astel's physics-material reasoner (asset layer L6). Given a "
    "structured description of a single 3D object and its parts, assign each "
    "region its real-world PHYSICAL material so the asset behaves correctly in "
    "game engines, simulators, and MPM physics. Rules:\n"
    "- regions: one entry per distinct part. For each, give: material (a "
    "concrete material name, e.g. 'oak wood', 'mild steel'); material_class "
    "(one of rigid | soft | cloth | fluid_adjacent | granular — the simulation "
    "behaviour); density_kg_m3 (a realistic bulk density, e.g. wood ~700, steel "
    "~7850, glass ~2500, rubber ~1100); friction (a dynamic friction "
    "coefficient, typically 0.1–1.2); restitution (bounciness in [0,1]).\n"
    "- articulation: hints for separable parts that would be jointed (e.g. a "
    "lid hinged to a box). Each is {parent, child, joint_type} with joint_type "
    "one of fixed | hinge | slider | ball | free. Use an empty list when the "
    "object is a single rigid piece.\n"
    "- notes: one short sentence on any material uncertainty.\n"
    "Use the most likely material when a part is ambiguous; never invent parts "
    "beyond those described. Output ONLY the JSON object."
)


def _format_user(spec: GenerationSpec) -> str:
    """Render a GenerationSpec into a stable, compact user turn.

    Deterministic (so the fixture key is stable) and information-dense: the
    object class, summary, each part with its visual material, and the metric
    size — everything the reasoner needs to assign physical materials.
    """
    parts = (
        "; ".join(f"{p.name}={p.material}" for p in spec.parts)
        if spec.parts
        else "(none specified)"
    )
    return (
        f"Object: {spec.object_class}\n"
        f"Summary: {spec.summary}\n"
        f"Parts: {parts}\n"
        f"Materials: {', '.join(spec.materials) if spec.materials else '(none)'}\n"
        f"Style: {spec.style}\n"
        f"Longest axis: ~{spec.target_scale.longest_axis_m} m"
    )


def build_physics_material_spec(
    spec: GenerationSpec,
    adapter: LLMAdapter,
    *,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 1024,
) -> PhysicsMaterialResult:
    """Assign per-region physics materials for ``spec`` via ``adapter``.

    Returns the validated :class:`PhysicsMaterialSpec` and the credit-ledger row
    (``stage="physics_material"``) for the call.
    """
    result = adapter.complete_structured(
        system=SYSTEM_PROMPT,
        user=_format_user(spec),
        schema=PhysicsMaterialSpec.json_schema(),
        model=model,
        max_tokens=max_tokens,
    )
    pm = PhysicsMaterialSpec.from_dict(result.data)
    ledger = ledger_entry(stage="physics_material", model=model, usage=result.usage)
    return PhysicsMaterialResult(spec=pm, ledger=ledger)
