"""Scene-layout LLM stage (CLAUDE.md §8 — scene seeds).

Turns a free-text scene prompt into a :class:`~astel_scene.layout.SceneLayout`
via any :class:`~astel_llm.adapter.LLMAdapter`.  Defaults to
:class:`~astel_llm.adapter.FixtureAdapter` pointed at the ``fixtures/``
sub-directory next to this module — meaning it runs **entirely offline, with no
API key and zero spend**, exactly like the Generation Spec and L6 stages.

Usage::

    from astel_scene.llm_stage import build_scene_layout

    layout, ledger = build_scene_layout(
        "a wooden table with a mug and a book on it"
    )
    # layout.objects has 3 SceneObjects; ledger carries the credit row.

Swap in :class:`~astel_llm.adapter.AnthropicAdapter` once an API key is
available — the stage code is unchanged either way.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from astel_llm.adapter import FixtureAdapter, FixtureMissingError, LLMAdapter
from astel_llm.pricing import ledger_entry

from .layout import Placement, SceneLayout, SceneObject

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default model (cheap extraction, not deep reasoning — Haiku-tier).
DEFAULT_MODEL = "claude-haiku-4-5"

#: Directory of pre-recorded fixture completions, relative to this file.
#: Tests that want to exercise the happy path pre-populate this directory
#: via :meth:`~astel_llm.adapter.FixtureAdapter.record`.
_FIXTURES_DIR = Path(__file__).parent / "fixtures"

#: Frozen system prompt.  Kept byte-identical across calls so that the
#: fixture-key hash is stable and prompt-caching can apply on live calls.
SYSTEM_PROMPT = (
    "You are Astel's scene-layout planner. Given a free-text description of a "
    "small multi-object 3D scene, decompose it into a JSON array of objects. "
    "Rules:\n"
    "- Return a JSON object with a single key \"objects\" whose value is an "
    "array.\n"
    "- Each element must have: object_id (a short snake_case identifier, "
    "unique in the scene), prompt (a single-sentence description of just that "
    "object, suitable for single-object generation), x (ground-plane X "
    "position in metres, float), z (ground-plane Z position in metres, float), "
    "yaw_deg (rotation about +Y in degrees, float), uniform_scale (relative "
    "size scale factor, float, typically 0.5–3.0).\n"
    "- Place objects so they do not overlap; the scene centre is (0, 0).\n"
    "- Never invent objects the prompt does not imply.\n"
    "- If the prompt describes only one object, return exactly one element.\n"
    "Output ONLY the JSON object."
)

#: JSON schema for the structured-output call.  Every object has
#: ``additionalProperties: false`` and no numeric/length constraints
#: (Anthropic structured-output requirement — ranges enforced in
#: :func:`_parse_objects` instead).
_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "objects": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "object_id": {"type": "string"},
                    "prompt": {"type": "string"},
                    "x": {"type": "number"},
                    "z": {"type": "number"},
                    "yaw_deg": {"type": "number"},
                    "uniform_scale": {"type": "number"},
                },
                "required": ["object_id", "prompt", "x", "z"],
            },
        }
    },
    "required": ["objects"],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_objects(raw: dict[str, Any]) -> list[SceneObject]:
    """Convert the structured-output dict into :class:`SceneObject` instances.

    Defensive: missing optional fields (``yaw_deg``, ``uniform_scale``) get
    sensible defaults; empty ``object_id`` or ``prompt`` is filled from the
    index.  Never fabricates object count — uses exactly what ``raw`` contains.
    """
    items = raw.get("objects", [])
    result: list[SceneObject] = []
    for i, item in enumerate(items):
        oid = str(item.get("object_id") or f"object_{i}").strip() or f"object_{i}"
        prompt = str(item.get("prompt") or f"object {i}").strip() or f"object {i}"
        x = float(item.get("x", 0.0))
        z = float(item.get("z", 0.0))
        yaw_deg = float(item.get("yaw_deg", 0.0))
        uniform_scale = float(item.get("uniform_scale", 1.0))
        # Guard against degenerate scale values.
        if uniform_scale <= 0.0:
            log.warning(
                "scene-layout LLM returned non-positive uniform_scale=%r for "
                "object_id=%r; replacing with 1.0",
                uniform_scale,
                oid,
            )
            uniform_scale = 1.0
        result.append(
            SceneObject(
                object_id=oid,
                prompt=prompt,
                placement=Placement(
                    object_id=oid,
                    yaw_deg=yaw_deg,
                    uniform_scale=uniform_scale,
                    translation=(x, 0.0, z),
                    ground_contact=True,
                ),
            )
        )
    return result


def _fallback_layout(prompt: str) -> tuple[SceneLayout, dict[str, Any]]:
    """Return a single-object layout echoing ``prompt`` (honest degradation).

    Used when the FixtureAdapter has no recorded completion for this prompt
    (or when the LLM returns empty/unusable output).  Mirrors the ``skipped``
    note convention used by the Generation Spec stage for unseen prompts so
    callers can detect the degraded path.
    """
    obj = SceneObject(
        object_id="object_0",
        prompt=prompt,
        placement=Placement(
            object_id="object_0",
            yaw_deg=0.0,
            uniform_scale=1.0,
            translation=(0.0, 0.0, 0.0),
            ground_contact=True,
        ),
    )
    layout = SceneLayout(objects=[obj], up_axis="+Y", ground_y=0.0)
    ledger: dict[str, Any] = {
        "stage": "scene_layout",
        "model": DEFAULT_MODEL,
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cost_usd": 0.0,
        "note": "skipped — no fixture for this prompt; single-object fallback returned",
    }
    return layout, ledger


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_scene_layout(
    prompt: str,
    *,
    adapter: LLMAdapter | None = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 2048,
    seed: int = 0,  # reserved for future deterministic generation
) -> tuple[SceneLayout, dict[str, Any]]:
    """Decompose ``prompt`` into a :class:`SceneLayout` via ``adapter``.

    Parameters
    ----------
    prompt:
        Free-text description of the scene (e.g. ``"a table with a mug on it"``).
    adapter:
        Any :class:`~astel_llm.adapter.LLMAdapter`.  Defaults to
        :class:`~astel_llm.adapter.FixtureAdapter` pointed at the package's
        ``fixtures/`` directory so the stage runs offline with no API key.
    model:
        Model string passed to the adapter.  Defaults to :data:`DEFAULT_MODEL`.
    max_tokens:
        Upper bound on completion length.
    seed:
        Reserved — included for API symmetry; not currently used.

    Returns
    -------
    tuple[SceneLayout, dict]
        ``(layout, ledger)`` where ``ledger`` is the credit-ledger row for the
        LLM call (or a zero-cost row on the fallback path).
    """
    if not prompt.strip():
        raise ValueError("prompt must be non-empty")

    if adapter is None:
        adapter = FixtureAdapter(_FIXTURES_DIR)

    try:
        result = adapter.complete_structured(
            system=SYSTEM_PROMPT,
            user=prompt.strip(),
            schema=_SCHEMA,
            model=model,
            max_tokens=max_tokens,
        )
    except FixtureMissingError:
        log.warning(
            "scene-layout: no fixture for prompt %r; returning single-object fallback",
            prompt,
        )
        return _fallback_layout(prompt)

    objects = _parse_objects(result.data)

    if not objects:
        log.warning(
            "scene-layout LLM returned no objects for prompt %r; "
            "returning single-object fallback",
            prompt,
        )
        return _fallback_layout(prompt)

    layout = SceneLayout(objects=objects, up_axis="+Y", ground_y=0.0)
    cost_ledger = ledger_entry(stage="scene_layout", model=model, usage=result.usage)
    return layout, cost_ledger
