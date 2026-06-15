"""The L6 physics-material stage, wired into the API text path (CLAUDE.md §3 L6).

For a text generation whose Generation Spec stage succeeded, this runs the
prompt's :class:`GenerationSpec` through ``astel_llm.build_physics_material_spec``
and, on success, stores the result as the **billable** ``l6.json`` layer artifact
(per-region material/density/friction/restitution + articulation hints). On a
fixture cache-miss it degrades to a non-billable ``physics-material.json``
"skipped" note — exactly like the Generation Spec stage, and behind the same
founder gate R-O2 (offline ``FixtureAdapter`` unless ``ASTEL_LLM_LIVE`` **and**
``ANTHROPIC_API_KEY`` are both set).

Storing ``l6.json`` only on success matters: ``billing._ARTIFACT_LAYER`` maps
``l6.json`` → the L6 add-on (4 credits), so a delivered L6 layer is charged while
a skip is not.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from typing import Any

from astel_llm import (
    FixtureMissingError,
    GenerationSpec,
    build_physics_material_spec,
)

from .config import Settings
from .generation_spec_stage import _select_adapter
from .storage import ArtifactStore

logger = logging.getLogger(__name__)

#: Billable artifact name (billing maps this to the L6 add-on). Written only
#: when the layer is genuinely produced.
_L6_ARTIFACT = "l6.json"
#: Non-billable transparency note for the skipped / offline-cache-miss case.
_SKIP_ARTIFACT = "physics-material.json"


def run_physics_material_stage(
    task_id: str,
    modality: str,
    spec_payload: dict[str, Any] | None,
    store: ArtifactStore,
    settings: Settings,
) -> dict[str, Any] | None:
    """Build + store the L6 physics-material layer for a text generation.

    Never raises. Returns the stored payload (``status: "ok" | "skipped"``) or
    ``None`` when the stage does not apply (non-text, or no successful spec to
    reason over).
    """
    if modality != "text" or not spec_payload or spec_payload.get("status") != "ok":
        return None

    try:
        spec = GenerationSpec.from_dict(spec_payload["spec"])
    except Exception:  # a malformed upstream spec must not fail the submit
        logger.exception("could not reconstruct GenerationSpec for %s", task_id)
        return None

    adapter, mode = _select_adapter(settings)
    try:
        result = build_physics_material_spec(spec, adapter)
    except FixtureMissingError:
        payload: dict[str, Any] = {
            "status": "skipped",
            "mode": mode,
            "reason": (
                "No cached fixture for this object's physics-material reasoning "
                "and live LLM is disabled (founder gate R-O2). Set ASTEL_LLM_LIVE=1 "
                "and ANTHROPIC_API_KEY to enable real L6 calls."
            ),
        }
        store.put(task_id, _SKIP_ARTIFACT, _dumps(payload))
        return payload
    except Exception:  # an LLM/stage failure must not fail the submit
        logger.exception("physics-material stage failed for %s", task_id)
        return None

    payload = {
        "schema": "astel.physics-material/v0",
        "status": "ok",
        "mode": mode,
        "spec": dataclasses.asdict(result.spec),
        "ledger": result.ledger,
    }
    store.put(task_id, _L6_ARTIFACT, _dumps(payload))
    logger.info(
        "physics-material (L6) for %s: regions=%d cost_usd=%s mode=%s",
        task_id,
        len(result.spec.regions),
        result.ledger.get("cost_usd"),
        mode,
    )
    return payload


def _dumps(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, indent=2).encode("utf-8")
