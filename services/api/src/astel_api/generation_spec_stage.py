"""The text-pipeline Generation Spec stage, wired into the API (CLAUDE.md §4).

For a text-modality generation this runs the prompt -> :class:`GenerationSpec`
stage from ``astel_llm``, stores the result as a ``generation-spec.json`` artifact,
and threads the LLM's metric size estimate into the asset's quality report (the
first non-``None`` scale the Truth Meter can show, honestly flagged as an
LLM estimate with a user-overridable confidence band).

**Founder gate R-O2 (no silent spend).** The stage is OFFLINE by default: it uses
``FixtureAdapter`` (replays cached completions, zero cost). It goes LIVE
(``AnthropicAdapter``, real spend) only when BOTH ``settings.llm_live`` is set AND
an ``ANTHROPIC_API_KEY`` is present -- so a key in the environment for other
reasons can never trigger a paid call. On a fixture cache-miss (the common offline
case for an unseen prompt) the stage degrades gracefully: it writes an honest
"skipped" note and never fails the generation.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
from typing import Any

from astel_llm import (
    AnthropicAdapter,
    FixtureAdapter,
    FixtureMissingError,
    LLMAdapter,
    build_generation_spec,
)

from .config import Settings
from .storage import ArtifactStore

logger = logging.getLogger(__name__)

_SPEC_ARTIFACT = "generation-spec.json"
_REPORT_ARTIFACT = "quality-report.json"


def _select_adapter(settings: Settings) -> tuple[LLMAdapter, str]:
    """Return ``(adapter, mode)`` -- live only behind the explicit founder gate."""
    if settings.llm_live and os.environ.get("ANTHROPIC_API_KEY"):
        return AnthropicAdapter(), "live"
    return FixtureAdapter(settings.llm_fixtures_dir), "fixture"


def run_generation_spec_stage(
    task_id: str,
    modality: str,
    prompt: str,
    store: ArtifactStore,
    settings: Settings,
) -> dict[str, Any] | None:
    """Build + store the Generation Spec for a text prompt. Never raises.

    Returns the stored payload dict (``status: "ok" | "skipped"``) or ``None``
    when the stage does not apply (non-text modality / empty prompt) or fails.
    """
    if modality != "text" or not prompt.strip():
        return None

    adapter, mode = _select_adapter(settings)
    try:
        result = build_generation_spec(prompt, adapter)
    except FixtureMissingError:
        payload: dict[str, Any] = {
            "status": "skipped",
            "mode": mode,
            "reason": (
                "No cached fixture for this prompt and live LLM is disabled "
                "(founder gate R-O2). Set ASTEL_LLM_LIVE=1 and ANTHROPIC_API_KEY "
                "to enable real Generation Spec calls."
            ),
        }
        store.put(task_id, _SPEC_ARTIFACT, _dumps(payload))
        return payload
    except Exception:  # an LLM/stage failure must not fail the submit
        logger.exception("generation spec stage failed for %s", task_id)
        return None

    payload = {
        "status": "ok",
        "mode": mode,
        "spec": dataclasses.asdict(result.spec),
        "ledger": result.ledger,
    }
    store.put(task_id, _SPEC_ARTIFACT, _dumps(payload))
    logger.info(
        "generation spec for %s: cost_usd=%s mode=%s",
        task_id,
        result.ledger.get("cost_usd"),
        mode,
    )
    return payload


def apply_spec_scale_to_report(
    task_id: str, store: ArtifactStore, spec_payload: dict[str, Any] | None
) -> None:
    """Overwrite the quality report's ``scale`` with the LLM size estimate.

    No-op unless a spec was produced (``status == "ok"``) and a report exists.
    The scale is marked ``method: "llm-estimate"`` with the spec's confidence
    band -- honest provenance, never presented as a measurement.
    """
    if not spec_payload or spec_payload.get("status") != "ok":
        return
    report_path = store.path_for(task_id, _REPORT_ARTIFACT)
    if report_path is None:
        return
    try:
        report = json.loads(report_path.read_text())
        ts = spec_payload["spec"]["target_scale"]
        report["scale"] = {
            "longest_axis_m": ts["longest_axis_m"],
            "confidence": ts["confidence"],
            "low_m": ts["low_m"],
            "high_m": ts["high_m"],
            "method": "llm-estimate",
            "source": "generation-spec",
        }
        store.put(task_id, _REPORT_ARTIFACT, _dumps(report))
    except Exception:  # report patching is best-effort, never fatal
        logger.exception("failed to apply spec scale to report for %s", task_id)


def _dumps(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, indent=2).encode("utf-8")
