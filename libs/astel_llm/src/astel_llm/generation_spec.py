"""The text-pipeline prompt -> Generation Spec stage (CLAUDE.md §4).

Turns a free-text prompt into a validated :class:`GenerationSpec` via any
:class:`LLMAdapter`, and returns the credit-ledger entry for the call. Defaults
to Haiku 4.5 (constrained extraction, not deep reasoning) with the spec schema +
system prompt held stable so prompt caching applies across generations.

Runs entirely offline with :class:`FixtureAdapter`; swap in
:class:`AnthropicAdapter` once an API key is available — the stage code is
unchanged either way.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .adapter import LLMAdapter
from .pricing import ledger_entry
from .spec import GenerationSpec

DEFAULT_MODEL = "claude-haiku-4-5"

#: Frozen so the cached prefix (tools -> system) stays byte-identical across
#: generations — only the per-prompt user turn varies (prompt-caching §).
SYSTEM_PROMPT = (
    "You are Astel's generation-spec parser. Given a user's text prompt "
    "describing a single 3D object, extract a STRUCTURED specification for a "
    "geometry-accurate Gaussian-splat asset. Rules:\n"
    "- object_class: a concise noun phrase for the whole object.\n"
    "- summary: one sentence describing the object.\n"
    "- parts: the distinct sub-parts, each with its dominant physical material "
    "(e.g. {name: 'handle', material: 'wood'}).\n"
    "- materials: the unique materials across all parts.\n"
    "- style: the visual/era style (e.g. 'modern', 'baroque', 'photoreal').\n"
    "- target_scale: your best metric size estimate of the object's LONGEST "
    "axis in METRES, as longest_axis_m, with a confidence in [0,1] and a "
    "plausible interval low_m <= longest_axis_m <= high_m. Be honest about "
    "uncertainty: a wide interval with low confidence is correct when the "
    "prompt under-specifies size. The user can override this.\n"
    "- symmetry: one of none | bilateral | radial | axial.\n"
    "Never invent parts the prompt does not imply. Output ONLY the JSON object."
)


@dataclass(frozen=True)
class GenerationSpecResult:
    """The parsed spec plus the credit-ledger row for the LLM call."""

    spec: GenerationSpec
    ledger: dict[str, Any]


def build_generation_spec(
    prompt: str,
    adapter: LLMAdapter,
    *,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 1024,
) -> GenerationSpecResult:
    """Parse ``prompt`` into a validated GenerationSpec via ``adapter``."""
    if not prompt.strip():
        raise ValueError("prompt must be non-empty")
    result = adapter.complete_structured(
        system=SYSTEM_PROMPT,
        user=prompt.strip(),
        schema=GenerationSpec.json_schema(),
        model=model,
        max_tokens=max_tokens,
    )
    spec = GenerationSpec.from_dict(result.data)
    ledger = ledger_entry(stage="generation_spec", model=model, usage=result.usage)
    return GenerationSpecResult(spec=spec, ledger=ledger)
