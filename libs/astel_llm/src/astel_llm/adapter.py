"""Model-agnostic LLM adapter (CLAUDE.md §5) with a fixture-backed default.

The whole LLM layer talks to one small interface, :class:`LLMAdapter`, so the
Generation Spec / L6 reasoning / QA-critique stages never import a vendor SDK
directly. Two implementations:

- :class:`FixtureAdapter` — replays cached responses keyed by a hash of
  ``(model, system, user)``. This is the DEFAULT for all development and tests:
  it needs no API key and incurs no spend (the founder-gate rule — see
  CLAUDE.md §10.2 / the model-tiering memo). Capture fixtures with
  :func:`FixtureAdapter.record`.
- :class:`AnthropicAdapter` — the real backend. Imports ``anthropic`` lazily
  (optional ``[live]`` extra) and is only constructed when an API key is present.
  The founder wires this in at the very end; nothing here calls it during normal
  development.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from hashlib import blake2b
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class TokenUsage:
    """Per-call token accounting (mirrors Anthropic's ``usage`` fields)."""

    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


@dataclass(frozen=True)
class StructuredResult:
    """A structured-output completion: parsed JSON + token usage + model id."""

    data: dict[str, Any]
    usage: TokenUsage
    model: str


class LLMAdapter(Protocol):
    """The one method every Astel LLM stage depends on."""

    def complete_structured(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        model: str,
        max_tokens: int = 2048,
    ) -> StructuredResult:
        """Return a JSON object conforming to ``schema`` plus token usage."""
        ...


def fixture_key(model: str, system: str, user: str) -> str:
    """Stable content hash identifying a cached completion."""
    h = blake2b(digest_size=12)
    h.update(model.encode("utf-8"))
    h.update(b"\x00")
    h.update(system.encode("utf-8"))
    h.update(b"\x00")
    h.update(user.encode("utf-8"))
    return h.hexdigest()


class FixtureMissingError(RuntimeError):
    """Raised when no recorded fixture matches a request."""


class FixtureAdapter:
    """Replays cached completions from a directory of ``<key>.json`` files."""

    def __init__(self, fixtures_dir: Path) -> None:
        self._dir = Path(fixtures_dir)

    def complete_structured(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        model: str,
        max_tokens: int = 2048,
    ) -> StructuredResult:
        key = fixture_key(model, system, user)
        path = self._dir / f"{key}.json"
        if not path.exists():
            raise FixtureMissingError(
                f"no fixture {path.name} for model={model!r}; record one with "
                "FixtureAdapter.record(...) (offline) or run the live adapter once."
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        usage = TokenUsage(**payload["usage"])
        return StructuredResult(data=payload["data"], usage=usage, model=model)

    def record(
        self, *, model: str, system: str, user: str, result: StructuredResult
    ) -> Path:
        """Persist ``result`` as the fixture for ``(model, system, user)``."""
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._dir / f"{fixture_key(model, system, user)}.json"
        path.write_text(
            json.dumps({"data": result.data, "usage": asdict(result.usage)}, indent=2),
            encoding="utf-8",
        )
        return path


class AnthropicAdapter:
    """Live Anthropic backend. Lazy-imports the SDK; needs an API key.

    Not exercised by tests or the fixture path. The founder constructs this with
    a real key at the end of M3 to enable paid calls.
    """

    def __init__(self, api_key: str | None = None) -> None:
        import anthropic  # noqa: PLC0415 (optional [live] dep, intentionally lazy)

        self._client = (
            anthropic.Anthropic(api_key=api_key)
            if api_key is not None
            else anthropic.Anthropic()
        )

    def complete_structured(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        model: str,
        max_tokens: int = 2048,
    ) -> StructuredResult:
        resp = self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=[
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user}],
            output_config={"format": {"type": "json_schema", "schema": schema}},
        )
        if resp.stop_reason == "refusal":
            raise RuntimeError("LLM refused the Generation Spec request")
        text = "".join(
            block.text
            for block in resp.content
            if getattr(block, "type", None) == "text"
        )
        data: dict[str, Any] = json.loads(text)
        usage = TokenUsage(
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            cache_read_input_tokens=getattr(
                resp.usage, "cache_read_input_tokens", 0
            )
            or 0,
            cache_creation_input_tokens=getattr(
                resp.usage, "cache_creation_input_tokens", 0
            )
            or 0,
        )
        return StructuredResult(data=data, usage=usage, model=model)
