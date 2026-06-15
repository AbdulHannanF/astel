# astel-llm

Astel's model-agnostic LLM layer (CLAUDE.md §5). The text pipeline's
**Generation Spec** stage (prompt → structured spec), a vendor-neutral adapter,
and a token-cost ledger.

## What's here

- `spec.py` — `GenerationSpec` (object class, parts, materials, style,
  metric `target_scale` with an explicit user-overridable confidence band,
  symmetry) + the Anthropic-compatible structured-output JSON schema.
- `adapter.py` — `LLMAdapter` protocol with two backends:
  - **`FixtureAdapter` (default)** — replays cached completions keyed by a hash
    of `(model, system, user)`. No API key, no spend. Record with `.record(...)`.
  - **`AnthropicAdapter`** — the live backend (optional `[live]` extra). Lazy SDK
    import; only constructed when an API key is present.
- `generation_spec.py` — `build_generation_spec(prompt, adapter)` → validated
  spec + credit-ledger row. Defaults to Haiku 4.5; system prompt + schema are
  frozen so prompt caching applies across generations.
- `pricing.py` — verified per-MTok rates (2026-06-15) + cache-discount math +
  `ledger_entry(...)` for the credit ledger.

## Founder gate (R-O2)

All development and tests run **offline** via `FixtureAdapter` — no Anthropic API
key, no paid call. To enable real calls at the end of M3:

```bash
uv sync --extra live          # install the anthropic SDK
export ANTHROPIC_API_KEY=...  # founder's key + spend cap
```

```python
from astel_llm import AnthropicAdapter, build_generation_spec
result = build_generation_spec("a brass pocket watch", AnthropicAdapter())
print(result.spec, result.ledger)   # ledger['cost_usd'] ~ $0.02–0.035 / gen (Haiku)
```

The stage code is identical for fixtures vs live — only the adapter changes.

## Develop

```bash
uv run ruff check . && uv run mypy && uv run pytest -q
```
