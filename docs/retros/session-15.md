# Session 15 retro (2026-06-15)

**Generation Spec LLM stage scaffolded on fixtures (M3 step 5).** Built CLAUDE.md
§5's model-agnostic LLM layer + the text-pipeline prompt→spec stage entirely
**offline** — no Anthropic API key, no spend. Founder gate R-O2 untouched.

Mode: Opus, inline. External API facts re-verified live via the claude-api
reference (training data 5 months stale, per CLAUDE.md §10.1) before writing code.

## 1. What shipped — `libs/astel_llm`

A new standalone library (mirrors the `astel_eval`/`astel_format`/`astel_splat_io`
convention: own pyproject, ruff + mypy --strict + pytest):

- `spec.py` — `GenerationSpec` (object_class, summary, parts[{name,material}],
  materials, style, `target_scale` with an explicit **user-overridable confidence
  band**, symmetry) + an Anthropic-structured-output-compatible JSON schema
  (`additionalProperties:false` everywhere, no numeric/length constraints — those
  are validated in `from_dict` instead).
- `adapter.py` — `LLMAdapter` protocol with two backends:
  - **`FixtureAdapter` (default)** — replays cached completions keyed by a hash of
    `(model, system, user)`; `.record(...)` captures them. No key, no network.
  - **`AnthropicAdapter`** — the live backend; lazy-imports the SDK (optional
    `[live]` extra), constructed only when a key is present. Uses
    `output_config.format` structured output + a cache-controlled system block.
- `generation_spec.py` — `build_generation_spec(prompt, adapter)` → validated spec
  + credit-ledger row. Haiku 4.5 default; frozen system prompt + schema so prompt
  caching applies across generations.
- `pricing.py` — verified per-MTok rates + cache-discount math + `ledger_entry`.

Gates green: ruff · mypy --strict (9 files) · **14 pytest**, all offline.

## 2. Verified API facts (live, 2026-06-15)

Haiku 4.5 `claude-haiku-4-5` $1/$5; Sonnet 4.6 `claude-sonnet-4-6` $3/$15; Opus
4.8 `claude-opus-4-8` $5/$25. Structured JSON =
`output_config={"format":{"type":"json_schema","schema":…}}`; objects must set
`additionalProperties:false` and avoid numeric/length constraints. Prompt caching
via `cache_control:{type:"ephemeral"}` on the system block (Haiku min cacheable
prefix 4096 tokens). Token counting via `messages.count_tokens`. These match
doc 13's cost plan (~$0.02–0.035/gen Haiku).

## 3. Honest gaps / carried forward

- **The founder gate (R-O2) is the ONLY remaining M3 step**: set
  `ANTHROPIC_API_KEY` + spend cap, `uv sync --extra live`, run one live
  `AnthropicAdapter` call. Stage code is identical fixtures-vs-live. No paid call
  until approved.
- `astel_llm` is **not yet imported by the API** — the Generation Spec stage isn't
  wired into `produce` / the text-pipeline orchestration yet (integration step).
- L6 physics-material reasoning + QA-critique stages (same adapter) are M4, not
  built here.
- No central CI runner enumerates the libs (no `.github/workflows`; git is local)
  — `astel_llm` follows the per-package manual-gate convention like its siblings.
- **Still nothing committed** — sessions 7–15 GPU + lib work remains in the working
  tree on the single "Beta" commit. Flagged again; awaiting founder go-ahead.

## 4. M3 status

Steps 1–5 of [13-m3-readiness](../research/13-m3-readiness.md) §4 are complete in
code: triage ✅, install spike ✅, L2 graduate + bake-off/DECISIONS#2 ✅, L3 A/B +
L2→L3 wiring ✅, Generation Spec scaffolded ✅. **Remaining is integration**
(wire generative pipeline + LLM stage into the API `produce` path and `.astel`
packaging) **and the single founder gate** (API key) — not new research. M3's
research/build arc is effectively done; M4 (world-awareness: L4/L5/L6) is next.

## 5. Next

(a) **Integration**: wire `astel_gpu.generative` (image→L2→L3) and
`astel_llm.build_generation_spec` into the API `produce` path; emit `.astel`
packages with l2+l3+report for generated assets. (b) **Founder**: provide the
Anthropic API key + spend cap to light up live Generation Spec calls. (c) Then
**M4** — L4 relighting, L5 collision/SDF + print path, L6 physics-material LLM
pass (reusing the `astel_llm` adapter).
