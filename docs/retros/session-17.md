# Session 17 retro (2026-06-15)

**M3 integration — part 2 (FINAL): the Generation Spec LLM stage is wired into the
API text path.** This closes the last code-side M3 integration item from session
15/16. The text pipeline now runs prompt → `GenerationSpec` on submit, stores it
as an artifact, and threads the LLM's metric size estimate into the asset's
quality report — all **offline by default, zero spend**, with the live path behind
an explicit double gate.

Mode: Opus, inline. No founder gate touched (no API key used, no spend, nothing
committed).

## 1. What shipped (`services/api`)

- New `astel_api.generation_spec_stage`:
  - `run_generation_spec_stage(task_id, modality, prompt, store, settings)` — for
    text modality with a non-empty prompt, builds the spec via `astel_llm`,
    serialises it (`dataclasses.asdict`) + the credit-ledger row to a stored
    `generation-spec.json`. **Never raises** (an LLM failure must not fail the
    submit).
  - `apply_spec_scale_to_report(...)` — overwrites the quality report's `scale`
    block with the spec's `target_scale` (`method:"llm-estimate"`,
    `source:"generation-spec"`, with the user-overridable confidence band). This
    is the first **non-`None` scale** the Truth Meter can show for a generated
    asset — honestly flagged as an estimate, never a measurement.
- `astel_api.main.create_generation` now runs the stage after produce dispatch
  (so it can patch the freshly-written report) and gained a `Settings` dependency.
- `astel_api.config.Settings`: `llm_live` (default `False`) + `llm_fixtures_dir`.
- `astel-llm` added as an API dependency (editable path; torch-free).

## 2. Founder gate R-O2 — no silent spend (double-gated)

The stage is OFFLINE by default (`FixtureAdapter`, replays cached completions,
zero cost). It goes LIVE (`AnthropicAdapter`, real spend) **only when BOTH**
`ASTEL_LLM_LIVE=1` **AND** `ANTHROPIC_API_KEY` is present — so a key in the
environment for other reasons can never trigger a paid call. The common offline
case (an unseen prompt with no cached fixture) degrades gracefully: the stage
writes a `generation-spec.json` with `status:"skipped"` and a reason naming R-O2,
and the generation completes normally. The live adapter is never constructed in
tests or default runs.

## 3. Gates

- API: ruff ✅ · mypy --strict ✅ (21 files) · pytest **35 passed + 1 skipped**
  (5 new: non-text no-op; cache-miss→skipped; recorded-fixture→spec+ledger;
  report scale patched; patch is a no-op when skipped).
- GPU project unchanged this session (still ruff·mypy 33·56 pytest from session 16).

## 4. Honest gaps / carried forward

- **Live spec calls remain founder-gated (R-O2).** To enable: set
  `ANTHROPIC_API_KEY` + a spend cap, `uv sync --extra live` in `libs/astel_llm`,
  set `ASTEL_LLM_LIVE=1`, run one call (~$0.02–0.035/gen Haiku). Offline, only
  prompts with a recorded fixture produce a spec.
- **Text modality still has no prompt-conditioned geometry** — the spec is
  produced and the scale is threaded, but the splats themselves still come from
  the render-then-refit smoke (no text→multiview→L2 model yet). The spec is ready
  to condition that stage when it's built.
- The spec is not yet consumed by the GPU producer (it's API-side metadata +
  report enrichment only); feeding `parts`/`materials`/`symmetry` into L2/L3
  conditioning and L6 is future work (M4 for L6).
- Web Truth Meter not yet re-verified live against an LLM-estimated scale
  (the report shape is additive/back-compatible, so it should render).
- **Still nothing committed** — sessions 7–17 on the single "Beta" commit.

## 5. M3 status — integration COMPLETE in code

All of [13-m3-readiness](../research/13-m3-readiness.md) §4 steps 1–5 are
complete **and integrated**: the generative image path runs through the API to a
full `.astel` package (session 16), and the Generation Spec stage runs in the
text path (this session). The only remaining M3 item is the founder's API key
(R-O2) to light up live spec calls — not new code.

## 6. Next

**M4 — world-awareness (L4/L5/L6):** L4 appearance/relighting decomposition, L5
collision/solidity (SDF → convex proxies + watertight isosurface for the print
path), L6 physics-material LLM pass (reuses the `astel_llm` adapter + the same
fixture/founder-gate pattern). Plus, opportunistically: re-verify the web
Truth Meter against a real GPU generation in-browser.
