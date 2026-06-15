# Session 21 retro (2026-06-15)

**M3 verification + MVP-polish at the founder's request; the "can I prompt it?"
question answered honestly; M4 entered with the L6 physics-material stage.**

Mode: Opus, inline. CPU only (one live GPU run on Box A to verify the generative
path). No founder gate touched; no spend.

## 1. Verification — every gate re-run, two real regressions fixed

The founder asked to "verify the workings." I re-ran every gate from scratch
rather than trusting the retros, and that caught two things the prior retros
reported as green:

- **API ruff E501** in the session-20 billing migration
  (`a1b2c3d4e5f6_billing_columns.py`) — the `mode` column line was 89 chars.
  Fixed (wrapped the `sa.Column(...)`).
- **The two GPU tests hard-failed under a plain `uv run pytest` on Box A.**
  `test_smoke_refit` / `test_synthetic_eval` guard only on
  `torch.cuda.is_available()`, but Box A *has* CUDA — so they ran and then died
  on gsplat's JIT compile (`FileNotFoundError` for `cl.exe`) because a plain
  pytest invocation lacks the MSVC env that `run-python.cmd` sets. Fixed with a
  shared **`requires_gsplat_runtime`** fixture (new `tests/conftest.py`) that
  also skips when `shutil.which("cl")` is `None`, so the documented command is
  green everywhere and the tests only *really* run through the launcher.

**Final gate state (all green):**

| Component | Result |
|---|---|
| API | ruff ✓ · mypy 25 ✓ · **56 pytest** (+5), 1 skipped |
| Web | **18 vitest** ✓ · tsc ✓ · eslint ✓ |
| `@astel/manifest` | **10 vitest** ✓ · tsc ✓ · eslint ✓ |
| libs `astel_*` | **97 pytest** (24 llm +10 solid +16 format +11 splat_io +36 eval) |
| GPU pipeline | ruff ✓ · mypy 34 ✓ · **55 pytest**, 2 skipped (skip cleanly off-launcher) |

## 2. The founder's question, answered honestly

> "Can I give it a text prompt now for model generation?"

**Not for geometry that matches the prompt — not yet.** Verified live over HTTP
(uvicorn, real POST): a text generation returns a valid layer stack
(`l0/l3.ply`, `.spz`, `.sog`, `package.astel`) + credit ledger, **but the shape
is a deterministic procedural placeholder** (`origin: stub`) unrelated to the
prompt, and the Generation Spec is `skipped` (no fixture, no key). The **only**
path that turns input into a real model today is **image → TripoSplat L2 → 2DGS
L3**, which I re-ran live on Box A (creature_butterfly, `--refine-iters 200`):
**65,536 gaussians, L2 11.1 s / L3 3.3 s, 0 non-finite, held-out 19.0 dB**, full
artifact contract incl. `l5.stl` + mass. **text → 3D needs a text→multiview
stage that is not built** — the real remaining M3-text gap.

## 3. Polish (so the MVP is honestly testable)

- Stub + GPU-smoke quality reports now state **explicitly** that the geometry is
  *not* derived from the prompt (was only "metrics are placeholders").
- The web dock shows a **modality-aware honesty hint** (text → placeholder +
  spec; image → live generative; video → not wired) so the demo never
  over-promises during input. The Truth Meter STUB pill / provenance bar already
  covered the *result*; this covers the *invitation*.
- New **[MVP_TESTING.md](../MVP_TESTING.md)** — how to test both paths, the
  honest text-vs-image table, and the gate commands. README + NEXT_STEPS
  de-staled (status now M3-closed / M4-in-progress).

## 4. M4 entered — L6 physics-material stage (the first world-awareness layer)

Built the **L6 physics-material & semantic** reasoning stage (CLAUDE.md §3 L6),
reusing `astel_llm` exactly like the Generation Spec stage — torch-free,
CPU-tested, **offline by default, no founder gate**:

- `astel_llm.physics_material`: typed `PhysicsMaterialSpec` (per-region
  `material` / `material_class` ∈ {rigid,soft,cloth,fluid_adjacent,granular} /
  `density_kg_m3` / `friction` / `restitution`, plus `ArticulationHint` joints),
  an Anthropic-structured-output-compatible `json_schema()`, range-validating
  `from_dict`, and `build_physics_material_spec(GenerationSpec, adapter)` →
  result + token ledger row (`stage="physics_material"`). Default Haiku 4.5
  (constrained material lookup); Sonnet is the documented upgrade
  (research doc 13 §3). **+10 lib tests.**
- API `physics_material_stage.run_physics_material_stage`: for a text gen with a
  successful spec, runs L6 and stores the **billable `l6.json`** layer on
  success, or a non-billable `physics-material.json` skip note on cache-miss —
  wired into `create_generation` after the spec stage, before billing. **+4
  endpoint-stage tests.**
- Billing **needed no change**: session 20 already mapped `l6.json` → the L6
  add-on (4 credits). Added a unit test proving a delivered `l6.json` charges the
  L6 add-on on a refine and never on a preview. **+1 billing test.**

## 5. Honest gaps / carried forward

- **L6 is text-only and spec-driven.** It reasons over the Generation Spec; the
  image path has no spec yet, so no L6 there. The VLM-over-renders variant
  (CLAUDE.md §3 L6 "reasoning pass over renders") is the follow-on.
- **L6 ↔ L5 not joined yet.** Density × the L5 solid volume gives real per-region
  mass; that join (and writing L6 into the `.astel` manifest as a bound layer) is
  the next L6 step. Today `l6.json` is a loose artifact + a billed layer.
- **L6 LLM cost isn't folded into the credit ledger's LLM line** (only the spec
  stage is). The flat L6 add-on (4 cr) prices the layer; the raw token cost is
  logged in `l6.json` for margin telemetry. Revisit when live spend is enabled.
- Still offline: no fixture ships, so the default text path skips L6 (writes the
  skip note). Real L6 output needs a fixture or the live key (R-O2).

## 6. Next

- **Founder decision surfaced** (see NEXT_STEPS banner): the highest-value build
  is arguably the **text→multiview bridge** (so a text prompt yields a real
  model — mission modality #1), *ahead of* finishing M4. Awaiting the call.
- Otherwise continue M4: join L6→L5 for per-region mass + bind L6 into `.astel`;
  then L4 relighting, metric-scale L5, CoACD + `.3mf` + printability.
