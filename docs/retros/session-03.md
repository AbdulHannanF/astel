# Session 03 Retro — M1 finished on CPU (2026-06-13)

*Mode: Opus plans, Sonnet subagents implement, Haiku for trivia. Minimal narration.
No GPU, no founder input required — exactly as scoped in `docs/NEXT_STEPS.md` (session-2 end).*

## Goal

Close out the five remaining M1 items, all CPU-only, so M1 exits with green CI, a
browser demo, a Temporal-backed resumable pipeline seam, and an eval-harness skeleton.

## What shipped (5 workstreams, 5 parallel Sonnet agents)

1. **Temporal engine** — graduated the spike into `services/api`. New
   `src/astel_api/temporal/{shared,activities,workflows,worker,devserver}.py` and a
   `TemporalTaskEngine` behind the existing `TaskEngine` protocol; the pure
   `workflow_progress_to_event` translation is unit-tested with no server. Engine is
   selected by `ASTEL_ENGINE=stub|temporal` (**stub stays default** so CI/pytest pass
   offline). Full `WorkflowEnvironment` integration test is gated behind
   `ASTEL_TEMPORAL_TESTS=1` (skipped by default — it downloads a test-server binary).
   `temporalio>=1.28` added; lazy imports keep module import temporalio-free.
2. **`libs/astel_format`** — Python reader/writer for the `.astel` package. Pydantic v2
   models mirroring `docs/specs/schemas/*`, `AstelPackage` zip writer (mimetype-first +
   STORED), jsonschema draft-2020-12 validation, path-traversal rejection,
   unknown-additive-key round-trip preservation, `build_minimal_package()`. 16 tests.
3. **`packages/@astel/manifest`** — TS manifest contract: types mirroring the schemas,
   `parseManifest`/`validatePaths` (ajv 2020-12) + `serializeManifest` preserving
   `extensions`/`extras`. 10 vitest. Added ajv/ajv-formats to the root lockfile.
4. **`libs/astel_splat_io`** — `.ply` (INRIA), `.spz` (Niantic v3 container, MIT),
   `.sog`/SOGS (PlayCanvas; meta.json + means/scales/sh0/quats WebP — `shN`, real
   k-means codebooks, PLAS sort, LOD bundles explicitly `NotImplementedError` and
   documented in `FORMATS.md`), plus the provenance sidecar (`*.astl.json` + UNORM8
   `.bin`) with a reorder-alignment golden test. 11 tests. Two real round-trip bugs were
   found and fixed during verification: an SPZ smallest-three quaternion bit-shift
   desync, and a libwebp lossless `exact=True` color-codebook corruption.
5. **`libs/astel_eval`** — frozen-corpus loader (50 cases transcribed to
   `corpus_v1.json`, asserted against `docs/eval/CORPUS.md`), `Adapter` protocol with
   unmistakably-non-real stub adapters (`available=False` + reason) for
   Astel/TRELLIS.2/Meshy/Tripo, an incremental runner, a Bradley-Terry MLE fit with
   bootstrap 95% CIs, and an M3-gate evaluator that reports losing cases with no spin.
   36 tests.

## Integration

- **`astel up`** (`scripts/up.ps1`, `pnpm run up`): dev mode (SQLite + stub + API + web,
  the verified path) by default; `-Temporal` brings up a local Temporal dev server +
  worker + API(`engine=temporal`) + web, erroring clearly if the `temporal` CLI is
  absent. Full prod-shaped deps remain `docker compose -f infra/docker-compose.yml up -d`.
- **CI** (`.github/workflows/ci.yml`): added a `libs` matrix job (astel_format /
  astel_splat_io / astel_eval: ruff + ruff-format + mypy + pytest, `uv sync --frozen`)
  and a `manifest` job (typecheck + lint + test). Still inert-until-remote (git local).

## Verification (all green, this box)

| Package | Gate result |
|---|---|
| services/api | ruff ok · mypy 14 files · pytest 12 pass, 1 skip |
| libs/astel_format | ruff ok · mypy 11 · pytest 16 |
| libs/astel_splat_io | ruff ok · mypy 11 · pytest 11 |
| libs/astel_eval | ruff ok · mypy 18 · pytest 36 |
| packages/@astel/manifest | tsc ok · eslint ok · vitest 10 |
| apps/web | (unchanged from M1: build + 9 vitest) |
| root | `pnpm install --frozen-lockfile` ok · `up.ps1` parses clean |

## Honest gaps / deferred

- Temporal **temporal-mode `astel up` and the gated integration test are unrun on this
  box** (no `temporal` CLI here). The stub path is the verified one; idempotent
  start/attach policy and `start-dev` flag mapping are written but not server-tested.
- SOG is a faithful **partial** writer (see `FORMATS.md`); SPZ implements the v3
  container, not the newer v4 ZSTD/TOC header.
- Eval adapters are stubs (no GPU/network); the harness is ready, the backends are M2+.
- Per-stage (not sub-tick) Temporal progress granularity — acceptable for M1.

## Next

M1 is closed. **M2 (capture path) is the first GPU milestone** — see `docs/NEXT_STEPS.md`.
Nothing here is blocking; the dev stack runs end-to-end on CPU today.
