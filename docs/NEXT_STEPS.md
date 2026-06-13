# NEXT_STEPS — Runway (updated 2026-06-13, end of session 6)

> **Session 6 done:** both session-5 carryover items closed — see
> [session-06 retro](retros/session-06.md). A real live-browser drag-drop of
> an image confirmed the `/v1/captures` round-trip and all 6 layer-stack
> artifacts (`l0.ply`/`l3.ply`/`l3.spz`/`l3.sog`/`package.astel`/
> `quality-report.json`) serve correctly from a running `astel up`-equivalent
> stack. Alembic migration scaffold added to `services/api` (async env.py,
> baseline migration covering the current `generations` table incl.
> `capture_id`); `create_all` stays for dev/test, Alembic is the path forward
> for real schema changes.

## Where we are

**Phase R closed. M1 CLOSED. M2 spine landed (CPU): real per-task artifacts flow end-to-end** (verified via a live browser round-trip in session 6), **including full `.astel` packages and `/v1/captures` uploads, with Alembic migrations now scaffolded.** Product named **Astel** (`.astel`).
- Stack chosen, deep-read, and license-audited ([DECISIONS.md](research/DECISIONS.md) v0.2,
  [LICENSE_AUDIT.md](research/LICENSE_AUDIT.md) v2).
- Task engine finalized = **Temporal** ([RA10](research/10-task-engine-spike.md)), now
  graduated into `services/api` behind the `TaskEngine` seam (stub default; `ASTEL_ENGINE=temporal`).
- `.astel` format implemented both sides: `libs/astel_format` (Python) + `packages/@astel/manifest` (TS).
- Splat exporters: `libs/astel_splat_io` — `.ply`/`.spz`/`.sog` + provenance sidecar.
- Blind-eval harness: `libs/astel_eval` — frozen corpus loader + Bradley-Terry + M3 gate (stub adapters).
- M1 monorepo runs end-to-end on CPU: web app (Spark viewer + Layer Inspector + Truth Meter),
  FastAPI + SSE, stub splat pipeline, CI (web · manifest · api · pipeline-stub · libs · license-gate),
  infra compose. See [ARCHITECTURE.md](architecture/ARCHITECTURE.md) and [session-03 retro](retros/session-03.md).
  Start it: `pnpm install` then **`pnpm run up`** (one-command bring-up; `-Temporal` for the durable engine)
  or `pnpm run dev:all` (web + API) / `pnpm dev` (web only).

## Two decisions still open — both GPU-gated (deferred by founder 2026-06-13)

1. **L3 representation**: 2DGS surfels vs 3DGS + GOF extraction — needs a GPU A/B on fuzzy content.
2. **Generative geometry prior**: TRELLIS.2 O-Voxel → surfel **distillation fidelity** (R-T1) —
   the single riskiest bet. First GPU job whenever GPU work resumes.

## Session 3 — finish M1 without a GPU ✅ DONE (2026-06-13)

All five items landed and green (see [session-03 retro](retros/session-03.md) for detail + honest gaps):
1. ✅ **Temporal engine integration** — `TemporalTaskEngine` behind the `TaskEngine` seam;
   `temporal server start-dev` managed via `astel up -Temporal`. Stub stays default; offline-safe tests.
2. ✅ **`.astel` packages** — `libs/astel_format` (Python) + `packages/@astel/manifest` (TS), round-trip tested.
3. ✅ **Export writers** — `libs/astel_splat_io`: `.ply`/`.spz`/`.sog` + provenance sidecar (SOG partial, documented).
4. ✅ **Blind-eval harness** — `libs/astel_eval`: corpus loader + Bradley-Terry + M3 gate + stub adapters.
5. ✅ **`astel up`** — `scripts/up.ps1` / `pnpm run up` (dev default; `-Temporal` opt-in).

M1 exit criteria met: green CI + browser demo + Temporal-backed resumable seam + eval-harness skeleton.

## Session 4 — real-artifact spine + first true browser SSE round-trip ✅ DONE (2026-06-13)

Closed the biggest "it's all simulated" gap from M1: the stub engine emitted progress events but
**never produced a file** — the viewer loaded a static checked-in `.ply` and the Truth Meter read a
static JSON. Now every generation produces and serves a **real, unique** asset on CPU, and the web
UI is driven by it. No GPU. All gates green (API: ruff·mypy-strict·17 pytest; web: eslint·tsc·15 vitest).

1. ✅ **Artifact store + producer** (`services/api`): `storage.py` (`LocalArtifactStore`, S3-swappable
   seam, path-traversal-guarded, `ASTEL_ARTIFACT_DIR`) + `producer.py` (deterministic per-task
   procedural splat seeded from `task_id`, reuses `libs/astel_splat_io.write_ply` via an editable path
   dep). On submit it writes `l3.ply` + an honest `quality-report.json` (`origin:"stub"` + explicit
   caveats — honesty channel intact; numbers flagged as illustrative, not measured).
2. ✅ **Serving route** `GET /v1/generations/{id}/artifacts/{name}` (FileResponse, 400 on bad name,
   404 if missing); `GenerationResource.artifacts[]` now lists what's on disk.
3. ✅ **Web wired to real output**: viewer loads the per-task `l3.ply` (sample = idle/fallback);
   Truth Meter renders the live API report with a mandatory **STUB** pill + caveat; Layer Stack
   L0–L3 reflect the SSE run ("4/8 ready").
4. ✅ **Fixed two real bugs found by the first live browser run** (M1 had only vitest + screenshots):
   (a) `App` and `GenerationDock` each held a *separate* `useGeneration()` instance → success never
   reached the viewer; lifted to one shared instance. (b) **SSE parser only split on `\n\n`** but
   sse-starlette emits **CRLF** (`\r\n\r\n`) → every event silently dropped, stream "completed" only
   on socket close. Parser is now CRLF/CR/LF-robust per the SSE spec; locked with a CRLF unit test.

Honest gaps: artifacts produced synchronously at submit in stub mode (fine for CPU stub; the durable
async path is the Temporal engine, unchanged); full `.astel` packaging + `.spz`/`.sog` exports +
`l0.ply` not yet wired into the producer (writers exist in `libs/astel_splat_io`); no `/v1/captures`
upload endpoint yet (Text path only — Image/Video tabs still send a placeholder string).

## Session 5 — close OPEN_ISSUES.md + full layer-stack artifacts + captures upload ✅ DONE (2026-06-13)

See [session-05 retro](retros/session-05.md) for full detail.

1. ✅ **P1** `astel_eval` suite 8m30s → ~4s: vectorized Bradley-Terry fixed-point fit +
   smoothing-tie prior (fixes a real MLE-divergence pathology on separated data) +
   relative early-stop.
2. ✅ **P2** `README.md` and `docs/architecture/ARCHITECTURE.md` de-staled (pnpm quickstart,
   `libs/` layout, all 6 CI jobs, real artifact flow, Temporal seam, current test counts).
3. ✅ **P3** producer now writes the full layer-stack artifact set per task: `l0.ply`,
   `l3.ply`, `l3.spz`, `l3.sog` (best-effort), `package.astel` (real `.astel` zip via
   `astel_format.builder.build_minimal_package`, fully-typed honest `QualityReport`), plus
   the existing Truth-Meter `quality-report.json`. New `POST /v1/captures` multipart upload
   (stores raw bytes, returns `capture_id`); web `GenerationDock` uploads Image/Video drops
   and threads `capture_id` into the generation request.

Honest gaps carried forward: captures are uploaded but not yet *consumed* (producer still
emits the stub splat regardless); no DB migration tooling (new `capture_id` column via
`create_all`, fine for dev SQLite); `.sog` remains best-effort per `astel_splat_io`'s own
docs; capture upload verified via automated tests, not yet a live-browser round trip.

## Session 6 — live-browser capture round-trip + Alembic scaffold ✅ DONE (2026-06-13)

See [session-06 retro](retros/session-06.md) for full detail.

1. ✅ Live browser pass: simulated drag-drop of an image onto the
   `GenerationDock`, confirmed `POST /v1/captures` (201) → `capture_id` →
   `POST /v1/generations` → SSE to "Asset ready" → Truth Meter STUB pill with
   `estimate (image)` → all 6 artifacts (`l0.ply`, `l3.ply`, `l3.spz`,
   `l3.sog`, `package.astel`, `quality-report.json`) served 200 with
   non-zero bodies. Added `"api"` entry to `.claude/launch.json`.
2. ✅ Alembic scaffold in `services/api`: `alembic.ini` +
   `migrations/` (async `env.py` reading `get_settings().database_url`,
   `target_metadata = Base.metadata`), baseline migration for the current
   `generations` table. `create_all` stays for dev/test; Alembic documented
   in ARCHITECTURE.md as the path for real schema changes.

Honest gap: Alembic is wired but no migration has run against persistent
data yet — first real test comes with the next schema change. Text- and
Video-modality live-browser passes still untested (only Image this session).

## M2 — Capture path (the first GPU milestone; starts when GPU work resumes)

photos/video → L0→L1→L3 (reality first), quality report v1, exports. Needs the 2×4090 box.
This is where the deferred GPU smoke tests live (CUDA-in-WSL sanity → gsplat reference train →
MapAnything orbit test → TRELLIS import check → the R-T1 distillation experiment).

## What the founder does

**Nothing is blocking right now** — M1 finishes on the dev box without you.

**When you want to scale onto the GPU boxes (your call — "make everything work, then scale GPU"):**
- Run [`scripts/setup-gpu-box.ps1`](../scripts/setup-gpu-box.ps1) as admin on the 2×4090 box
  (`100.87.142.33`, SSH is already open); send the Windows username; confirm ≥150 GB free.
- (The 3×3080 box `100.70.127.42` was offline this session — bring it online when convenient.)

**Anytime, non-blocking & free:**
- Film the 10 orbit videos in [eval/CORPUS.md](eval/CORPUS.md) §capture (phone, slow orbit) —
  these become real capture-path test assets for M2.

**Later (only when the LLM layer starts, ~M3):** Anthropic API key — the agent will estimate
monthly cost before anything is spent.

**Decisions already settled (do not re-ask):** name = Astel; git stays local for now; GPU
deferred until you say otherwise.
