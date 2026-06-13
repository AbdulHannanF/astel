# Astel — Architecture (M1 skeleton)

*Updated 2026-06-13. This documents what actually exists in the repo today and,
explicitly, where the dev-mode skeleton deviates from the intended production
topology. Decisions behind these choices live in
[`docs/research/DECISIONS.md`](../research/DECISIONS.md).*

## Monorepo layout

```
apps/web            Vite + React 19 + TS (strict) product app — viewer, Layer
                    Inspector, Truth Meter, generation dock. Spark splat renderer.
services/api        FastAPI gateway: generations API + SSE progress. SQLAlchemy 2
                    (async). Task work runs behind a TaskEngine seam. Per-task
                    artifacts (l3.ply, quality-report.json) via storage.py/producer.py.
pipelines/stub      Procedural sample-splat generator → apps/web/public/samples/.
packages/           Shared TS packages (@astel/manifest; @astel/sdk reserved).
libs/               Shared Python libraries (uv-managed, each with its own
                    pyproject.toml):
                      astel_format     — reader/writer for the .astel package
                                         format (manifest-v0).
                      astel_splat_io   — splat export/import writers: .ply, .spz,
                                         .sog, and provenance sidecars.
                      astel_eval       — blind-eval harness: corpus loader,
                                         adapters, runner, Bradley-Terry scoring
                                         scaffold for the M3 gate.
infra/              docker-compose.yml — prod-shaped stack (Postgres/MinIO/Temporal).
.github/workflows/  ci.yml — web / manifest / api / pipeline-stub / libs /
                    license-gate jobs.
docs/               research, specs (.astel manifest), eval corpus, architecture.
experiments/        spikes (task-engine-spike → Temporal decision).
```

Workspace tooling: **pnpm** (`pnpm-workspace.yaml` globs `apps/*`, `packages/*`)
for JS; **uv** per-service for Python (`services/api`, `pipelines/stub` each have
their own `pyproject.toml` + `uv.lock`).

## Request flow (M1)

```
Browser (apps/web)
  POST /v1/generations {modality, prompt}      ──▶  row inserted (QUEUED), task id returned
                                                  ──▶  produce_artifacts() writes a real
                                                        per-task l3.ply + quality-report.json
                                                        into the ArtifactStore (synchronous,
                                                        stub mode)
  GET  /v1/generations/{id}/events  (SSE)       ──▶  TaskEngine.run() streams ProgressEvent
        L0_SEED → L1_DENSE → L2_COARSE → L3_REFINED, each with shaped metrics
        (splats, PSNR, Chamfer mm, VRAM GB), terminal event flips row to SUCCEEDED
  GET  /v1/generations/{id}/artifacts/{name}    ──▶  FileResponse from the ArtifactStore
                                                        (400 on bad name, 404 if missing)
  GET  /v1/pipeline                             ──▶  static StageSpec list for the progress rail
```

On submit, `services/api/src/astel_api/producer.py` (`produce_artifacts`) builds a
deterministic procedural `SplatCloud` seeded from the task id, writes it as `l3.ply`
via `astel_splat_io.write_ply`, and writes an honest `quality-report.json`
(`schema: astel.quality-report/v0`, `origin: "stub"`, explicit caveats — illustrative
metrics, not measured) — both stored via `storage.py`'s `LocalArtifactStore`
(filesystem under `ASTEL_ARTIFACT_DIR`, path-traversal-guarded, S3-swappable seam).
`GenerationResource.artifacts[]` lists what's on disk. On success, the web app loads
the per-task `l3.ply` and live quality report in the Spark viewport (the checked-in
sample `apps/web/public/samples/astel-sample.ply` + `.report.json` remains the
idle/fallback state before a generation completes), and the Truth Meter renders the
live report with a mandatory **STUB** pill + caveat.

## The seams that matter (designed so prod swaps in without route changes)

- **`TaskEngine` protocol** (`services/api/.../engine.py`): `InProcessStubEngine`
  (asyncio sleeps + jittered durations + shaped metrics, no GPU, no external deps)
  remains the default. A `TemporalTaskEngine` implementing the same
  `run(task_id) -> AsyncIterator[ProgressEvent]` protocol now also exists
  (`services/api/src/astel_api/temporal/`), selected via `ASTEL_ENGINE=stub|temporal`
  (`temporal server start-dev` managed via `astel up -Temporal`). Temporal was
  finalized by spike ([RA10](../research/10-task-engine-spike.md)).
- **Provenance channel**: reserved in the `.astel` manifest spec
  ([docs/specs](../specs/manifest-v0.md)) before any pipeline writes it — per
  binding architecture decision #1 (retrofitting provenance is impossible).

## Dev-mode vs production deviations (intentional, tracked)

| Concern | M1 dev (today) | Production (target) | Why deferred |
|---|---|---|---|
| Database | SQLite via `aiosqlite` (`astel_dev.db`, gitignored) | Postgres (`infra/docker-compose.yml`) | No Docker / no admin on dev box; DB URL is env-driven so prod is a config swap |
| Object storage | `LocalArtifactStore` on the local filesystem (`ASTEL_ARTIFACT_DIR`, gitignored) | MinIO/S3 for layer artifacts | Filesystem store is an S3-swappable seam; same interface |
| Task engine | `InProcessStubEngine` default; `TemporalTaskEngine` exists, opt-in via `ASTEL_ENGINE=temporal` | Temporal durable workflow as default; SSE subscribes to event history | Temporal engine is integrated but untested on this box (no `temporal` CLI here) |
| Artifacts | Procedural per-task `l3.ply` + `quality-report.json` (`origin: "stub"`), produced synchronously at submit | Real GPU reconstruction outputs (L0–L7) | Full `.astel` packaging, `.spz`/`.sog` exports, `l0.ply`, and async production await M2 GPU pipeline |
| GPU pipeline | stub only (procedural PLY) | gsplat/TRELLIS/MapAnything on the 2×4090 box | Founder deferred GPU work 2026-06-13 — "make everything work, then scale GPU" |
| Auth / credits | none | API keys + credit ledger (spec §7) | Post-M1 |

## Database migrations (Alembic)

Schema changes to `services/api/src/astel_api/db.py` (SQLAlchemy `Base.metadata`)
are tracked as Alembic migrations under `services/api/migrations/`. `init_db()`
still runs `create_all` for fresh dev/test SQLite databases (no migration
ceremony needed for a from-scratch DB), but any change to the ORM models from
here on should also get a migration so Postgres deployments can upgrade
in place.

`migrations/env.py` runs in async mode (`async_engine_from_config` +
`asyncio.run`, matching the app's `create_async_engine` usage) and pulls the
DB URL from `astel_api.config.get_settings().database_url` — no separate
connection string to maintain.

```
cd services/api
uv run alembic upgrade head                          # apply migrations
uv run alembic revision --autogenerate -m "..."      # generate a new migration after model changes
```

## Verified local checks (all green 2026-06-13)

```
pnpm -C apps/web build         # tsc -b && vite build
pnpm -C apps/web test           # vitest — 15 tests
pnpm -C apps/web lint           # eslint + tsc --noEmit
pnpm -C packages/manifest test  # @astel/manifest — 10 tests

cd services/api   && uv run ruff check . && uv run mypy . && uv run pytest   # 17 tests + 1 skip (Temporal-gated)
cd pipelines/stub && uv run ruff check . && uv run mypy . && uv run pytest   # 14 tests

cd libs/astel_format   && uv run ruff check . && uv run mypy && uv run pytest   # 16 tests
cd libs/astel_splat_io && uv run ruff check . && uv run mypy && uv run pytest   # 11 tests
cd libs/astel_eval     && uv run ruff check . && uv run mypy && uv run pytest   # ~36 tests
```

Run the full dev loop with `pnpm run dev:all` (web + API together, see
`scripts/dev.ps1`), or `pnpm dev` for the web app alone (Vite proxies `/v1` and
`/healthz` to the API at :8000).

## Toolchain (dev box, verified)

Node 24 · pnpm 11.6 · Python 3.14 system (uv pins 3.12 for service envs; CI uses
3.12) · git. No Docker, no admin, no CUDA GPU (Quadro P1000) — the pipeline runs
in stub mode here; real GPU stages target the 2×4090 box when GPU work resumes.
