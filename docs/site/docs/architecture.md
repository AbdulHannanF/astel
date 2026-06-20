# Architecture

A high-level view of how Astel is put together. The authoritative, code-level
document lives in the repo at `docs/architecture/ARCHITECTURE.md`.

## Monorepo layout

```
apps/web         Vite + React 19 + TS product app — splat viewer (Spark), Layer
                 Inspector, Truth Meter, Relight Studio, Physics Sandbox.
services/api     FastAPI gateway: generations API + SSE progress + captures upload +
                 billing. SQLAlchemy 2 (async). Work runs behind a TaskEngine seam
                 (stub default; Temporal opt-in).
pipelines/gpu    Real producer: text/image/video → TripoSplat L2 → 2DGS L3 → full
                 .astel stack, on gsplat. Runs out-of-process so torch stays out of
                 the API environment.
pipelines/stub   CPU procedural producer (the default; deterministic per task).
libs/            Torch-free Python libraries (format, splat IO, eval, LLM, solid,
                 appearance, dynamics, scene, LOD) — each a uv project with its own gates.
packages/        TS @astel/manifest + @astel/sdk; Python astel_sdk + astel-mcp.
plugins/         Unity + UE5 importers (consume the flat engine.json sidecar).
```

## Request flow

```
Browser (apps/web)
  POST /v1/captures            ──▶ upload image/video bytes → capture_id
  POST /v1/generations         ──▶ row inserted (QUEUED); the producer writes a real
                                    per-task layer stack into the ArtifactStore
  GET  /v1/generations/{id}/events  (SSE)
                               ──▶ L0_SEED → L1_DENSE → L2_COARSE → L3_REFINED progress,
                                    terminal event flips the row to SUCCEEDED/FAILED
  GET  /v1/generations/{id}/artifacts/{name}
                               ──▶ FileResponse from the ArtifactStore
  GET  /v1/pricing             ──▶ credit price schedule (preview/refine tiers)
```

## Producer seam

The API never imports torch. `ASTEL_PRODUCER=stub` (default) runs the CPU procedural
generator in-process; `ASTEL_PRODUCER=gpu` shells out to `pipelines/gpu` so the heavy
CUDA stack is isolated. Both converge on the same `.astel` artifact contract
(`l0.ply`, `l3.ply`, `l3.spz`, `l3.sog`, `l3.glb`, `engine.json`, `package.astel`,
`quality-report.json`, plus L2/L4/L5/L6/L7 artifacts when produced), so the viewer
and downstream consumers see one contract regardless of producer.

## Task engine

Multi-stage work runs behind a `TaskEngine` interface. The default stub engine streams
progress synchronously (fine for the CPU producer); the durable, resumable path is the
Temporal engine (`ASTEL_ENGINE=temporal`), which models each layer stage as an activity.

## Storage & infra

Artifacts go through a `LocalArtifactStore` (S3-swappable seam, path-traversal
guarded). `infra/docker-compose.yml` brings up the prod-shaped stack
(Postgres / MinIO / Temporal); local dev defaults to SQLite + on-disk artifacts.

For the full picture — including coordinate conventions, the billing model, and the
`.astel` spec — see the [coordinate conventions](coordinate-conventions.md),
[glTF export](gltf-export.md), and the repo's `docs/specs/manifest-v0.md`.
