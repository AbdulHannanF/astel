# Session 04 retro — real-artifact spine + first true browser SSE round-trip

**Date:** 2026-06-13
**Mode:** Opus planned/reviewed; two Sonnet subagents implemented focused slices; Opus did
integration debugging + the SSE/state fixes. No GPU. Nothing committed (awaiting founder go).

## Goal

Founder asked to (1) run the server to see the web foundations, then (2) continue and complete the
next task. M2 (capture) is GPU-gated and deferred, so the doctrine-consistent "next task" was the
**non-GPU spine M2 sits on**: make Astel produce and serve **real, unique** splat assets end-to-end
on CPU, and drive the viewer + Truth Meter from them.

## What shipped

- **`services/api/storage.py`** — `LocalArtifactStore` (layout `{root}/{task_id}/{name}`,
  `ASTEL_ARTIFACT_DIR`, name sanitization `^[A-Za-z0-9._-]+$`), `ArtifactStore` Protocol + cached
  `get_artifact_store`. This is the object-storage seam (local now, S3 later).
- **`services/api/producer.py`** — `stable_seed` / `synth_cloud` (deterministic per-task procedural
  splat, ~48k points, valid 3DGS params) / `build_quality_report` / `produce_artifacts`. Reuses
  `libs/astel_splat_io.write_ply` via an **editable path dep** (`[tool.uv.sources]`); bumped the API
  to `requires-python >=3.12` to match the lib + the service env.
- **API wiring** — `create_generation` produces artifacts on submit (guarded; never 500s the submit);
  `GenerationResource.artifacts[]`; serving route `GET /v1/generations/{id}/artifacts/{name}`
  (400 bad name / 404 missing / traversal-safe).
- **Honest quality report** — `quality-report.json` carries `origin:"stub"` + a mandatory `caveats`
  line. The web Truth Meter renders a **STUB** pill + the caveat whenever `origin !== "measured"`.
- **Web** — viewer loads the per-task `l3.ply` on success (static sample = idle + load-failure
  fallback); Truth Meter maps the API report; Layer Stack L0–L3 derive from the SSE stage
  (`liveLayers`). Verified live: "4/8 ready", "Asset ready · 48k splats", STUB pill + caveat present.

## Two real bugs the first live browser run exposed

M1 was "verified" with vitest + static screenshots, so neither showed up before:

1. **Split-brain generation state.** `App` and `GenerationDock` each called `useGeneration()`
   (separate hook instances). The generation ran in `GenerationDock`'s instance; `App`'s stayed
   `idle`, so `succeeded` never fired and the viewer/Truth Meter never switched to the real asset.
   Fix: lifted to one instance in `App`, passed `state`/`start`/`cancel` to `GenerationDock` as props.

2. **SSE line-ending mismatch.** The client parser split records on `\n\n` only, but `sse-starlette`
   emits **CRLF** (`event: progress\r\ndata: …\r\n\r\n`). Result: every event was silently dropped
   and the stream only "ended" when the socket closed → progress never advanced. Confirmed by reading
   raw bytes through the Vite proxy (`firstChunkMs: 0`, so not a buffering issue). Fix: parser now
   matches `\r\n\r\n | \n\n | \r\r` and splits lines on `\r\n | \n | \r`, per the SSE spec. Locked
   with a CRLF unit test (the old test used LF, which is why this slipped).

**Lesson:** add a real (non-mocked) browser SSE round-trip to the verification checklist; LF-only
mocks hid a CRLF bug for a whole milestone.

## Gates

- `services/api`: ruff ✓ · mypy --strict ✓ · 17 passed / 1 skipped (temporal-gated).
- `apps/web`: eslint+tsc ✓ · 15 vitest passed (added the CRLF case).

## Honest gaps / next

- Artifacts are produced **synchronously at submit** in stub mode (fine for CPU; the durable async
  path is the Temporal engine, untouched). Status lifecycle still SSE-driven.
- Producer writes only `l3.ply` + `quality-report.json`. Not yet wired: `l0.ply`, the full `.astel`
  package (`libs/astel_format.build_minimal_package`), and `.spz`/`.sog` exports (writers exist).
- **No `/v1/captures` upload yet** — Text path only; Image/Video tabs still send a placeholder
  string. That + `.astel` packaging is the natural session-5 slice (still no GPU needed).
- Screenshot tool times out on the heavier WebGL render loop; a11y snapshots are the reliable proof.
