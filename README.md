# Astel

**Geometry-accurate, world-aware, layered Gaussian splat assets — from text, images, or video.**

Astel (formerly codename AURIGA) is the successor to mesh-era generative 3D tools. Its native
and only product representation is the Gaussian splat: every generated asset is a **Layer
Stack** — sparse seed → dense metric point cloud → coarse gaussians → refined surface gaussians →
decomposed appearance → collision/solidity → physics-material semantics → (optional) dynamics —
packaged as one `.astel` asset that drops into Unreal, Unity, Blender, the web, USD pipelines,
and a 3D printer.

> Name locked **Astel** (founder decision, 2026-06-13); the package format is `.astel`.
>
> **Status — the build plan is complete (M0–M6).** Phase R and every milestone in CLAUDE.md §9
> are implemented and tested at the library/producer level. The full gate suite — ruff ·
> mypy --strict · pytest (9 Python libs + GPU pipeline + API) and tsc -b · eslint · vitest
> (web app + TS packages) — is green. What remains is **launch hardening** (CI, production
> deploy, monitoring) and the **GPU-real upgrades** each milestone honestly deferred at the
> torch boundary (real 4DGS video tracking, the text→multiview quality bridge, live LOD/scene
> wiring). See **[docs/LAUNCH_CHECKLIST.md](docs/LAUNCH_CHECKLIST.md)** and the
> **[post-M6 roadmap](docs/research/18-post-m6-roadmap.md)**.
>
> **Generation paths today:** text → 3D (SDXL/FLUX → TripoSplat L2 → 2DGS L3), image → 3D
> (TripoSplat L2 → 2DGS L3), and video → static reconstruction from the sharpest frame (L7
> dynamics tracking is honestly *not* yet performed — the asset is a static L3 with an explicit
> caveat). **To test the MVP, start with [docs/MVP_TESTING.md](docs/MVP_TESTING.md).** Runway:
> [docs/NEXT_STEPS.md](docs/NEXT_STEPS.md).

## Repository map

| Path | What |
|---|---|
| `CLAUDE.md` | The binding mission spec (constraints, layer model, milestones) |
| `apps/web` | The product web app (splat viewer, Layer Inspector, Truth Meter, Relight Studio, Physics Sandbox) |
| `services/api` | FastAPI gateway (generations API, SSE progress, captures upload, billing, Temporal seam) |
| `pipelines/` | `gpu` (real text/image/video producer on gsplat) and `stub` (CPU procedural sample generator) |
| `libs/` | Torch-free Python libraries: `astel_format`, `astel_splat_io`, `astel_eval`, `astel_llm`, `astel_solid`, `astel_appearance`, `astel_dynamics`, `astel_scene`, `astel_lod` |
| `packages/` | `manifest` (TS `.astel` reader/writer), `sdk-ts` (`@astel/sdk`), `sdk-python` (`astel_sdk` + `astel-mcp` MCP server) |
| `plugins/` | Unity package + UE5 plugin (consume the flat `engine.json` physics sidecar) |
| `tools/` | `loadtest` harness; `colmap` (gitignored local install) |
| `infra/` | docker-compose for the prod-shaped stack (Postgres/MinIO/Temporal) |
| `docs/` | Phase-R research, `DECISIONS.md`, `.astel` specs, eval corpus, architecture, retros, the MkDocs site (`docs/site`), launch checklist |

## Phases (build plan §9)

- **R — Research & decisions** ✅ closed (`docs/research/DECISIONS.md`, positioning, risk register)
- **M1 — Skeleton** ✅ monorepo, Temporal task-engine seam, FastAPI + SSE, stub pipeline, web viewer
- **M2 — Capture path** ✅ photos/video → splats (DTU geometry numbers, COLMAP SfM, 2DGS L3 surface refinement)
- **M3 — Generative path** ✅ text *and* image → splats (TripoSplat L2 → 2DGS L3), Generation Spec LLM stage, preview/refine billing
- **M4 — World-awareness** ✅ L4 relighting, L5 collision/solidity + print path (.3mf/.stl), L6 physics-material; Relight Studio + Physics Sandbox
- **M5 — Pipeline-readiness** ✅ Unity + UE5 plugins, KHR_gaussian_splatting glTF export, Python + TypeScript SDKs, MCP server, docs site
- **M6 — Dynamics & scenes** ✅ L7 dynamics core (bound into `.astel`, validated vs analytic ground truth), scene-seed layout/composition, LOD streaming, hardening + security review + launch checklist

Honest gaps (all tracked in [the launch checklist](docs/LAUNCH_CHECKLIST.md) and
[post-M6 roadmap](docs/research/18-post-m6-roadmap.md)): CI is now wired
(`.github/workflows`) but has not executed on a remote yet (the repo is local-only);
real per-frame 4DGS video tracking, live LOD streaming in the viewer, scene generation
wired into the API, and compiler-verified engine plugins (no licensed Unity/UE5 runners
here) all remain.

## Dev quickstart

```
pnpm install
pnpm run up         # one-command bring-up (web + API together)
```

Alternatives: `pnpm run dev:all` (web + API together) or `pnpm dev` (web app only;
Vite proxies `/v1` and `/healthz` to the API at :8000).

`pnpm run up` **auto-detects the GPU**: on the 2×4090 box it runs the real
generative producer (`ASTEL_PRODUCER=gpu` — text/image → SDXL/TripoSplat → 2DGS
L3, so every generation is a real prompt-conditioned splat); with no GPU it falls
back to the CPU **stub** (procedural placeholder). Force either with
`pnpm run up -- -Gpu` / `-Stub`. Generation is **asynchronous** — the request
returns immediately and real per-stage progress streams over SSE while the job
runs in the background. To drive this box's GPUs from a laptop, run
`pnpm run up -- -BindHost 0.0.0.0` and see
[docs/REMOTE_ACCESS.md](docs/REMOTE_ACCESS.md). Full path walkthrough:
[docs/MVP_TESTING.md](docs/MVP_TESTING.md).

## Splat cleanup (floater removal)

Feed-forward generators (TripoSplat L2) spray a minority of junk splats around the
real object — faint dark "smoke", oversized translucent halo blobs, elongated
"needle" streaks, and disconnected floater clusters. Because the L3 distillation
freezes positions and learns to *reproduce* the L2 appearance, it would otherwise
bake those floaters into the final asset. The generative path therefore runs a
**geometry-preserving** cleanup (`pipelines/gpu/.../splat_clean.py`) on the L2 cloud
**before** distillation (so the distillation targets — and the framing
normalization — are clean) plus a cheap final pass on L3.

Filters: opacity floor (smoke), elongation cap (needles; leaves flat surfels
alone), oversize cap (halo blobs), and **connected-component removal** — keep the
large connected cluster(s), drop only floaters that are *spatially disconnected*
from the body. The cardinal rule is **never delete real geometry**: connected
components is density-agnostic, so a thin or weakly-sampled but *attached* surface
region survives (it's still graph-connected), and worst case the cleaner
under-cleans rather than eating the object. (Statistical outlier removal — global
density threshold — is **off by default**: it deletes legitimate low-density
regions on real generated clouds; it removed ~40% of a real helmet. It's opt-in via
`ASTEL_CLEAN_SOR_ITERS`.) Every stage's removal count is logged into the run
metrics (`l2_clean` / `l3_clean`) — nothing is dropped silently.

Cleanup is **on by default**. Tune per run via env vars (no code change):

| Env var | Default | Effect |
|---|---|---|
| `ASTEL_CLEAN` | `1` | `0`/`off` disables all cleanup |
| `ASTEL_CLEAN_OPACITY_MIN` | `0.04` | drop splats fainter than this (raise to cut more smoke) |
| `ASTEL_CLEAN_MAX_ELONGATION` | `16` | drop needles above this `s_max/s_mid` (lower = stricter) |
| `ASTEL_CLEAN_MAX_SCALE_FACTOR` | `12` | drop blobs larger than `factor × median` extent |
| `ASTEL_CLEAN_COMPONENTS` | `1` | `0` disables connected-component floater removal |
| `ASTEL_CLEAN_CC_RADIUS_FACTOR` | `6` | link radius = `factor × median NN dist` (lower = stricter, risks fragmenting) |
| `ASTEL_CLEAN_CC_MIN_FRACTION` | `0.01` | keep components ≥ this fraction of the largest |
| `ASTEL_CLEAN_CC_MIN_SIZE` | `64` | absolute min splats to keep a component |
| `ASTEL_CLEAN_SOR_ITERS` | `0` | >0 enables statistical outlier removal (density-sensitive; use with care) |

A/B a single generation with `python -m astel_gpu.generative --image IMG --no-clean`
vs. without, to see the cleaned-vs-raw asset side by side.

## Running the gates

CI (`.github/workflows/ci.yml`) runs these gates on every push/PR — covering all 9
Python libs, the API, the Python SDK, the load-test tool, the web app, and the TS
packages. The GPU pipeline (torch + gsplat/CUDA) runs on a self-hosted runner via
`.github/workflows/gpu.yml`. The repo is local-only today, so the same gates are also
run by hand:

```
# Python libs (per lib under libs/*):     uv run ruff check . && uv run mypy && uv run pytest -q
# GPU pipeline (CPU subset):              uv run --directory pipelines/gpu pytest -q
# API:                                    uv run --directory services/api pytest -q
# Web app + TS packages:                  pnpm -C apps/web lint && pnpm -C apps/web test
```
