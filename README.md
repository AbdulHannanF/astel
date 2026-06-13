# Astel

**Geometry-accurate, world-aware, layered Gaussian splat assets — from text, images, or video.**

Astel (formerly codename AURIGA) is the successor to mesh-era generative 3D tools. Its native
and only product representation is the Gaussian splat: every generated asset is a **Layer
Stack** — sparse seed → dense metric point cloud → coarse gaussians → refined surface gaussians →
decomposed appearance → collision/solidity → physics-material semantics → (optional) dynamics —
packaged as one `.astel` asset that drops into Unreal, Unity, Blender, the web, USD pipelines,
and a 3D printer.

> Name locked **Astel** (founder decision, 2026-06-13). Repo folder still says AURIGA;
> rename at will — nothing in-tree depends on the folder name.
> Status: **Phase R closed. M1 closed. M2 spine landed (CPU)** — real per-task artifacts
> flow end-to-end — see [docs/NEXT_STEPS.md](docs/NEXT_STEPS.md).

## Repository map

| Path | What |
|---|---|
| `CLAUDE.md` | The binding mission spec (constraints, layer model, milestones) |
| `docs/meshy-analysis.md` | Reverse-engineering analysis of the incumbent competitor |
| `docs/research/` | Phase R: research plan, literature/ecosystem reviews, `DECISIONS.md`, risk register |
| `docs/specs/` | `.astel` package format + manifest JSON Schemas |
| `docs/eval/` | Frozen blind-eval corpus + scoring protocol |
| `apps/web` | The product web app (viewer, Layer Inspector, Truth Meter) |
| `services/api` | FastAPI gateway (generations API, SSE progress) |
| `pipelines/` | Pipeline code (stub generator today; real stages from M2) |
| `experiments/` | Spikes — graduate into the product or get deleted |
| `infra/` | docker-compose for the prod-shaped stack (Postgres/MinIO/Temporal) |

## Phases

- **R — Research & decisions** ✅ closed
- **M1** Skeleton: monorepo, task engine, API, stub pipeline, web viewer ✅ closed
- **M2** Capture path: photos/video → splats (reality first) ← in progress: CPU spine
  landed (real per-task artifacts); capture path GPU-gated
- **M3** Generative path: text & image → splats
- **M4** World-awareness: relighting, collision, physics-materials, print path
- **M5** Engine plugins, SDK, MCP server, docs site
- **M6** Dynamics, scenes, hardening, launch

## Dev quickstart (current state)

```
pnpm install
pnpm run up         # one-command bring-up (web + API together)
```

Alternatives: `pnpm run dev:all` (web + API together) or `pnpm dev` (web app only;
Vite proxies `/v1` and `/healthz` to the API at :8000).

GPU pipeline work is deferred by founder decision (2026-06-13): make everything work in
stub/CPU mode first, then scale onto the GPU boxes.
