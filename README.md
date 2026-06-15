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
> Status: **Phase R · M1 · M2 · M3 closed; M4 (world-awareness) in progress.** The
> real generative path is **image → 3D** (TripoSplat L2 → 2DGS L3, verified live on
> the GPU box). **Text → 3D is not built yet** (text→multiview is the next stage) —
> a text prompt today returns a placeholder shape + structured spec, clearly
> flagged. **To test the MVP, start with [docs/MVP_TESTING.md](docs/MVP_TESTING.md).**
> Runway: [docs/NEXT_STEPS.md](docs/NEXT_STEPS.md).

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
- **M2** Capture path: photos/video → splats (reality first) ✅ spine landed
  (DTU geometry numbers, COLMAP SfM, 2DGS L3 surface refinement)
- **M3** Generative path: image → splats ✅ (TripoSplat L2 → 2DGS L3); Generation
  Spec LLM stage ✅; preview/refine billing ✅. **Caveat: text → splats is not
  wired** — only the LLM spec runs for text; the geometry generator is image-only.
- **M4** World-awareness: relighting, collision, physics-materials, print path
  ← in progress (L5 solidification landed; L6 physics-material next)
- **M5** Engine plugins, SDK, MCP server, docs site
- **M6** Dynamics, scenes, hardening, launch

## Dev quickstart (current state)

```
pnpm install
pnpm run up         # one-command bring-up (web + API together)
```

Alternatives: `pnpm run dev:all` (web + API together) or `pnpm dev` (web app only;
Vite proxies `/v1` and `/healthz` to the API at :8000).

The default producer is the CPU **stub** (procedural placeholder geometry, works on any
box). The real GPU generative path (image → splats) runs on the 2×4090 box with
`ASTEL_PRODUCER=gpu` — see [docs/MVP_TESTING.md](docs/MVP_TESTING.md) for both paths.
