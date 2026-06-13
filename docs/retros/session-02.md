# Session 2 Retro — 2026-06-13 — Phase R close + M1 skeleton

## Headline

Phase R is **closed**, and M1's skeleton is **up and green**. The product is now named
**Astel** (founder decision; package format `.astel`). Six workstreams ran in parallel as
subagents; the lead (Fable 5) planned, merged, and verified.

## What was produced

**Phase R (closed):**
- `08-deep-reads.md` — method-section reads of 2DGS, PGSR, TRELLIS/SLAT, MapAnything,
  PhysGaussian, RTR-GS; adopted losses/regularizers with actual λ values + gsplat-mode mapping.
- `09-metrics-targets.md` — per-layer (L0–L7) CI-gated accuracy metrics + initial targets,
  justified from paper numbers; provenance/scale-honesty/golden-file cross-cutting gates.
- `10-task-engine-spike.md` + `experiments/task-engine-spike/` — **Temporal finalized** by a
  hands-on 3-stage toy: killed the worker mid-stage, it resumed from the activity heartbeat with
  zero custom checkpoint code; dev server ~125 MB RAM, state survives restart. Celery rejected.
- `12-trellis-import-audit.md` + `LICENSE_AUDIT.md` v2 — TRELLIS v1 gaussian head is clean
  as-is; TRELLIS.2 O-Voxel decode needs a **one-line lazy-import fork** to dodge nvdiffrast.
  SOG ref impl (MIT), Spark SH0–SH3, VGGT-1B-Commercial all closed.
- `docs/specs/manifest-v0.md` + 5 JSON Schemas — the `.astel` package format: glTF-shaped
  buffer table, per-primitive UNORM8 **provenance channel** carried to glTF/USD/.spz, mixed
  kernel-type headroom, quality-report block.
- `docs/eval/CORPUS.md` — frozen blind-eval corpus v1 (20 text / 20 image / 10 capture) +
  Bradley-Terry scoring protocol + immutability policy.
- `DECISIONS.md` → **v0.2**: 4 rows flipped 🟡→✅ (L3 refinement losses, L2 generative
  foundation, L6 physics, L4 appearance) + Temporal ✅. Two honest 🟡 remain, both GPU-gated.

**M1 (skeleton up):**
- `apps/web` — Vite + React 19 + TS strict. Spark renders the checked-in sample splat;
  Layer Stack (L0–L7) with per-layer states + scrub rail; Text/Image/Video generation dock;
  Truth Meter card; designed offline/empty states. Verified visually (premium dark studio
  aesthetic) — the founder's "UI/UX first" priority is met.
- `services/api` — FastAPI + async SQLAlchemy/SQLite + SSE; `TaskEngine` seam with
  `InProcessStubEngine` streaming L0→L3 progress and shaped metrics.
- `pipelines/stub` — procedural Gaussian-splat `.ply` generator.
- `.github/workflows/ci.yml` (web/api/stub/license-gate, inert until a remote exists),
  `infra/docker-compose.yml` (prod topology), `scripts/dev.ps1`, `ARCHITECTURE.md`.
- **All checks green:** web build + 9 vitest; api ruff/mypy-strict + 8 pytest; stub 14 pytest.

## What went well / what to fix

- ✅ Parallel-subagent fan-out (6 at once) collapsed ~2 sessions of Phase-R work into one.
- ✅ Lead verified independently before each commit (re-ran builds/tests; screenshotted the UI)
  rather than trusting agent self-reports.
- ⚠ The M1 scaffold agent hit a session limit mid-verification, leaving the web build red
  (missing `vite-env.d.ts`, `@types/node`, vitest config type) and the CI/infra/docs files
  unwritten. Caught and finished by the lead + a focused Sonnet agent. Lesson: scope the
  scaffold agent smaller, or checkpoint its work.
- ⚠ PS 5.1 read the UTF-8 `dev.ps1` without BOM and mangled non-ASCII chars (`—`, `·`),
  breaking quote pairing. Fix: keep `.ps1` scripts ASCII-only.

## Carryover

- Two 🟡 decisions remain **GPU-gated** (founder deferred GPU work): L3 2DGS-vs-3DGS+GOF A/B,
  and the TRELLIS.2→surfel distillation-fidelity experiment (R-T1, the single riskiest bet).
- Blind-eval **corpus** exists; the **harness runner** is not built yet (M1 exit item).
