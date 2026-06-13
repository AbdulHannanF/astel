# DECISIONS.md — Pipeline & Stack Decisions

*v0.2 — session 2, 2026-06-13. Status legend: ✅ decided · 🟡 draft (GPU experiment required)
· ⬜ deferred to milestone. Every entry cites its research note. License policy at bottom.*

*v0.2 update: six paper deep-reads ([RA8](08-deep-reads.md)) settle the technique selections;
per-layer CI targets in [RA9](09-metrics-targets.md). Rows flipped ✅ where the method is now
fully specified; rows kept 🟡 carry a GPU smoke-test/ablation that is **deferred per user decision
2026-06-13** — the design is settled, the empirical A/B is not. Task-engine row untouched (other
workstream).*

## Per-stage decisions

| Stage | Chosen | Runner-up | Why | License | Status |
|---|---|---|---|---|---|
| Rasterization backbone | **gsplat** (nerfstudio) | 3dgrut (raster path) | Apache, most active community, built-in 2DGS/AA/AbsGS/MCMC/3DGUT — four papers pre-reimplemented permissively ([RA1](01-core-and-surface.md)) | Apache-2.0 | ✅ |
| Ray-traced/relight rendering | **3dgrut** (NVIDIA) | gsplat 3DGUT mode | Secondary rays for L4 validation + distorted cameras for capture ([RA1](01-core-and-surface.md)) | Apache-2.0 | ✅ |
| L3 representation | **2DGS surfels** (gsplat mode) | 3DGS + GOF-style extraction | Real per-splat normals (exact via ray–splat intersection) feed L4/L5; **2DGS over-smooths fuzzy content → A/B vs 3DGS+GOF needs GPU, deferred 2026-06-13** ([RA8 §1](08-deep-reads.md)) | Apache-2.0 | 🟡 |
| L3 refinement losses | MCMC budget densification + AA + reimplemented **PGSR** (single-view edge-aware normal + multi-view FB-reprojection + patch-NCC + exposure affine) + DN-Splatter monocular priors | SuGaR-style alignment | **Technique set settled**: PGSR = published leader (DTU 0.52mm); weights λ=100/.015/.15/.03 read off ([RA8 §2](08-deep-reads.md)); minimal weight subset (PGSR vs prior overlap) = deferred ablation | our code on Apache | ✅ |
| Generative foundation (L2) | **TRELLIS-image-large gaussian head** (MIT, 16 GB) | LGM (speed tier) | **Confirmed only open *native* gaussian generator** at this quality (z→K gaussians w/ tanh-bounded offset; TRELLIS.2 verified mesh-only so v1 is the head); gaussian decode dodges NC `diffoctreerast` ([RA8 §3](08-deep-reads.md)) | MIT (gaussian head only) | ✅ |
| Generative geometry prior (L3 supervision) | **TRELLIS.2-4B internal O-Voxel prior → distill to surfels** — geometry decode only; appearance guidance via MV-Adapter; prior views rendered by our own renderer (NOT nvdiffrast) | TRELLIS v1 end-to-end | **Model choice ✅** (MIT, SOTA geom+PBR); **distillation fidelity (does TRELLIS.2 geom survive surfel fit?) is the single riskiest bet → GPU de-risk deferred 2026-06-13**; nvdiffrast/nvdiffrec NC boundary in [LICENSE_AUDIT.md](LICENSE_AUDIT.md), clone-time import check pending ([RA8 §3](08-deep-reads.md)) | MIT core / NC deps excluded | 🟡 |
| Text conditioning | LLM Generation Spec → **FLUX.1-schnell** T2I → background removal → image pipeline | TRELLIS-text-xlarge direct | Meshy lesson: manufacture ideal conditioning images ([RA2](02-generative.md)) | Apache-2.0 (verify) | 🟡 |
| Multi-view guidance | **MV-Adapter** | Era3D / newer | Plug-and-play on maintained T2I bases; also texture-refinement capable ([RA2](02-generative.md)) | ⚠ verify | 🟡 |
| Capture front-end (L0/L1) | **MapAnything `-apache` ckpt** | VGGT (if ckpt relicensed) | Apache code+weights, metric output, pose-free↔posed flexible, active ([RA3](03-capture-video.md)) | Apache-2.0 | ✅ |
| Pose refinement | **GLOMAP/COLMAP BA** on MapAnything init | pure feed-forward | Classical BA still wins final accuracy on dense orbits; CPU-only → 3080-box CPUs ([RA3](03-capture-video.md)) | BSD | ✅ |
| Metric scale | **Consensus: MapAnything + MoGe-2 + SfM/EXIF**, with reported CI | DA3METRIC-LARGE swap-in | Redundancy → honest confidence interval (Truth Meter input) ([RA3](03-capture-video.md)) | Apache/MIT | ✅ |
| L7 dynamics | Own deformation-field 4DGS on gsplat (M6) | Faster-GS (Apache) as base | No turnkey permissive 4DGS exists ([RA3](03-capture-video.md)) | our code | ⬜ M6 |
| L6 physics sim | MPM-on-gaussians over **NVIDIA Warp** | Taichi / Genesis; i-PhysGaussian implicit integrator as drop-in | **Math + engine settled**: PhysGaussian kinematics (a_p=F A_0 Fᵀ, MLS-MPM, SH-rot via polar-decomp); **interior filling from our L5 SDF, not opacity ray-march** (our improvement); Warp example maturity / sandbox latency = GPU eval, deferred 2026-06-13 ([RA8 §5](08-deep-reads.md)) | Apache | ✅ |
| L6 semantics | LLM/VLM reasoning pass (Anthropic adapter) + SAM-class part segmentation | — | Own code on API; per-region material/density/friction ([RA4](04-physics-lighting-print.md)) | Apache (SAM2 verify) | 🟡 |
| L4 appearance | Per-gaussian PBR (albedo/metallic/roughness/L_ind) via **deferred shading on gsplat** + split-sum IBL + jointly-optimized env map; **driven by real 2DGS normals**; TRELLIS.2 PBR seeds generative path | RTR-GS hybrid radiance-transfer branch | **RTR-GS validates deferred-PBR-on-gaussians** (relight PSNR 28.9 ORB); we improve it w/ real L3 normals (vs its pseudo-normals); port white-light + metal-prior + staged-PBR schedule; transfer-MLP branch dropped ([RA8 §6](08-deep-reads.md)) | our code | ✅ |
| L5 solidity/print | L3 depth → **Open3D TSDF → sparse SDF → marching cubes**; **CoACD** convex decomp; trimesh mass props; .3mf/.stl via trimesh/lib3mf | GOF-style direct isosurface | All-permissive chain; SDF also feeds physics volume ([RA4](04-physics-lighting-print.md)) | MIT/BSD | ✅ |
| Web viewer | **Spark** (Three.js) | PlayCanvas engine; GaussianSplats3D | MIT, loads all our formats, per-splat edit API → Layer Inspector/Truth Meter heatmaps, mobile-fast ([RA5](05-formats-engines.md)) | MIT | ✅ |
| Export formats | .ply, .spz, .sog, glTF+KHR_gaussian_splatting (RC now, ratifies ~Q2'26), USD/USDZ, .auriga manifest; print: .3mf/.stl | — | Spec §1.5; ride standards ([RA5](05-formats-engines.md)) | open | ✅ |
| Task engine | **Temporal** (MIT; dev = single binary, prod = Postgres-backed) | Celery+Redis | Spike-proven 2026-06-13 ([RA10](10-task-engine-spike.md)): kill worker mid-stage → resumes from activity heartbeat, completed stages never re-run, zero custom checkpoint code; dev server ~125 MB RAM idle, state survives restart via `--db-filename`; Celery rejected (no clean no-admin Windows Redis path, DIY checkpoint tables) | MIT | ✅ |
| API/data/storage | FastAPI + Postgres + MinIO/S3 + SSE/webhooks | — | Spec defaults accepted §5 | OSS | ✅ |
| LLM layer | Anthropic API via model-agnostic adapter | — | Spec §5; token budget ≤ ~20k/generation logged to credit ledger | — | ✅ |

## Architecture-shaping decisions (binding unless revisited in writing)

1. **Provenance channel from day one** (✅): every gaussian carries a measured↔generated
   confidence scalar from L0 through export. Reserved in the `.auriga` manifest schema in M1
   *before* any pipeline exists. Rationale: retrofitting provenance is impossible; it's the
   Truth Meter's substrate and spec §10.4's "sacred" channel.
2. **Generate-prior-then-distill** (🟡): the generative path's geometric accuracy comes from
   distilling a TRELLIS.2 internal prior into surface splats, not from native splat diffusion.
   v0.2: model selection settled (RA8 §3 — TRELLIS.2 mesh-only confirmed, used as internal
   geometry/PBR prior only); **distillation fidelity remains the riskiest single bet** → de-risk
   experiment is the first GPU job (deferred per user decision 2026-06-13).
3. **Splat budgets are MCMC budgets** (✅): product tiers (100k/1M/5M) map to MCMC's fixed
   gaussian counts — exact, billable, predictable VRAM.
4. **Reality first** (✅, spec §9 M2): capture path ships before generative path — measured
   accuracy is the brand; the generative path then inherits trusted infrastructure.
5. **Engine plugins wrap existing renderers** (🟡): Unity/UE5 plugins add AURIGA import +
   physics auto-setup on top of proven open splat renderers; we don't write engine renderers
   from scratch (M5 re-evaluation point).

## License policy (binding)

- Ship only Apache/MIT/BSD-licensed code and weights; weights and code verified **separately**.
- NC-licensed work (Inria 3DGS lineage, DUSt3R/MASt3R, UniDepth, Metric3D, Hunyuan territory
  terms) may inform design; reimplement published math on permissive backbones; never vendored.
- Every dependency's license recorded in `docs/research/LICENSE_AUDIT.md` (created in M1 CI:
  automated license-check gate).
- Audit status: see [LICENSE_AUDIT.md](LICENSE_AUDIT.md) (v1, 2026-06-12). Closed ✅:
  MV-Adapter, FLUX.1-schnell, SAM2, Warp, SPZ (all permissive). Closed ❌: nvdiffrast/nvdiffrec
  (NVIDIA-NC → TRELLIS.2 usage boundary defined). Remaining 🔍 (clone-time, session 3):
  TRELLIS v1/v2 import graphs, SOG impl, VGGT-1B ckpt, NanoGS, Spark SH limits, Taichi/Genesis
  (moot unless Warp disappoints).

## Product decisions (added second pass, 2026-06-12 — see [RA7b](07-free-tier-consumer-strategy.md))

| Decision | Content | Status |
|---|---|---|
| **Free-tier doctrine** | Beat Meshy's free tier on three axes — generosity (cheap L0–L2 previews, private assets, free API quota), capability (capture input, relight, physics, local mode — things paid Meshy lacks), trust (Truth Meter for everyone) — NOT by out-spending on free text-to-3D credits | ✅ |
| **Blind-eval harness** | Fixed corpus (20 text / 20 image / 10 capture); blind side-by-sides vs Meshy-free, Tripo, raw TRELLIS.2. Built in M1; M3 exit-gated on ≥ raw TRELLIS.2 (all) and ≥ Meshy-free (majority). Published — it's marketing | ✅ |
| **Consumer UX first-class from M2** | Drag-drop → asset, zero visible settings by default; API underneath (spec §7 intact) | ✅ |
| **Finishing pipeline = consumer-quality workstream** | Cleanup, retries, metric scale, watertight/print checks carry M2/M3 acceptance metrics (not deferred to M4 polish). Printability is a pipeline property — our SDF path makes any generator's output printable | ✅ |
| **Positioning guardrail** | Launch messaging leads capture + print + honesty + generosity; never claims text-to-3D parity until the eval harness proves it | ✅ |
| **Dev environment** | GPU boxes stay Windows + **WSL2** (no dual-boot switching); founder runs one-time install, agent drives via SSH thereafter | ✅ |

## RA7 — orchestration note (RESOLVED ✅ 2026-06-13)

Temporal finalized by hands-on spike ([RA10](10-task-engine-spike.md), code in
`experiments/task-engine-spike/`): 3-stage toy pipeline on Windows, no admin, no Docker.
Kill-worker-mid-stage resumed from the activity heartbeat (completed stages never re-ran);
dev-server restart with `--db-filename` preserved full workflow history. Footprint fits
`astel up`: ~125 MB RAM idle, ~4 s warm start (binary is large on disk, ~553 MB unpacked —
acceptable, ships as a downloaded tool not a vendored file). Embedding plan: dev = managed
`temporal server start-dev` subprocess; prod = Temporal-on-Postgres sharing our instance.
Note: temporalio SDK requires Python ≤3.12/3.13 (not 3.14) — pin worker envs accordingly.
Celery+Redis rejected: no clean no-admin Windows Redis path, resume semantics would be DIY.
