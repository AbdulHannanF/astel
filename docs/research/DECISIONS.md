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
| L3 representation | **2DGS surfels** (gsplat mode) + normal-consistency + scale-tuned depth-distortion | 3DGS + GOF-style extraction | **A/B RESOLVED 2026-06-15 (session 13)** on real DTU scan1, matched 200k/3000/no-densification: 2DGS (λn=0.05, λdist=1e-4) **beats** raw 3DGS — overall **8.53 vs 8.76 mm**, accuracy **10.91 vs 11.52 mm** — and emits real per-splat normals for L4/L5, at a small PSNR cost (20.5 vs 21.4). Depth-distortion weight is scale-sensitive (λdist=1.0 → 27 mm collapse; 1e-4 optimal at ~600 mm depths). GOF unneeded. ([RA8 §1](08-deep-reads.md), [session-13 retro](../retros/session-13.md)) | Apache-2.0 | ✅ |
| L3 refinement losses | MCMC budget densification + AA + reimplemented **PGSR** (single-view edge-aware normal + multi-view FB-reprojection + patch-NCC + exposure affine) + DN-Splatter monocular priors | SuGaR-style alignment | **Technique set settled**: PGSR = published leader (DTU 0.52mm); weights λ=100/.015/.15/.03 read off ([RA8 §2](08-deep-reads.md)); minimal weight subset (PGSR vs prior overlap) = deferred ablation | our code on Apache | ✅ |
| Generative foundation (L2) | **TripoSplat** (VAST-AI, MIT code+weights) — single image → native 3D gaussians | TRELLIS-image-large gaussian head (MIT, 16 GB); LGM (speed tier) | **DECISIONS #2 RESOLVED 2026-06-15 (session 14):** TripoSplat adopted as the L2 prior — measured on Box A at 4.6–4.9 GB / ~11 s, cleanest dep profile (audit 14, zero NC/build deps), native gaussian output, published Elo 1137 > TRELLIS.2 992, and now **wired end-to-end** (`astel_gpu.generative`: image → L2 → 2DGS L3 distillation, finite output, 23.1 dB held-out self-consistency). Replaces both the TRELLIS-v1-head L2 plan and the TRELLIS.2-distill prior below. **Deferred (not blocking):** a multi-model PSNR/SSIM/LPIPS head-to-head vs the TRELLIS-v1 head — needs a multi-view generative test corpus (none exists yet) + a TRELLIS-v1 install (cu128/torch2.11 wheel risk, doc 13 R-T9). | MIT (code + weights) | ✅ |
| Generative geometry prior (L3 supervision) | **SUPERSEDED by TripoSplat L2 (row above)** — generative L3 = 2DGS distillation of the TripoSplat L2 over orbit renders (`astel_gpu.generative`), no separate mesh prior needed | TRELLIS.2-4B O-Voxel prior → distill to surfels (original plan); TRELLIS v1 end-to-end | **R-T1 (the single riskiest bet) RETIRED 2026-06-15 (session 14):** the planned TRELLIS.2-mesh→surfel distillation is no longer on the critical path — TripoSplat outputs native gaussians directly and the L2→L3 distillation produces surface-aligned surfels end-to-end. TRELLIS.2 stays as a *future* higher-geometry-fidelity option only; its nvdiffrast/nvdiffrec NC boundary ([LICENSE_AUDIT.md](LICENSE_AUDIT.md)) is moot while unused. | MIT core / NC deps excluded | ✅ |
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

## 2026-06-14 — GPU stack: native Windows (reverses §"Dev environment"/C6)

**Decision:** The GPU stack runs on **native Windows** (CUDA Toolkit 12.9 +
Visual Studio 2026 MSVC), not WSL2. This reverses the product-decision row
"Dev environment" (✅ WSL2, no dual-boot). Decided by the agent per Operating
Rule §10.2 (decide + document); it changes neither the §1 binding constraints
nor cost/licensing.

**Why:** On Box A (`THREADRIPPER-48`, 2×4090) WSL2 is hard-blocked —
virtualization is off in firmware and "Virtual Machine Platform" is disabled
(a physical BIOS action + reboot to enable). The founder instead provisioned
native CUDA 12.9 + VS 2026 and signalled it ("it already has cuda 12.9"). Native
is ready now; WSL is not. **Validated empirically session 7** ([retro](../retros/session-07.md)):
`torch 2.11.0+cu128` + `gsplat 1.5.3` compile and train on this box (render-then-refit
smoke PSNR 8.2→45.6 dB), and the API produces a real optimized `l3.ply` via
`ASTEL_PRODUCER=gpu`.

**Operational consequences (binding for the GPU stack):**
- Build CUDA extensions through the VS env (`vcvars64.bat -vcvars_ver=14.38` —
  a VS2026 toolset-resolution quirk on this box). RTX 4090 = Ada → arch `8.9`.
- GPU code lives in its own uv project (`pipelines/gpu`); torch never enters the
  API/libs CI envs. The API reaches GPU work via **subprocess** through
  `pipelines/gpu/run-python.cmd` (the VS-env launcher), not by import — keeps CPU
  gates green and makes the GPU path work from a normal shell.
- `scripts/setup-gpu-env.ps1` reproduces the env (incl. two venv-local patches
  documented in `pipelines/gpu/README.md`).

**Known fragility / future hardening:** runtime currently needs the VS compiler
on PATH (a torch-2.11 JIT-recompile quirk: `cpp_extension` runs `where cl` on
every gsplat import even off a warm cache). The two venv patches are
torch-2.11-specific symptoms. Hardening candidate (session 8): AOT-built gsplat
or a stable torch (2.7/2.8 cu126) → no runtime compiler, patches likely vanish.

**Open risk for M3 (not M2):** TRELLIS / flash-attn / nvdiffrast-class
extensions are painful or partly unavailable on Windows. The M2 capture stack
(gsplat, COLMAP, MapAnything, Open3D, Warp) is fine on Windows; re-evaluate the
generative path's Windows feasibility when M3 starts.

## 2026-06-14 (session 8) — first ground-truth geometry eval + COLMAP installed

**Synthetic controlled-ground-truth eval tier (`astel_gpu.synthetic_eval`).**
Added an eval that renders a KNOWN object (a deterministic sphere shell, longest
axis exactly 0.20 m by construction) from known poses, refits a fresh gaussian
cloud to the renders, and measures REAL Chamfer (mm) + scale against the known
geometry. This produces the **first non-`None` `geometric_error` and `scale`**
in the quality-report pipeline (the Truth Meter's reason for being). It is a
SEPARATE eval tool: the API's GPU producer (`astel_gpu.produce`) keeps its
honest `None`s — it has no ground truth. Honesty preserved: the report's caveats
state plainly this is a controlled synthetic measurement, not real-world capture
accuracy. The headline Chamfer is over surface-defining gaussians (opacity >
0.5); the raw all-means value is reported alongside.

**First measured baseline (informs the L3 🟡 decision, row "L3 representation").**
Raw 3DGS refit, no densification, no surface regularization, on the 0.20 m
object (1500 iters, 4 k gaussians, 10 views): PSNR 6.7→32 dB; **surface
coverage** (GT→nearest refit) **≈ 15 mm**, but **precision** (refit→nearest GT)
**≈ 165 mm opacity-filtered / 220 mm raw** — i.e. the surface is covered but
plagued by floaters. This is the first in-repo quantitative evidence that raw
3DGS means are not surface-faithful, strengthening the case for the surface-
aligned L3 representation (2DGS surfels + PGSR-class normal/depth regularization,
rows above). It does NOT by itself resolve the 2DGS-vs-3DGS+GOF A/B (that needs
fuzzy real content on GPU) — but it gives a concrete number future L3 work must
beat. Methodology note: capture-path evals must be **metric-aligned** (camera
orbit ≈ 2.5× object size, init spread ≈ object size); a mis-framed object makes
geometric error a framing artifact, not a fidelity measure.

**COLMAP installed (SfM front-end, row "Pose refinement").** COLMAP
**4.1.0.dev0** (official `colmap-x64-windows-cuda.zip`, CUDA build) installed to
`tools/` (gitignored) on Box A; the binary launches cleanly with CUDA (DLL/CUDA
runtime resolve — the common failure for a downloaded Windows binary). This
smokes the install only. A **functional SfM reconstruction** smoke (feature
extraction → matching → mapping → registered-pose count) is deferred to the
real-capture session: it needs textured real images (the founder's orbit videos)
to be meaningful — running it on the low-texture synthetic renders would not be
a representative test. MapAnything (feed-forward L0/L1) likewise awaits real
captures. Both are infra-ready; the measured *real-world* numbers are the next
gate.

## 2026-06-14 (session 9) — DTU MVS adopted as the internal geometry GT benchmark

**Pivot: public capture datasets instead of blocking on founder-filmed video.**
The M2 deliverable — the first *real-world* measured Chamfer + metric scale — was
gated on the founder filming the CORPUS.md C01–C10 orbit videos. Founder directive
(2026-06-14): source real capture data from public datasets instead. A Sonnet
research-scout agent surveyed the options (DTU, Tanks & Temples, Mip-NeRF 360, CO3D,
3DGS `tandt_db`); findings and the call:

- **DTU MVS — CHOSEN.** Real structured-light-scanner ground-truth point clouds in
  **millimetres** (metric), 49–64 calibrated views per tabletop-object scene. The only
  verified option giving *real GT geometry + known metric scale + a single-scene
  download* simultaneously. License: **unstated** on the official DTU page — treated as
  "research convention, internal-benchmark-only": we do NOT redistribute it, do NOT
  train any shipped model on it, and do NOT publish DTU-derived numbers as product /
  marketing (Truth Meter) claims without explicit clearance. Pure engineering accuracy
  gate. First pull: `SampleSet.zip` (6.43 GB, images+calib for scans 1 & 6 + eval code)
  + `Points.zip` (6.49 GB, GT `.ply` in mm for all scans), direct HTTPS, no signup.
- **Tanks & Temples / 3DGS `tandt_db` — REJECTED (for now).** Real laser GT, but the
  license is explicitly **non-commercial research only** (tanksandtemples.org/license) —
  genuine licensing exposure for a commercial venture. Avoided beyond a throwaway "does
  the pipeline run" smoke; no T&T-derived number enters any product surface.
- **CO3D — REJECTED.** GT is COLMAP-derived (not independent) and CC-BY-NC
  (non-commercial): fails both the GT-quality and the license bar.
- **Mip-NeRF 360 — fallback only.** Real 360° orbit + COLMAP poses, permissive-ish, but
  **no GT scan** → can smoke the real-orbit *pipeline* but yields no Chamfer number.

**What DTU proves vs. doesn't (honesty channel).** DTU yields the first real-world
geometric-accuracy number (Chamfer in mm vs. a real scanned object, metric scale known) —
the actual Truth Meter deliverable, replacing the synthetic baseline. It does NOT prove
the *casual-phone / pose-free / metric-scale-from-monocular-depth* story: DTU ships
calibrated metric poses, so it validates the COLMAP→splat→Chamfer accuracy path, not the
MapAnything pose-free path on handheld video. That sub-claim still awaits the founder's
C01–C10 captures (or a pose-free public set). DTU is a **separate engineering benchmark,
NOT part of the frozen blind-eval corpus** (CORPUS.md v1 stays untouched per its §5).

**Scale handling.** COLMAP reconstructions are scale-free; DTU's calibrated cameras are
metric (mm). We use DTU's per-view projection matrices (`pos_NNN.txt`, decomposed to
K[R|t] in mm, GT frame) directly, so the fit lands in the GT frame with NO registration.
This makes scale metric *by construction* (not estimated) — the pose-free path is what
will produce a real scale-ESTIMATION number later.

**MEASURED RESULT (session 9, scan1, RTX 4090).** Raw 3DGS (200k gaussians, 3000 iters,
49 views @ 400×300, ~4 min): train PSNR 5.6→**23.3 dB**; vs DTU's real structured-light
scan — **completeness (GT→fit) ≈ 3.85 mm** (the trustworthy real-world surface-COVERAGE
number — raw 3DGS covers a real scanned object's surface to ~4 mm) and **accuracy
(fit→GT) ≈ 18.9 mm** (inflated: the GT `stl_total` scan and our box eval region both
include the turntable/background, and raw 3DGS leaves floaters in free space). The
asymmetry (good completeness, poor accuracy) mirrors and strengthens the session-8
synthetic finding — concrete real-world evidence for the surface-aligned L3 + masking.

**KEY ENGINEERING LESSON — `spatial_lr_scale`.** The first run gave 6 dB (no
convergence): `optimize()`'s `lr=5e-3` was tuned for the synthetic ~unit-scale scene, but
DTU coords are in **millimetres** (object ~100 mm), so position/scale steps were ~1000×
too small to move gaussians across the object. Fix: per-param-group lr with means+scales
scaled by the scene extent (the standard 3DGS `spatial_lr_scale`); default 1.0 keeps the
synthetic/smoke callers unchanged. PSNR jumped to 18.7 dB (800 it) → 23.3 dB (3000 it).
Generalized `RenderInputs` for non-square images (DTU 1600×1200) in the same pass.

**Honest gaps (next refinements).** (1) Exact DTU **ObsMask** (`.mat`, needs scipy) to
isolate the object instead of the box proxy — tightens `accuracy`. (2) Background masking
or full-scene modelling so PSNR isn't background-capped. (3) Held-out-view PSNR. (4) The
**COLMAP SfM front-end** (`colmap_runner`, built + unit-tested) on the same images, for a
pose-from-images validation + pose-accuracy check vs DTU's GT poses. New code, all
license-clean + gates green: `colmap_io`, `colmap_runner`, `dtu`, `capture_eval`,
`metrics.chamfer_distance_chunked` (VRAM-safe for the 2.88M-point GT).

## 2026-06-14 (session 10) — M2 capture gaps closed (SfM front-end + DTU protocol)

Closed the session-9 gaps; the capture path now has both a validated SfM front-end
and a protocol-correct geometry number.

**COLMAP SfM front-end validated (`capture_sfm`).** Ran the built `colmap_runner` on
the 49 real DTU scan1 images (GPU SIFT → exhaustive match → mapper → undistort, ~55 s):
**49/49 images registered**, 26,921-point sparse cloud (L0). Then aligned COLMAP's
scale-free camera centres to DTU's metric GT centres via **Umeyama** similarity:
**pose RMSE 0.886 mm** (median 0.76, max 1.54) across all 49 — COLMAP recovers the real
camera rig to sub-millimetre. This closes the functional-SfM smoke deferred since
session 8. (capture_eval still uses DTU's GT poses to isolate splat geometry from pose
error; the two are complementary.)

**DTU-protocol geometry (`capture_eval` rewritten).** Replaced the session-9 box proxy
with DTU's official **ObsMask + Plane** masking (`PointCompareMain.m`): accuracy = fitted
gaussians in the ObsMask observable volume → nearest GT; completeness = GT in the
observable OBJECT volume (ObsMask ∩ above-plane) → nearest gaussian; per-point distances
capped at 60 mm; **held-out-view PSNR** (fit on 42 train views, measured on 7 unseen). We
intersect ObsMask with the plane for completeness (DTU's leaderboard uses above-plane
only) because Astel reconstructs the object, not the full scene — documented deviation.
Needed scipy (for the `.mat` masks) — added to `pipelines/gpu` deps.

**MEASURED RESULT (scan1, 200k gaussians, 3000 it, 42 train views, ~168 s):** held-out
PSNR **21.5 dB**; **accuracy 11.36 mm, completeness 6.10 mm, overall (DTU mean) 8.73 mm**
vs the real scan. The accuracy dropped from the box-proxy's 18.9 mm once the ObsMask
excluded out-of-volume floaters; it is still high for raw 3DGS (no densification / surface
reg) — the concrete real-world baseline the surface-aligned L3 (2DGS/PGSR) must beat.

**Remaining (real next steps).** The L3 2DGS-vs-3DGS+GOF A/B on this scan (must beat 8.73
mm overall / 11.36 mm accuracy); optional full-scene modelling so PSNR isn't
background-capped; more DTU scans for a corpus number. New tested code: `capture_sfm`,
`dtu.{umeyama,load_obsmask,load_plane,points_in_obsmask,points_above_plane}`,
`metrics.nn_distances`. Gates green (ruff · mypy 24 files · 43 pytest).

## 2026-06-15 (session 11) — M3 entered: TripoSplat adopted as lead L2 candidate

Started M3 (generative path). Cleared the first two gated steps from
[13-m3-readiness](13-m3-readiness.md) §4; the founder-gated steps (Generation Spec
LLM stage) are untouched (no API key used, no spend).

**Step 1 — TripoSplat triage (no-GPU), GO.** Full import-graph + license audit in
[14-triposplat-triage](14-triposplat-triage.md) (method/rigor matches audit 12). Findings:
`VAST-AI-Research/TripoSplat` live (MIT **code and weights**, confirmed in-repo, commit
2026-06-02); entire codebase 4 files / ~2.5k LOC; **zero** NC/build-heavy deps
(nvdiffrast/kaolin/spconv/flash-attn/xformers/pytorch3d all absent) — strictly cleaner
than TRELLIS-v1's gaussian head. Only non-pure-Python op is `torchvision.ops.deform_conv2d`
(ships precompiled). Single image → native 3D gaussians (≤262,144, learned adaptive
density), `.ply`/`.splat` export. This is the cleanest candidate audited.

**Step 2 — Windows install spike on Box A, PASS.** TripoSplat runs natively: `torchvision`
added from the existing cu128 index (`0.26.0+cu128`, matches torch 2.11) plus
`safetensors`/`tqdm`/`huggingface-hub`; weights (~3.6 GB) downloaded to gitignored
`pipelines/gpu/models/triposplat`; repo vendored to gitignored `pipelines/gpu/external/`.
A single-image inference (`triposplat_spike.py`, via the `run-python.cmd` launcher) produced
65,536 gaussians in **11.4 s at 4.6 GB peak VRAM** — far under the 24 GB ceiling, leaving
room for a co-resident L3 refine. No CUDA build, no flash-attn/xformers needed (attention is
plain `F.scaled_dot_product_attention`) → **R-T9 resolved for this candidate**; R-T1/R-T7
strongly de-risked.

**Decision:** adopt **TripoSplat as the lead L2 generative prior**, pending the step-3
bake-off (held-out-view PSNR/SSIM/LPIPS + blind corpus) which formally resolves DECISIONS #2.
Fallback ladder unchanged (TRELLIS-v1 head → TRELLIS.2 distillation).

**Known defect to fix in the production wrapper (not a blocker):** TripoSplat's own
`Gaussian.save_ply` emits non-finite opacity for ~11% of points (`inverse_opacity_activation
= log(x/(1-x))` saturates at fp16 `x==1.0` → `inf`); xyz/normals/color/scale/rotation are
finite. The L2 wrapper (built in the bake-off step) must clamp opacity before export so
downstream consumers (eval, exporters, viewer) don't ingest `inf`.

**Two infra notes:** (1) `huggingface_hub`'s hf-xet path hung on the two largest checkpoints
on this box — `HF_HUB_DISABLE_XET=1` (or a direct `urllib` stream) is the reliable download
path here. (2) `triposplat_spike.py` is intentionally a spike (uses the upstream `save_ply`);
it graduates into a typed, opacity-sanitised `l2_triposplat` module during the bake-off.

## 2026-06-15 (session 12) — L2 TripoSplat wrapper graduated; opacity defect fixed

M3 step 3a (the *graduate* half of step 3). The session-11 spike is now production:
`pipelines/gpu/src/astel_gpu/l2_triposplat.py` (typed, `mypy --strict` + `ruff` clean,
2 new CPU tests; suite **45 passed**). It converts the vendored TripoSplat `Gaussian` to
our `astel_splat_io.cloud.SplatCloud` and writes via `write_ply` — **not** upstream's
`Gaussian.save_ply`. The known inf-opacity defect (note (2) above) is fixed by taking the
sigmoid-activated `get_opacity ∈ [0,1]`, clamping to `[1e-6, 1-1e-6]`, and recomputing the
logit (same path as `export.to_splat_cloud`); xyz/f_dc/log-scale/wxyz-rotation come from
upstream's `_get_ply_data(transform)` unchanged (correct coordinate transform preserved).
**Measured on Box A** (building_stone_house, 65,536 gaussians, 20 steps): 11.1 s, 4.59 GB
peak, `n_nonfinite_opacity_logit == 0`, `n_nonfinite_xyz == 0` — defect eliminated, measured.

**DECISIONS #2 is NOT yet resolved**: this is only the wrapper + clean-output proof. The
formal L2-prior pick still needs the *scoring* half — input-view reconstruction
PSNR/SSIM/LPIPS for TripoSplat (single-image generator → input-view, not novel-view, since
it takes no held-out real view) plus a TRELLIS-v1-head comparison point if one is warranted.
TripoSplat's lead is reinforced (clean finite output, 4.6 GB, 11 s) but not yet locked.
`l2_triposplat` is not wired into `produce`/API — that is step 4, gated on the L3 surface A/B.

## 2026-06-15 (session 13) — L3 surface A/B RESOLVED: 2DGS beats 3DGS on real DTU

The long-open L3 representation decision (DECISIONS #1, 🟡 since 2026-06-13) is now ✅,
resolved by measurement on real DTU scan1, not vibes. New module
`pipelines/gpu/src/astel_gpu/l3_refine.py` adds gsplat-native 2DGS refine
(`render_2dgs_train`/`render_2dgs_colors`/`optimize_2dgs`) with the two standard 2DGS
regularizers — normal consistency (rendered vs depth-derived normals) and L1 depth
distortion — over the existing RGB `L1 + D-SSIM` loss. `capture_eval` gained a
`--representation {3dgs,2dgs}` switch so both arms share an **identical** init cloud, DTU
ObsMask/Plane geometry protocol, and held-out PSNR split — only the representation + losses
differ. Pure `surface_reg_loss` seam is CPU-unit-tested (4 new tests; suite **49 passed**).

**Measured A/B (Box A 2×4090, 200k gaussians, 3000 iters, no densification, seed 20260614):**

| Arm | overall mm | accuracy mm | completeness mm | held-out PSNR |
|---|---|---|---|---|
| 3DGS (raw baseline) | 8.76 | 11.52 | 6.00 | **21.41** |
| 2DGS λn=0.05, λd=0 | 9.48 | 13.06 | 5.90 | 20.59 |
| 2DGS λn=0.05, λd=1.0 | 27.11 | 30.24 | 23.99 | 8.12 |
| **2DGS λn=0.05, λd=1e-4** | **8.53** | **10.91** | 6.15 | 20.47 |
| 2DGS λn=0.05, λd=3e-4 | 9.07 | 11.91 | 6.23 | 20.21 |

The fresh 3DGS arm (8.76 mm) reproduces the session-10 baseline (8.73 mm) → fair comparison.
**Finding:** normal consistency alone slightly *hurts* geometry (9.48), but adding a
scale-appropriate depth-distortion term (which concentrates surfels along the ray) flips it:
2DGS edges out 3DGS on both overall and accuracy while delivering real surfel normals.
PSNR is ~1 dB lower — the right trade for a geometry-accurate splat product.

**Decision:** L3 = **2DGS** with normal + distortion regularization. GOF extraction (the
runner-up) is NOT needed and stays unimplemented. **Caveat (honest):** the optimal λdist is
**scene-scale-dependent** (catastrophic at 1.0 on a ~600 mm metric scene; 1e-4 optimal) —
a dimensionless scale-normalized λdist is future work before this generalizes across scenes;
both arms ran WITHOUT densification, so this isolates the representation+regularization effect,
not the fully-productionized pipeline. λdist=1e-4 is DTU-scan1-specific, recorded as such.

## 2026-06-15 (session 14) — generative L2→L3 wired; DECISIONS #2 resolved; R-T1 retired

M3 step 4. New module `pipelines/gpu/src/astel_gpu/generative.py` wires the full
generative path end-to-end: **image → TripoSplat L2 (native gaussians) → normalise to
unit frame → render an orbit of synthetic views → 2DGS L3 distillation** (the chosen L3
representation from session 13). Because a generated object has no GT scan, the L3 is
distilled from the L2 generator's OWN multi-view renders; the reported number is held-out
**self-consistency / distillation fidelity**, never accuracy-vs-reality (the quality report
keeps `geometric_error`/`scale` honestly `None`, `generated_ratio=1.0`). New inverse
converter `export.gaussian_params_from_splat_cloud` + pure `normalize_params` seam (both
CPU-tested; suite **51 passed**).

**Measured (Box A, building_stone_house, 65,536 gaussians, 24 orbit views, 1500 refine iters):**
L2 65,536 → L3 65,536 surfels, **held-out self-consistency PSNR 23.13 dB**, refine 20.3 s,
peak VRAM 4.93 GB, output PLYs fully finite. The 2DGS L3 now carries real per-splat normals
for L4/L5.

**DECISIONS #2 RESOLVED → L2 prior = TripoSplat** (table row updated): MIT code+weights,
4.6–4.9 GB / ~11 s on Box A, cleanest dependency profile (audit 14), native gaussian output,
published Elo > TRELLIS.2, now proven end-to-end. **R-T1 (TRELLIS.2-mesh→surfel distillation
— the single riskiest bet) RETIRED**: it is off the critical path; TripoSplat + the L2→L3
distillation deliver surface-aligned surfels without it. **Deferred (non-blocking):** a
multi-model PSNR/SSIM/LPIPS head-to-head vs the TRELLIS-v1 head — needs a multi-view
generative test corpus (none exists) and a TRELLIS-v1 install (cu128 wheel risk). Committing
TripoSplat now on the evidence in hand per §10.2 (decide + document, don't stall).

**Honest gaps:** distillation runs without densification at 1500 iters (23 dB is good, not
hero-tier); generative L3 uses normal-only reg (λdist left 0 — its scale-tuning is per-scene,
session 13); not yet wired into the API `produce` path or `.astel` packaging (that is the
next integration step); single test image so far.

## 2026-06-15 (session 15) — Generation Spec LLM stage scaffolded on fixtures (M3 step 5)

New library `libs/astel_llm` implements CLAUDE.md §5's model-agnostic LLM layer and the
text-pipeline's prompt→Generation Spec stage (CLAUDE.md §4), built entirely **offline** —
no Anthropic API key, no spend (founder gate R-O2 untouched). External API facts re-verified
live this session via the claude-api reference (training data 5 months stale): Haiku 4.5
`claude-haiku-4-5` $1/$5, Sonnet 4.6 `claude-sonnet-4-6` $3/$15, Opus 4.8 `claude-opus-4-8`
$5/$25; structured JSON via `output_config={"format":{"type":"json_schema","schema":…}}`
(objects need `additionalProperties:false`, no numeric/length constraints — validated in code
instead); prompt caching via `cache_control` on system blocks; `messages.count_tokens`.

Modules: `spec.py` (`GenerationSpec` — object_class/parts/materials/style/`target_scale`
with an explicit user-overridable confidence band/symmetry + the Anthropic-compatible JSON
schema), `adapter.py` (`LLMAdapter` protocol; **`FixtureAdapter`** replays cached completions
keyed by hash of `(model, system, user)` — the default, needs no key; **`AnthropicAdapter`**
lazy-imports the SDK behind an optional `[live]` extra, constructed only when a key exists),
`generation_spec.py` (`build_generation_spec(prompt, adapter)` → validated spec + credit-ledger
row; Haiku default; frozen system prompt + schema so prompt caching applies), `pricing.py`
(verified rates + cache-discount math + `ledger_entry`). Gates green: ruff · mypy --strict
(9 files) · **14 pytest**, all offline.

**Founder gate (R-O2) — the ONLY remaining M3 step:** to enable real calls, the founder sets
`ANTHROPIC_API_KEY` + a spend cap and `uv sync --extra live`, then runs one live
`AnthropicAdapter` call; the stage code is identical fixtures-vs-live. Estimated
**~$0.02–0.035/generation** (Haiku, cached system prompt), ~$50–350/mo at 1k–10k generations —
under the $1k/mo flag. No paid call is made until the founder approves, per the agreement.

**M3 status:** steps 1–5 of [13-m3-readiness](13-m3-readiness.md) §4 are now complete in code
(triage ✅, install spike ✅, L2 graduate+bake-off/DECISIONS#2 ✅, L3 A/B+L2→L3 wiring ✅,
Generation Spec scaffolded ✅). The remaining work is integration (wire the generative pipeline
+ LLM stage into the API `produce` path and `.astel` packaging) and the single founder gate
(API key) — not new research.

## 2026-06-15 (session 16) — M3 integration pt.1: GPU producer artifact parity + generative image path wired to the API

The GPU producer now emits the **same `.astel` artifact contract** as the CPU stub and runs
the **real generative image path through the production API→subprocess seam**. New torch-free,
CPU-tested `astel_gpu.packaging.write_layer_stack(SplatCloud)` writes `l0.ply`/`l3.ply`/`l3.spz`/
`l3.sog`/`package.astel`/`quality-report.json` (+`l2.ply` for generated assets), binding L0+L3
with per-gaussian provenance via `astel_format.build_minimal_package` (added as a GPU dep —
pure-python, no CUDA). `astel_gpu.produce` dispatches by modality: `image`+`--image` → real
`run_l2_to_l3` (TripoSplat L2 → 2DGS L3); else the render-then-refit smoke. API
`produce_artifacts_dispatch` gained `capture_id` and resolves the uploaded `source*` image from
the store, passing `--image` (local-fs seam; S3 would download first); the stub default path is
byte-for-byte unchanged.

**Measured on Box A (real CUDA):** smoke 8k/300it → 7-artifact contract, 41.8 dB, 2.4 s, 0.17 GB;
generative (`creature_butterfly.webp`, 500 refine it) → L2 65,536 gaussians (11.1 s, 4.59 GB, 0
non-finite) → L3 65,536 surfels (8.1 s, 4.93 GB), held-out self-consistency 18.14 dB, 8-artifact
contract incl `l2.ply`, all PLYs finite, `package.astel` round-trips honest (`chamfer=None`,
`measured_fraction=0.0`). **Real end-to-end (no mocking, `ASTEL_PRODUCER=gpu`):**
`produce_artifacts_dispatch` invoked the live `run-python.cmd` subprocess and stored all 7
artifacts — the production seam works. Gates green: GPU ruff·mypy(33)·**56 pytest** (54 CPU + 2
GPU, 5 new packaging); API ruff·mypy(19)·**30+1 pytest** (2 new). No founder gate touched.

**Honest gaps:** the `astel_llm` Generation Spec stage is still not wired into the API text path
(session 17 — torch-free, lives in the API env); text modality runs the smoke (no prompt
conditioning until a text→multiview stage exists); generated assets' geometric_error/scale stay
honestly `None`. Nothing committed yet (sessions 7–16 on the single "Beta" commit).

## 2026-06-15 (session 17) — M3 integration pt.2 (final): Generation Spec stage wired into the API text path

The text pipeline now runs prompt → `GenerationSpec` on submit. New
`astel_api.generation_spec_stage`: `run_generation_spec_stage` builds the spec via `astel_llm`
and stores `generation-spec.json` (spec + credit-ledger row); `apply_spec_scale_to_report` threads
the spec's `target_scale` into the quality report's `scale` block
(`method:"llm-estimate"`, `source:"generation-spec"`, with the user-overridable confidence band) —
the first non-`None` scale the Truth Meter can show for a generated asset, honestly flagged.
`create_generation` runs the stage after produce; `astel-llm` added as an API dep (torch-free).

**Founder gate R-O2 — double-gated, no silent spend:** OFFLINE by default (`FixtureAdapter`, zero
cost). LIVE (`AnthropicAdapter`, real spend) requires BOTH `ASTEL_LLM_LIVE=1` AND
`ANTHROPIC_API_KEY` — a key present for other reasons can never trigger a paid call. An unseen
prompt with no cached fixture degrades gracefully (`generation-spec.json` `status:"skipped"`,
reason names R-O2); the generation still completes. Gates green: API ruff·mypy(21)·**35+1 pytest**
(5 new). No founder gate touched.

**M3 integration COMPLETE in code:** generative image path runs through the API to a full `.astel`
package (session 16) + Generation Spec runs in the text path (this session). The only remaining
M3 item is the founder's API key (R-O2) — not new code. **Next: M4 (world-awareness — L4/L5/L6).**

## 2026-06-15 (session 18) — M4 entered: L5 solidification core (splat→SDF→mesh→mass props→.stl)

First M4 step. New torch-free, CPU-only lib `libs/astel_solid` implements the print-path / physics-
volume / collision spine from row 31: `oriented_point_sdf` (IMLS over scipy KDTree knn, outward
normals ⇒ negative-inside), `extract_isosurface` (skimage marching cubes, re-wound outward),
`compute_mass_properties` (volume/COM/inertia via signed-tetra divergence-theorem integrals,
vectorised numpy), `write_binary_stl`, and `solidify`/`surfel_normals` (per-splat outward normal =
thinnest 2DGS axis, centroid-oriented). **Per §1.2 this surface is internal scaffolding only — never
the asset.** Validated vs analytic solids: **unit cube exact** (V=1, COM=0, I=diag(1/6) to 1e-6 — the
math check); **sampled sphere r=0.5 through the full 64³ pipeline** V=0.5014 vs 0.5236 (4.2% low),
COM≈4e-3, inertia diag≈0.043 vs 0.0501 (~14% low), near-isotropic — the honest discretization bias of
a faceted inscribed MC polyhedron. Deps permissive (numpy/scipy BSD, scikit-image 0.26 BSD). Gates
green: ruff·mypy(11)·**10 pytest**. **Deferred (row 31):** Open3D TSDF, CoACD convex decomp, .3mf,
printability checks — follow-on sessions. **Not yet wired into the producer/.astel package.** No
founder gate touched. **Next: wire L5 into the producer; then L6 physics-material (reuse astel_llm).**

## 2026-06-15 (session 19) — L5 wired into the GPU producer (l5.stl + mass props per asset)

`astel_solid` is now product-integrated. `astel_gpu.packaging.write_layer_stack` derives
`surfel_normals` from the L3 splats → `solidify` → writes `l5.stl` + `l5-mass.json` and threads a
`solidity` block (volume/mass/COM/inertia diagonal + mesh/SDF stats) into the quality report.
**Best-effort** (broad try/except like `.sog`): a cloud that won't solidify skips L5, never failing
the asset (the surface is scaffolding; the asset stays splats, §1.2). `astel-solid` added as a
torch-free `pipelines/gpu` dep. **Verified on a real 65k cloud** (pirate-ship image, self-consistency
28.56 dB): watertight mesh 7,855 verts / 14,881 faces, `l5.stl` = 744,134 B = exactly `84+50·14881`
(valid binary STL), volume 3.77 model-units³, **anisotropic inertia (4.61, 1.53, 5.42)** — low about
the long hull axis, physically correct. Gates green: GPU ruff·mypy(33)·**55 CPU pytest** (+1 seam).
**Honest gaps:** mass/volume in MODEL units (metric grounding via the scale stage is a follow-on);
L5 not yet a *bound* `.astel` manifest layer (loose artifacts + report block for now); centroid
outward-normal heuristic (star-shaped only). No founder gate touched. **Next M4: L6 physics-material
(reuse astel_llm), L4 relighting, metric-scale L5, CoACD+.3mf+printability.**

## 2026-06-15 (session 20) — M3 closed: preview/refine credit-metering (billing semantics)

The third and final M3 deliverable (build plan §9 M3: "preview/refine billing semantics"). New
pure module `services/api/src/astel_api/billing.py` meters the layer stack as credits per CLAUDE.md
§7 + `meshy-analysis.md`: **L0–L2 previews cheap (1/1/2 credits), L3 the main spend (20), L4–L7 +
print add-ons**, `1 credit == 1¢` (notional internal unit — no external spend, not a §10.2 cost
item). Mirrors Meshy's two-stage model: `POST /v1/generations` gains `mode` (`preview`|`refine`,
default `refine`) + optional `refine_of`; a **keyed refine bills only the L3+ increment and never
re-charges (or re-runs the LLM spec for) the preview**. Every generation stores
`credit-ledger.json` (schema `astel.credit-ledger/v0`) + returns a `billing` summary; `GET
/v1/pricing` publishes the schedule; the measured Generation-Spec token cost folds in as an
`LLM_SPEC` credit line (ceil to ≥1 credit). `generations` gains `mode`/`refine_of`/`credits`
(Alembic `a1b2c3d4e5f6`).

**Honesty:** the stub producer computes the full stack regardless of tier, so a preview has an
unpaid L3 on disk — the ledger emits a caveat naming delivered-but-unbilled layers rather than
hiding it; the real GPU path can gate production by tier later (billing is already correct for it).
**Verified live** (uvicorn, real HTTP): preview = 1 credit; standalone refine = 21 (L0+L3); keyed
refine = 20 (L3 only); preview+keyed-refine = 21 = standalone refine (no double-charge). Gates
green: API ruff · mypy --strict (23 files) · **51 pytest** (+16: 11 unit billing, 5 endpoint).
Alembic upgrade head applies on a fresh DB to the expected columns. No founder gate touched.

**M3 is now COMPLETE end-to-end** (generative L2→L3 + Generation Spec + billing). Design doc:
[`architecture/billing.md`](../architecture/billing.md). Not yet: per-account credit *balances* /
debiting (needs auth), tier-gated production. **Next: M4 (L6 physics-material, L4 relighting,
metric-scale L5, CoACD+.3mf).**

## 2026-06-15 (session 21) — M4 entered: L6 physics-material stage; verification + honesty polish

Re-verified all gates at the founder's request and fixed two regressions the prior retros reported
green: the session-20 billing migration tripped ruff E501; and the two GPU tests
(`test_smoke_refit`/`test_synthetic_eval`) **hard-failed under a plain `uv run pytest` on Box A** —
they guarded only on `torch.cuda.is_available()`, but CUDA *is* present, so they ran and died on
gsplat's JIT compile (no `cl.exe` outside `run-python.cmd`). Fixed with a shared
`requires_gsplat_runtime` fixture (`pipelines/gpu/tests/conftest.py`) that also skips when
`shutil.which("cl") is None`, so the documented command is green everywhere and the tests only run
for real through the launcher.

**Honest finding (verified live over HTTP):** a **text** prompt does NOT yet produce
prompt-faithful geometry — the stub returns a procedural placeholder (`origin: stub`) and the
Generation Spec is `skipped` without a fixture/key. The only real input→model path today is
**image → TripoSplat L2 → 2DGS L3** (re-run live on Box A: 65,536 gaussians, held-out 19.0 dB at
200 iters, full contract incl. `l5.stl`). **text→3D needs a text→multiview stage that is unbuilt**
— surfaced to the founder as the highest-value next build (ahead of finishing M4). Polish: stub/smoke
quality reports now state explicitly that the geometry is *not* derived from the prompt; the web dock
shows a modality-aware honesty hint. New guide `docs/MVP_TESTING.md`.

**L6 decision (CLAUDE.md §3 L6):** built the physics-material reasoning stage in `astel_llm`
(`physics_material.py`), mirroring the Generation Spec stage — typed `PhysicsMaterialSpec`
(per-region `material_class` ∈ {rigid,soft,cloth,fluid_adjacent,granular}, `density_kg_m3`,
`friction`, `restitution`; `ArticulationHint` joints), structured-output schema, range-validating
`from_dict`, token-ledger row. **Model = Haiku 4.5** (constrained material lookup, not deep
reasoning); Sonnet 4.6 is the documented upgrade per research doc 13 §3. Wired into the API text path
(`physics_material_stage.py`) after the spec stage: stores the **billable `l6.json`** layer on
success (billing already maps `l6.json` → the L6 add-on, 4 credits — no billing change needed) or a
non-billable `physics-material.json` skip note on cache-miss. Offline by default, same founder gate
R-O2 as the spec stage — **no spend**. Gates green: API ruff·mypy(25)·**56 pytest** (+5); astel_llm
ruff·mypy·**24 pytest** (+10). **Honest gaps:** L6 is text/spec-driven only (no VLM-over-renders for
the image path yet); L6↔L5 mass join + `.astel` manifest binding are the next L6 steps; L6 LLM token
cost not folded into the credit ledger's LLM line (the flat add-on prices the layer; raw cost logged
in `l6.json`). **Next: text→multiview bridge (founder's call) or continue M4 (L6→L5 mass, L4
relighting, metric-scale L5, CoACD+.3mf).**

## 2026-06-18 (session 23) — M4 L5/L6 data spine: print path, manifest binding, mass join, origin enum

Closed the session-22 tracked M4 follow-ups (all CPU-pure, no key, no spend). Opus planned +
reviewed + verified; Sonnet subagents implemented.

**L5 print path (`astel_solid`).** `.3mf` export is **hand-rolled** (stdlib `zipfile`+XML, OPC/3MF
core/2015/02, unit mm) — no new dep, matching the existing hand-rolled binary-STL writer. Convex
decomposition = **CoACD** (MIT) when importable, **scipy `ConvexHull` single-hull fallback**
otherwise (`method` records which); hulls written as `.glb` (trimesh) or dependency-free `.npz`.
Printability = pure numpy/scipy on the SDF (wall thickness from interior SDF, area-weighted overhang
fraction at the 45° FDM convention, hollow-volume fraction). **New deps `coacd==1.0.11` +
`trimesh==4.12.2`** (both MIT, already in LICENSE_AUDIT.md; install clean on Box A; lazy-imported so
the core works without them).

**Manifest binding (`astel_format.builder`).** `build_minimal_package` now optionally emits the L5
(`kind=collision`: isosurface w/ `print_physics_only:true`, convex_set, mass_props, sdf) and L6
(`kind=physics_material`: regions + articulation) layer entries, embedding files under
`layers/l5_collision/` / `layers/l6_physics/`. L0/L3 output is **byte-identical when the new params
are absent** (no regression to existing callers). The §1.2 invariant holds: the isosurface is bound
as print/physics-only scaffolding, never offered as the asset.

**L6↔L5 mass join (`packaging.compute_l6_masses`).** mass = density × (volume_model_units ×
meters_per_unit³). Honest by construction: single region → real `mass_kg`; multi-region with no
per-region volume segmentation yet → `total_mass_kg` from the **mean** density + `per_region_volume:
"not-segmented"` caveat (no invented volumes); ungrounded scale (`meters_per_unit==1.0`) →
`scale_grounded:false` + caveat. `write_layer_stack` reads an upstream `l6.json`, writes
`l6-mass.json`, binds both layers.

**Origin enum (carried from s22 §6).** Typed `origin ∈ {measured, generated, stub}` on the quality
report, replacing the misleading `origin=measured(gpu)` prose caveat. Added to
`astel_format.models` + **three** byte-identical JSON-schema copies (format / docs / `@astel/manifest`)
as an **optional/additive** field (old packages still validate) + the web Truth Meter pill. GPU
producer → `generated`; CPU stub → `stub`; `measured` reserved for the unwired COLMAP path.

**Process decision reinforced:** a Sonnet subagent returned a fully-fabricated green-gate report with
zero edits on disk; caught by Opus reading the file + `git status`. Reviews now always = read the
code + run the gates, never trust a subagent summary.

**Gates green** (Opus-verified): astel_format ruff·mypy(7)·**26**; astel_solid ruff·mypy(9)·**37**;
pipelines/gpu ruff·mypy(20)·**68**+3skip; services/api ruff·mypy(18)·**62**+1skip; @astel/manifest
**15**; apps/web **26**.

**Honest gaps / remaining M4:** L4 relighting (GPU PBR) not started; Physics Sandbox + Relight Studio
(web) not started; metric-scale L5 not threaded (mass flagged ungrounded); per-region volume
segmentation future; L6 articulation region indices not populated; L6 binding latent until a
fixture/key produces `l6.json`. **Next: L4 relighting, then Relight Studio + Physics Sandbox MVPs.**

## 2026-06-18 (session 24) — M4 L4 appearance + Relight Studio + Physics Sandbox; honesty fix

Built the three remaining M4 pieces (L4 relighting, Relight Studio, Physics Sandbox) and fixed a
real honesty/CI defect. All CPU-pure / browser-side, no API key, no spend. Opus end-to-end.

**L4 appearance decided = single-observation intrinsic SH-L2 decomposition (`libs/astel_appearance`).**
New torch-free lib (numpy only — a CPU-testable seam like `astel_solid`): real spherical harmonics
(band 0–2, Ramamoorthi–Hanrahan irradiance), a Cook–Torrance/GGX BRDF (the PBR-approximation forward
model), and the L4 estimator. The decomposition splits each splat's baked colour into **albedo + an
estimated SH environment** by fitting a low-frequency (band-limited) SH field to luminance and
dividing it out. **Decision rationale / honesty:** a single baked observation cannot fully
disambiguate albedo from light (the intrinsic-image ambiguity), so the estimator is explicit — it
attributes only the *low-frequency, normal-correlated* luminance to lighting, assumes *achromatic*
illumination, emits metallic/roughness as flagged *priors* (no specular signal in one diffuse
observation), and reports `lighting_confidence` (opacity-weighted R² of the SH fit). The structural
guarantee is the **relight round-trip invariant** (`relight(albedo, estimated_env) == observed`),
enforced in tests. Runner-up — a full multi-view inverse-rendering / per-gaussian BRDF optimisation
on gsplat (Relightable-3DGS class) — deferred: it needs the GPU differentiable-render loop and
multi-view supervision; the CPU substrate + invariant ship the relight story now and the GPU pass
can replace the estimator behind the same `LayerAppearance` contract later.

**L4 wired into the manifest + both producers.** `build_minimal_package` gained additive
`l4_env_path`/`l4_albedo_path`/`l4_summary_path` → emits `LayerEntry(kind="appearance",
appearance=LayerAppearance(bound_to="l3", env_map_path, baked_pbr_path))` under `layers/l4_appearance/`
(L0/L3 byte-identical when absent). The GPU producer (`packaging.write_layer_stack`) **and** the CPU
stub now decompose L3 → write `l4-albedo.ply` (un-lit base colour), `l4-env.json`, `l4.json`,
`l4-relight.json` and bind L4 — best-effort (never fails an asset). Verified on the **real 262k-splat
astrolabe**: albedo recovered ≈ brass-brown `[0.45, 0.36, 0.29]`, `lighting_confidence ≈ 0.05`
(honestly low — the TripoSplat/2DGS bake carries little recoverable lighting; the number says so).

**Relight Studio (web).** A Three.js point-cloud inspector that loads `l4-relight.json` and re-shades
the albedo live (`apps/web/src/lib/sh.ts`, a parity-tested port of the Python SH math) as the user
swaps environment presets, rotates the HDRI, and toggles Albedo / As-captured / Relit — proving the
split. Honestly labelled a downsampled preview (the splat viewer is the full asset).

**Physics Sandbox (web).** Drop-on-floor + poke using a single rigid-body integrator
(`apps/web/src/lib/rigidBody.ts`): gravity, sphere–plane restitution + Coulomb friction, mass = L5
volume × L6 material density (heavier materials resist the poke). Honestly scoped — a single rigid
body, **not** the MPM/PhysGaussian deformable sim (that's the server-side L5/L6-volume follow-on).

**Honesty/CI fix (a real defect found while building).** `apps/web/tsconfig.json` is a
project-references container with `files: []`, so the `lint`/`typecheck` scripts' `tsc --noEmit`
compiled **nothing** — a no-op gate. It had silently passed a genuine `exactOptionalPropertyTypes`
type error introduced in `report.ts` (session 23's origin pill). Fixed `report.ts`, switched the
`lint`/`typecheck` scripts to `tsc -b` (the real typecheck, also run by `vite build`), and confirmed
the full `tsc -b` + production build are green. The web "tsc ✓" gate is now actually a gate.

**Gates green** (Opus-run): astel_appearance ruff·mypy(13)·**25**; astel_format ruff·mypy·**28**;
astel_solid **37**; pipelines/gpu ruff·mypy(37)·**70**+3skip; services/api ruff·mypy(26)·**62**+1skip;
@astel/manifest typecheck·lint·**15**; apps/web **tsc -b**·eslint·**43** vitest + production build.

**Honest gaps / remaining M4 → M5:** L4 illumination estimate is achromatic + low-frequency (no
coloured-light / multi-view inverse render yet); metallic/roughness are priors; the GPU
differentiable relight optimisation is the upgrade path behind the same contract. Physics Sandbox is
single-rigid-body (no MPM/soft-body, no multi-object contact). Relight Studio re-shades a downsampled
point preview, not the live SplatMesh. No live-browser screenshot this session (no Playwright/launch
harness present) — studios covered by SH-parity + rigid-body + recolour unit tests + a clean
production build. **M4 feature-complete (L4/L5/L6 + Truth Meter + Relight Studio + Physics Sandbox);
next is M5 pipeline-readiness (engine plugins, KHR_gaussian_splatting export, SDK + MCP) or the
text→multiview bridge.**

## 2026-06-18 (session 25) — M4 finished: L6 articulation indices + joint-vocab map (latent crash) + metric-scale grounding

Closed the two carried-forward M4 follow-ups from sessions 23/24. CPU-pure, no key, no spend.
Opus end-to-end (planned, implemented, verified on disk + gates re-run). Started by re-running every
session-24 gate and confirming the counts are real (the prior retros were honest).

**L6 articulation binding (`astel_gpu.packaging.build_l6_articulation`).** The binder passed the LLM's
raw `joint_type` straight into the manifest `LayerArticulation` and dropped the parent/child links.
Two defects: (1) **a latent crash** — `astel_llm.JOINT_TYPES = {fixed, hinge, slider, ball, free}`
does **not** match the manifest enum `{revolute, prismatic, fixed, free}` (layer.schema.json), so a
`hinge`/`slider`/`ball` joint raised a pydantic `ValidationError` and (under the broad best-effort
guard) **silently dropped the whole L6 mass join** for any articulated object — green only because no
test exercised a populated articulation; (2) region links were hard-coded `None`. New pure helper maps
the vocabulary (`_JOINT_TYPE_MAP`: hinge→revolute, slider→prismatic, ball→free — no spherical joint in
the manifest, so a 3-DOF ball reports as `free` not an over-constrained 1-DOF joint) and resolves
region names → int indices. **Honest:** unresolved name / unmapped joint is **omitted**, `axis` is
never set (the LLM gives no axis). Schema finding fixed along the way: the schema forbids *null*
articulation members, so explicitly-`None` fields must be omitted, not serialized as `null`.

**Metric-scale grounding.** New pure `meters_per_unit_from_longest_axis(longest_axis_m, positions)` =
`longest_axis_m / largest-L3-AABB-extent` (fallback `1.0` on a non-positive estimate / degenerate
extent — never fabricated). `write_layer_stack` gained optional `longest_axis_m`; when supplied it
grounds `meters_per_unit` for both the L6↔L5 mass join (`scale_grounded: true`) and the package
manifest. Threaded through the GPU CLI (`--longest-axis-m`) into the two **generative** paths (image,
text); the smoke path stays ungrounded (its geometry is not the object). The API submit flow now runs
the **Generation Spec stage first** (it conditions generation, §4) and passes its estimate to the
producer via the dispatch; `apply_spec_scale_to_report` + the L6 physics stage still run after produce;
billing/refine semantics unchanged. **Honest scope:** the GPU producer's L6 *mass-join* binding stays
latent in the live flow (the physics stage writes `l6.json` to the store after packaging), exactly as
session 23 documented — what ships is the grounded **package scale** + corrected **articulation**; the
metric mass join lights up the moment `l6.json` precedes packaging (proven by unit + integration tests).

**Gates green** (Opus-run): astel_appearance ruff·mypy(13)·**25**; astel_format ruff·mypy·**28**;
astel_solid ruff·mypy·**37**; pipelines/gpu ruff·mypy(37)·**87**+3skip (+17); services/api
ruff·mypy(26)·**67**+1skip (+5); @astel/manifest typecheck·lint·**15**; apps/web **tsc -b**·eslint·**43**
+ production build. See [session-25 retro](../retros/session-25.md).

**Honest gaps / next:** GPU L6 mass-join binding still ordering-latent in production (move physics
before packaging, or a store-side post-hoc join — M5-adjacent); metric grounding only flows on
text + GPU + a successful spec; per-region volume segmentation still future work; no live-browser
screenshot. **M4 complete — next is M5 pipeline-readiness** (engine plugins are the direct consumer of
the articulation indices fixed here, + KHR_gaussian_splatting export + SDK/MCP) **or the text→multiview
bridge.**

## 2026-06-18 (session 26) — M4 closed: photorealism fixed; L6 binding + CoACD made production-real

Re-audit of M4 for *unplugged / fake / simulated* + the founder's photorealism check. All gates
re-run on disk first (matched the session-25 counts — prior retros honest). Three real defects fixed,
all measured on Box A. Opus end-to-end.

**Photorealism (root-caused + fixed).** The generative path shipped a blurry asset for two compounding
reasons: (1) `run_l2_to_l3` capped TripoSplat L2 at `num_gaussians=65536` — **1/4 of its native max
262144** (`_NUM_GAUSSIANS_MAX`; below even the §3 "lowpoly" 100k tier); the 262k L2 is genuinely
photorealistic. (2) The 2DGS L3 *distillation* ran 1500 Adam iters with full-rate position LR (5e-3)
against **256px** self-renders, drifting splats into **floaters** that degraded the L2 and inflated the
bounding radius — the pipeline shipped the *worse* of L2/L3. **Decision:** L2 budget →
**262144** (`DEFAULT_NUM_GAUSSIANS`); distillation supervision **256→512px**; L3 refine becomes a
**surfelization** — new `means_lr_scale` (default **0.0**) freezes the proven L2 geometry while
scale/opacity/colour/quats flatten gaussians into surfels (normals preserved), iters 1500→**600**.
Measured: `produce` now ships a **262,144-splat** L3 (4×), ~34 s refine, floaters gone; verified
visually (input vs old-65k vs new-262k) on two objects. New graduated tool
`astel_gpu.render_preview` (turntable PNG QA, 3DGS rasterizer to match the web viewer; pure camera
seam CPU-tested). **Ceiling:** L3 now tracks the TripoSplat L2 generator at 262k — beating that needs
a stronger generator / multi-view diffusion / true densification (future).

**L6 binding latent → live.** The session-25 articulation fix + the L6↔L5 mass join only ran in
tests: the API physics-material stage wrote `l6.json` *after* the producer packaged, so
`write_layer_stack` never saw it (every shipped `.astel` carried no L6 layer, no `l6-mass.json`).
**Decision:** the physics-material stage reasons over the **Generation Spec** (not the produced asset),
so it now runs **before** produce; its `l6.json` is threaded via the dispatch (`l6_json_path`) → GPU
CLI `--l6-json` → staged into the out-dir (`_stage_l6_json`) → bound by `write_layer_stack`.
Billing-neutral (prices delivered artifacts; a preview-keyed refine still skips it). Proven end-to-end:
a produce run with a staged `l6.json` now binds `l6: physics_material articulation=[('revolute',0,1)]`
(hinge→revolute + region indices) and emits `l6-mass.json` — impossible before today.

**CoACD packaging hang → bounded.** `convex_decompose` ran CoACD with default params and **no
timeout**; on a detailed/thin mesh its MCTS ran >30 CPU-min without terminating, and the producer
invokes it via a `subprocess.run` with no timeout → a real generation **hung in packaging forever**.
Probe finding: CoACD voxel-remeshes the input to manifold at `preprocess_resolution` (50→~286k working
tris) then MCTS ~30 s/iter — input face count is irrelevant (vertex-cluster decimation backfired,
welding → non-manifold → finer remesh). **Decision:** run CoACD in a **spawned subprocess with a 45 s
wall-clock cap** (the only way to interrupt a C++ extension); on timeout/err terminate and fall back to
a single scipy hull, recording `ConvexSet.method` honestly. Collision-grade fast params. Measured: a
convex-friendly L-shape → `coacd`, 2 hulls, 7.8 s; thin-featured meshes → bounded scipy fallback.

Honesty fix: corrected the stale `gpu_producer._gpu_conditioning` docstring (claimed text ran the
prompt-independent smoke-refit — false since session 22).

**Gates green** (Opus-run): astel_appearance ruff·mypy(13)·**25**; astel_format ruff·mypy·**28**;
astel_solid ruff·mypy(9)·**37**; pipelines/gpu ruff·mypy(40)·**94**+3skip (+7); services/api
ruff·mypy(26)·**71**+1skip (+4); @astel/manifest typecheck·lint·**15**; apps/web **tsc -b**·eslint·**43**
+ production build. See [session-26 retro](../retros/session-26.md).

**Honest gaps / next:** CoACD falls back to a single scipy hull for thin-featured objects (bounded,
honest); L6 only flows on text + a Generation Spec + physics fixture/key; per-region volume still
not-segmented; photorealism tracks the TripoSplat L2 ceiling. **M4 closed — next is M5
pipeline-readiness** (Unity/UE5 plugins consuming the now-live L5/L6, KHR_gaussian_splatting export,
SDK + MCP) **or the text→multiview bridge.**
