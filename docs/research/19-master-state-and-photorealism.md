# 19 — Master State & Photorealism Roadmap (single source of truth)

> **Written 2026-06-21, consolidating sessions 1–34.** Purpose (founder directive):
> one document so the next agent **never re-runs Phase R**, knows exactly what was
> decided, what is actually built, what is *claimed* but not built, and what to do
> next to make **text → photorealistic Gaussian splats** real. Authored by reading
> every doc in `docs/research/` + `docs/retros/01..29` + the architecture/spec docs
> + all 19 memory files, and by **verifying the live code** (not trusting retros).
> Where a retro/memory claim disagreed with the source tree, the **code wins** and
> the discrepancy is flagged. Status legend: ✅ real & verified · 🟡 partial / opt-in
> / honest-stub · ❌ not implemented · ⚠️ correction of a stale belief.

---

## 0. TL;DR for the next agent (read this first)

1. **The build plan (CLAUDE.md §9, M0–M6) is 100% implemented and gated.** There is
   no M7. Everything since is **post-M6 quality + launch work** (sessions 30–34),
   which is where the founder's pain is.
2. **The product's text→3D path works end-to-end and is honest, but it is NOT
   photorealistic, and that is a known, root-caused, *architectural* ceiling — not a
   bug.** Default path = `prompt → SDXL (1 image) → TripoSplat (single-image
   feed-forward, 262k splats, SH degree 0) → 2DGS "L3 refine" that is actually a
   frozen-position distillation`. Quality is capped by **(a)** single-image
   conditioning, **(b)** SH degree 0 (no view-dependent specular → "painted clay"),
   **(c)** a refine stage that *cannot add information* (it re-fits the L2's own
   renders with frozen positions). See §6.
3. **TRELLIS clarification (the founder's specific concern):** there are **two
   different models** and they were conflated in a recent session. **TRELLIS.2 =
   mesh output (off-thesis), known mesh-only since session 2 — correctly rejected.**
   **TRELLIS v1 has a native *Gaussian* head that is MIT and nvdiffrast-clean
   (audit, session 12) and was the *original* L2 choice.** Concluding "TRELLIS is
   unusable because it's a mesh model" is **half-right**: it's true of v2, false of
   v1. See §7. Neither is currently wired in (we shipped TripoSplat instead).
4. **Things the founder believed are "not implemented" but actually ARE** (⚠️
   corrections): **glTF `KHR_gaussian_splatting` export is implemented** (session 27,
   `astel_splat_io/gltf.py`) — but at **SH degree 0** and RC-schema, no engine
   round-trip test. Unity + UE5 plugins, Python+TS SDKs, and an MCP server are all
   implemented (session 27) but **never compiler-verified against a real engine**.
   See §8.
5. **The single biggest lever toward photorealism = a real multi-view-consistent
   front end + a real (un-frozen, densified) refine + SH degrees ≥ 2.** The
   machinery for the refine (`densify.py`, `refine.py`) and a no-new-weights
   multi-view enhancer (`mv_enhance.py`) **already exist but are OFF by default and
   unproven**; SH > 0 is **not started** and has a large blast radius. See §6 and §10.

---

## 1. Mission & the layered asset model (binding, unchanged)

Astel (née AURIGA) builds **Gaussian-splat assets only** — never meshes as a
product output — from **text / image(s) / video**, that are geometry-accurate,
world-aware (collision/physics/lighting/semantic layers), 3D-printable (via an
*internal* splat→SDF→watertight path), and drop-in for UE5/Unity/Blender/web/USD.

The invention is the **Layer Stack** (CLAUDE.md §3), persisted as one `.astel`
package (zip + manifest):

| Layer | Meaning | Built? (see §3/§4) |
|---|---|---|
| **L0** Seed/sparse cloud | SfM points or generative samples + per-point confidence | ✅ (subsample of L2/stub) |
| **L1** Dense cloud | metric-scaled, normals, semantic logits | ❌ **never written by any producer** |
| **L2** Coarse gaussians | fast feed-forward (TripoSplat) | ✅ image/text; ❌ capture/video |
| **L3** Refined surface gaussians (hero) | 2DGS surfels + normals; budgeted | ✅ but = **distillation, not refinement** |
| **L4** Appearance/lighting | albedo/roughness/metallic + env decomposition | 🟡 CPU achromatic SH-L2 estimate, not a GPU inverse-render |
| **L5** Collision/solidity | SDF → convex proxies + watertight print surface + mass | ✅ real, bounded (CoACD timeout) |
| **L6** Physics-material/semantic | per-region material/density/friction + articulation | 🟡 LLM stage, offline fixtures (no key), text-only |
| **L7** Dynamics | 4DGS deformation field for video | 🟡 LBS field + binding real; **no real video tracking** |

The **provenance channel** (measured↔generated confidence per gaussian) and the
**Truth Meter** (per-asset honesty report) are real and wired from L0 to export.

---

## 2. Phase R research — the decisions, verified and updated

This section is the "don't research again" payload. The full rationale + paper math
lives in `docs/research/01..10` and `DECISIONS.md`; this is the consolidated,
**current-status** view. Every external fact below was last live-verified in the
session noted (training data is stale by definition — re-verify only if adopting).

### 2.1 The generative-model landscape (THE clarification)

This is the most-misunderstood area and the founder's specific concern. There are
three distinct open, MIT-licensed models; they are **not interchangeable**:

| Model | Output | Conditioning | License | Windows feasibility | Status in Astel |
|---|---|---|---|---|---|
| **TripoSplat** (VAST-AI) | **native 3D gaussians** (≤262,144, learned density) | **single image** | MIT code **+ weights** | **Trivial** — no CUDA build, no flash-attn/xformers/spconv; only `torch`/`torchvision` + 4 pure-py pkgs (audit, session 14) | ✅ **ADOPTED, wired, shipping** as the L2 generator |
| **TRELLIS v1** (Microsoft) | **native gaussian head** (`TrellisImageTo3DPipeline.run()['gaussian']`) + RF + mesh heads | single image (or text via text models) | MIT code+weights | **Surmountable** — gaussian-head path is nvdiffrast-clean (audit, session 12); but uses **sparse structured latents** → needs spconv/flash-attn/xformers, whose prebuilt wheels target cu124/torch2.5/py3.10 while Box A is cu128/torch2.11/py3.12 → must build via vcvars or use `ATTN_BACKEND=xformers` / `SPCONV_ALGO=native` (R-T9) | ❌ **never installed/wired**; was the *original* L2 plan, superseded by TripoSplat |
| **TRELLIS.2** (Microsoft) | **mesh + PBR only** (O-Voxel; the v1 gaussian/RF heads were **dropped**) | single image | MIT code+weights | **Hard / off-thesis** — the *texturing* path needs `nvdiffrast`/`nvdiffrec` (NVIDIA **non-commercial**), needs ≥24 GB; the geometry decode is clean-after-a-1-line-fork-patch but **produces a mesh** | ❌ correctly **rejected** for a splats-only product |

**Key correction (⚠️):** "TRELLIS is a mesh model, unusable" is true **only of
TRELLIS.2**. It has been known since **session 2** (RA2: *"Critical caveat:
mesh-only output"*) and re-confirmed in the session-12 import audit. **TRELLIS v1's
gaussian head is a legitimate, license-clean splat generator** and remains the
strongest open alternative to TripoSplat for a *better* L2 (structured latents tend
to give better global geometry than a single feed-forward pass). A recent session
re-derived the v2-is-mesh fact and stopped at Tier 1, without revisiting that **v1
is the usable one** — that is the work the founder halted. The original research
already settled this; see §7 for the definitive write-up.

**Why TripoSplat was chosen over TRELLIS v1 (sessions 11–14, measured):** cleanest
dependency profile of any candidate (no CUDA build at all), native gaussian output,
4.6–4.9 GB / ~11 s on Box A, published Elo 1137 > TRELLIS.2 992, MIT code+weights.
The L2-prior decision (DECISIONS #2) and the riskiest bet (R-T1, the TRELLIS.2
mesh→surfel distillation) were both **resolved/retired** by adopting TripoSplat. A
head-to-head PSNR/SSIM/LPIPS vs the TRELLIS-v1 gaussian head was **deferred and never
run** (needs a multi-view generative corpus + a v1 install) — it is the open A/B if
we want to push L2 quality.

### 2.2 Per-stage decisions (consolidated, with current status)

| Stage | Chosen | Status now | Note / what's unbuilt |
|---|---|---|---|
| Rasterization backbone | **gsplat** (Apache, nerfstudio) | ✅ in use (1.5.3, cu128) | built-in 2DGS/AA/AbsGS/MCMC; runtime needs `cl.exe` on PATH (torch-2.11 JIT quirk) |
| Ray-traced/relight render | 3dgrut (Apache) | ❌ not installed | only needed for L4 validation / Relight Studio GT |
| **L3 representation** | **2DGS surfels** + normal-consistency + scale-tuned depth-distortion | ✅ **resolved on real DTU** (session 13): 2DGS 8.53 mm vs 3DGS 8.76 mm overall, real normals | λdist=1e-4 is **DTU-scan1-specific** (scale-dependent); a dimensionless λdist is unbuilt |
| L3 refinement losses | MCMC budget densification + AA + reimplemented **PGSR** (edge-aware normal, FB-reprojection, patch-NCC, exposure) + DN-Splatter priors | ❌ **PGSR multi-view losses never implemented**; `densify.py`/`refine.py` exist but are opt-in and only do L1+D-SSIM+surface-reg+perceptual | the biggest geometry-accuracy lift was scoped (RA8 §2) and never built |
| **Generative L2** | **TripoSplat** (single image) | ✅ wired, shipping | runner-up = TRELLIS v1 GS head (§7) |
| Generative L3 | 2DGS **distillation** of L2 self-renders, frozen positions | ✅ but **adds no quality** (§6) | this is the core ceiling |
| Text conditioning | LLM Generation Spec → **T2I (SDXL default / FLUX opt-in)** → bg-removal → image path | 🟡 SDXL→TripoSplat real; Generation Spec runs **offline fixtures only** (no API key) | multi-view T2I (MV-Adapter) **never built** |
| Multi-view guidance | **MV-Adapter** (or current SOTA — re-verify) | ❌ not built | the principled fix for single-image hallucination (G2) |
| Capture front-end L0/L1 | **MapAnything** `-apache` ckpt + GLOMAP/COLMAP BA | 🟡 **COLMAP runner real & DTU-validated** (49/49 imgs, 0.886 mm pose RMSE); **MapAnything never installed**; **neither wired into the producer** | capture path is offline-eval-only |
| Metric scale | consensus MapAnything + MoGe-2 + SfM/EXIF, reported CI | ❌ not built; generated assets use the **LLM size estimate** (when a fixture/key exists) else ungrounded | |
| L4 appearance | per-gaussian PBR via deferred shading on gsplat + split-sum IBL | 🟡 **CPU achromatic low-freq SH-L2 estimate** (`astel_appearance`), not the GPU inverse-render | relight round-trip invariant holds; metallic/roughness are flagged priors |
| L5 solidity/print | L3→SDF→marching cubes→.stl/.3mf; **CoACD** convex decomp; mass props | ✅ real, **CoACD bounded by a 45 s subprocess timeout** (hung forever before, session 26) | print surface is internal scaffolding only (§1.2) |
| L6 physics-material | LLM/VLM reasoning + SAM part-seg | 🟡 LLM `PhysicsMaterialSpec` real but **offline fixtures, text-only**; SAM/VLM-over-renders never built | articulation→manifest-enum mapping fixed (session 25/26) |
| L6 physics sim | MPM-on-gaussians over **NVIDIA Warp** (PhysGaussian math, fill from our L5 SDF) | ❌ not built; web Physics Sandbox is a **single rigid body** | math settled (RA8 §5), Warp module unwritten |
| L7 dynamics | affine-LBS deformation field; GPU 4DGS video fit | 🟡 LBS field + `.astel` binding real & analytic-GT-validated; **no real per-frame video tracking** | |
| Web viewer | **Spark** (Three.js, MIT) | ✅ in use | supports SH0–SH3; we only feed it SH0 |
| Export formats | .ply, .spz, .sog, **glTF+KHR_gaussian_splatting**, USD/USDZ, .astel; print .3mf/.stl | ✅ .ply/.spz/.sog/.astel/.stl/.3mf; ✅ **glTF KHR (SH0, RC schema)**; ❌ **USD/USDZ never built** | see §8 |
| Task engine | **Temporal** (MIT) | 🟡 `TemporalTaskEngine` exists; **default is the in-process async JobManager** (real SSE, session 31) | Temporal path opt-in, 1 skipped test |
| Engine plugins | Unity package + UE5 plugin (auto physics from L5/L6) | 🟡 **code-complete, never compiler-verified** (no licensed runners) | §8 |
| SDK + MCP | Python + TS SDK, MCP server | ✅ code-complete + unit-tested | §8 |
| LLM layer | Anthropic adapter, Haiku=spec, Sonnet=upgrade | 🟡 **double-gated offline by default** (FixtureAdapter); **no API key ever used, zero spend** | ~$0.02–0.035/gen estimated |

### 2.3 Metrics & CI-gated targets (RA9 — defined, partially realized)

Per-layer targets exist (`09-metrics-targets.md`): e.g. L3 product target **Chamfer
≤ 1% of bbox diagonal in measured regions**, held-out PSNR ≥ 26, normal err ≤ 15°;
L5 **100% watertight (hard zero-tolerance)**; L1 **scale-CI coverage ≥ 0.85**
(under-coverage = the Truth Meter lying = hard fail). **Reality:** the synthetic +
DTU eval harness is real and produces real Chamfer/PSNR, but **no `eval_targets.yaml`
CI floor is enforced** (there is no running CI at all — see §8). The numbers we have
are point measurements in retros, not regression-gated invariants.

### 2.4 Licensing posture (clean)

Ship only Apache/MIT/BSD code **and** weights, verified separately. NC work (Inria
3DGS lineage, DUSt3R/MASt3R, nvdiffrast/nvdiffrec, Tanks&Temples/CO3D datasets) may
inform design but is **never vendored**; published math is reimplemented on gsplat.
All adopted models (TripoSplat, SDXL, FLUX.1-schnell, SAM2, Warp, SPZ, MapAnything
`-apache`, VGGT-1B-**Commercial**) are permissive. DTU is internal-benchmark-only
(no redistribution, no training, no product claims). See `LICENSE_AUDIT.md`.

---

## 3. What was actually built — milestone history (M0–M6)

Each milestone shipped a **torch-free CPU-validated core + honest GPU-deferred
notes**, gates green. Condensed from the retros + DECISIONS log:

- **M0 (Phase R, sessions 1–2):** `DECISIONS.md`, RA1–RA10 research notes, RISKS,
  LICENSE_AUDIT, positioning. Temporal chosen by a hands-on Windows spike.
- **M1 Skeleton (sessions 3–5):** monorepo, FastAPI + Postgres + local artifact
  store, Temporal engine, `.astel` format (`astel_format`), splat IO
  (`astel_splat_io`: .ply/.spz/.sog), web viewer (Spark), provenance channel, stub
  producer (procedural torus). 
- **M2 Capture (sessions 7–10):** native-Windows GPU stack (CUDA 12.9 + VS2026,
  arch 8.9), gsplat trains on Box A; **COLMAP SfM front-end validated on real DTU**
  (49/49 registered, pose RMSE 0.886 mm); **DTU-protocol Chamfer eval** (raw 3DGS
  baseline 8.73 mm overall). First real geometry numbers.
- **M3 Generative (sessions 11–20):** TripoSplat triaged → installed → graduated to
  `l2_triposplat.py` (opacity defect fixed); **L3 2DGS-vs-3DGS A/B resolved on DTU**
  (2DGS wins); `generative.py` wires image→L2→L3; Generation Spec LLM stage
  (`astel_llm`, offline); GPU producer artifact parity through the API subprocess
  seam; preview/refine **billing** semantics.
- **M4 World-awareness (sessions 18–26):** L5 solidify (`astel_solid`: SDF→watertight
  →.stl/.3mf, mass props, CoACD bounded); L6 physics-material LLM stage + manifest
  binding + L5↔L6 mass join + articulation enum mapping; L4 appearance
  (`astel_appearance`, CPU SH-L2 intrinsic decomposition + relight round-trip
  invariant); web **Relight Studio** + **Physics Sandbox** (single rigid body);
  **photorealism fix**: L2 cap raised 65k→**262k**, L3 made a frozen-position
  surfelization @512px. Truth Meter `origin` enum.
- **M5 Pipeline-readiness (session 27):** **glTF KHR_gaussian_splatting export**
  (`astel_splat_io/gltf.py`, SH0), coordinate-convention module + doc
  (gltf/Unity/UE5), **Unity package**, **UE5 plugin**, **Python SDK + MCP server**,
  **TS SDK**, MkDocs docs site (10 pages).
- **M6 Dynamics & scenes (session 29 — final milestone):** `astel_dynamics`
  (affine-LBS L7 field, analytic-GT-validated, `.astel`-bound), `astel_scene`
  (multi-object layout + ground-contact composition + offline layout-LLM),
  `astel_lod` (importance LOD with nested-subset guarantee, producer + web consumer);
  `.astel` reader hardening (size-validated L7 `.bin`); load-test harness;
  LAUNCH_CHECKLIST. **Finding: there was NO CI in the repo.**

## 3b. Post-M6 work (sessions 30–34) — NOT in the retros, only in memory

This is the recent, **uncommitted** work where the founder's quality concern lives.
The git tree shows it as modified `generative.py`/`produce.py`/`text_to_image.py`
plus untracked `densify.py`, `refine.py`, `mv_enhance.py`, `image_qa.py`,
`geometry_qa.py` and their CPU tests.

- **Async generation (session 31):** `POST /v1/generations` returns immediately;
  a background `JobManager` (`services/api/.../jobs.py`) runs the pipeline and
  streams **real** SSE progress (the old fake-replay SSE is gone). `pnpm run up`
  auto-detects the GPU and sets `ASTEL_PRODUCER=gpu`. Gallery is live-wired.
- **Truth Meter black-screen + gallery-blank fixes (session 30):** generated reports
  carry null geometric_error/scale (honest) — the web must render "not measured",
  not `.toFixed()`; an `ErrorBoundary` now wraps the router. Gallery viewport CSS
  height bug fixed.
- **Splat cleanup (session 32, `splat_clean.py`):** removes TripoSplat floaters via
  **connected-components** (NOT statistical outlier removal — SOR ate ~40% of a real
  helmet); generative-path only (never capture — §1.3/§10.4); ON by default.
- **Tier 0 QA gates (session 32–33):** `image_qa.py` (critic scoring a T2I image for
  TripoSplat suitability), `text_to_image.generate_image_best_of_n` (draw N=4, keep
  the critic's best), `geometry_qa.py` (degenerate-cloud critic), surfaced in the
  Truth Meter. **Verified on real SDXL/GPU.** This fixes **reliability**
  (same-prompt-sometimes-wrong), not photorealism.
- **Tier 1 foundation (session 33, `densify.py` + `refine.py`):** real Adaptive
  Density Control (clone/split/prune/opacity-reset) + an **un-frozen, densified**
  refine loop with perceptual loss; opt-in via `ASTEL_L3_REFINE` (default OFF).
  **Measured on Box A:** synthetic random-init **8.87→31.77 dB** (engine works); but
  on the real 262k TripoSplat cloud supervised by its **own self-renders**,
  distillation **23.05 dB beat** densified **20.14 dB** — because moving positions
  drifts from the config that made the self-render targets. **Load-bearing
  conclusion: the densified engine is correct but *starved of new information*
  without external multi-view targets.**
- **MV-enhance (session 34, `mv_enhance.py`, IN PROGRESS):** the chosen "external
  targets" source = **SDXL img2img** over the L2 orbit renders (no new weights). GPU
  finding: SDXL **does inject real detail** (legible clock dial vs mushy base) but
  raw `combine="replace"` made the final asset **dark/collapsed** (per-view exposure
  disagreement → refine averages to mush — the documented multi-view-inconsistency
  risk, confirmed). Fix in progress: `detail_transfer` = base exposure + only SDXL's
  high-frequency structure (now the default `combine="detail"`). **Left with 2
  failing tests (test bugs, not module bugs)** — see the memory note
  `splat-quality-diagnosis-and-qa-gates` for the exact asserts to fix; then re-run
  `out/verify_mv.py`.

---

## 4. Current capability matrix (the real wiring, updated from doc 15)

The session-15 wiring audit is now stale (L4/L5/L6/L7, async jobs all landed since).
This is the **current** truth. Legend: ✅ real & input-conditioned · 🟡 honest
stub/placeholder (flagged) · ❌ missing.

| Layer / stage | Text·GPU | Image·GPU | Video·GPU | CPU-stub (any) |
|---|---|---|---|---|
| L0 seed | ✅ (subsample of L2) | ✅ | 🟡 static-recon frame or smoke | 🟡 torus |
| L1 dense | ❌ never written | ❌ | ❌ | ❌ |
| L2 coarse | ✅ TripoSplat (from SDXL image) | ✅ TripoSplat | ❌ | ❌ |
| L3 refined | ✅ **but = frozen distillation** | ✅ same | 🟡 static L3, "dynamics not tracked" | 🟡 torus as l3 |
| L4 appearance | 🟡 CPU SH-L2 estimate | 🟡 | 🟡 | 🟡 |
| L5 collision/print | ✅ | ✅ | ✅ | ❌ (stub skips) |
| L6 physics-material | 🟡 offline fixture, **text-only** | ❌ (modality guard) | ❌ | ❌ |
| L7 dynamics | ❌ | ❌ | ❌ **no real tracking** | ❌ |
| Generation Spec (LLM) | 🟡 offline fixture | ❌ | ❌ | 🟡 |
| Exports .ply/.spz/.sog/.astel | ✅ | ✅ | ✅ | ✅ |
| glTF KHR (SH0) | ✅ via `astel_splat_io.gltf` (library; not auto-emitted per asset) | ✅ | ✅ | ✅ |
| Billing | ✅ | ✅ | ✅ | ✅ |

**The only fully input-conditioned, real-geometry cells are Image·GPU and Text·GPU
(via SDXL→TripoSplat).** Capture (multi-photo) and video remain offline-eval-only /
honest-placeholder. SH is degree 0 everywhere.

---

## 5. Updated hardware reality (corrects stale assumptions) ⚠️

- **We are on Box A = `THREADRIPPER-48`, 2× RTX 4090 (24 GB each), native Windows**
  (CUDA 12.9 + VS 2026, arch 8.9; WSL2 hard-blocked by firmware). Early research
  framed the 3080 box (10–12 GB) as preview-tier-only and the single 4090 ceiling as
  "tight" for a 24 GB model like TRELLIS.2-4B — but **Box A has two 4090s**, so a
  generator on one GPU + a refine on the other is feasible; the JobManager currently
  serialises GPU work with a `Semaphore(1)` to be safe on single-GPU VRAM, **leaving
  the second 4090 idle** — a free lever for the multi-view pipeline.
- **Measured footprints leave huge headroom:** TripoSplat 4.6–4.9 GB / ~11 s;
  2DGS refine ~5 GB; densified refine peaked **1.2 GB**. The 24 GB ceiling is **not**
  the current bottleneck. The real ceiling is **algorithmic** (single image, SH0,
  no real refine), not VRAM.
- **What 24 GB *does* cap:** the **5M-splat cinematic tier** (needs the 32–48 GB
  recommended tier per CLAUDE.md §6) and a co-resident TRELLIS.2-4B-class model.
  Standard (1M) and lowpoly (100k) budgets are comfortable.
- **Runtime fragility to keep in mind:** gsplat JIT needs `cl.exe` on PATH (use
  `run-python.cmd`); GPU work must be driven via the PowerShell tool / the launcher
  by absolute Windows path, not Git Bash.

---

## 6. The quality problem — root-cause diagnosis (why text→splats look "painted clay")

The text→3D path produces an honest, watertight-derivable, layered asset that is
**not photorealistic**. This is **architectural, not a bug**, and was root-caused
across sessions 26/32/33/34. The default production pipeline is:

```
prompt
  → SDXL-base best-of-N (ONE chosen 2D image)          ← single-image conditioning
  → TripoSplat (single-image feed-forward, 262k, SH0)  ← quality ceiling lives here
  → splat_clean (connected-components floater removal)
  → 2DGS "L3 refine" = distillation @512px, FROZEN positions (means_lr_scale=0.0)
  → L4 (post-hoc CPU achromatic un-lighting), L5/L6, package
```

**Three deficits cap photorealism (each maps to a tier):**

| Deficit | Consequence | Fix tier |
|---|---|---|
| **Single-image conditioning** | back/sides hallucinated; identity drift around the object | **Tier 1**: real multi-view-consistent front end |
| **Refine adds no information** — it re-fits the L2's *own* renders with frozen positions & fixed count; it is **distillation/surfelization, not refinement**. CLAUDE.md §4's "L3 refinement with multi-view-diffusion guidance" is **NOT implemented.** | nothing recovers detail or fixes geometry; the asset can only *lose* vs L2 | **Tier 1**: un-frozen densified refine vs **external** multi-view targets |
| **SH degree 0** — the entire format (`SplatCloud.colors_dc` only; no `f_rest`/higher-order SH anywhere; .ply omits `f_rest_*`; glTF exports `sh_degree:0`) | flat, view-independent color → no specular, no glints → "painted clay" | **Tier 2**: SH degrees 2–3 (+ later per-gaussian BRDF for real relighting) |

**What has been tried, and the measured outcome:**

- **Tier 0 (reliability) — DONE & verified.** Best-of-N image selection + image/
  geometry QA critics fixed "same prompt sometimes wrong" and degenerate clouds.
  This raised the *floor*, not the *ceiling*.
- **Tier 1 engine — BUILT, but proven only on synthetic.** The densified un-frozen
  refine recovers detail decisively when the target carries new info (synthetic
  8.87→31.77 dB). On the real cloud against **self-render** targets it **loses** to
  distillation (20.14 vs 23.05 dB) because self-renders contain no new information.
  **The engine is correct; it is starved.** It is OFF by default for exactly this
  reason — enabling it in production *today* would regress quality.
- **MV-enhance (the no-new-weights external-target source) — IN PROGRESS, not
  working yet.** SDXL img2img injects real detail per view, but per-view exposure
  inconsistency collapses the refined asset; the `detail_transfer` fix is half-landed
  with 2 failing tests left to repair. This is the cheapest path to a *real* external
  target, but it is fundamentally limited (img2img is not 3D-consistent).

**The honest conclusion:** to break the ceiling you must feed the (already-built)
densified refine with **genuinely multi-view-consistent, higher-information
targets**, and you must extend the format to carry **view-dependent color (SH)**.
Everything else (cleanup, QA, billing, layers) is already in place. See §10.

---

## 7. The TRELLIS situation — definitive (founder's specific concern)

**What happened:** A recent session investigated TRELLIS as the Tier-1 multi-view
front end, concluded "TRELLIS.2 outputs mesh+PBR, is Linux-oriented and needs
flash-attn/nvdiffrast/custom CUDA — infeasible on native Windows," and stopped. That
conclusion is **correct about TRELLIS.2** but it **re-derived a fact already
established in session 1/2** and it **missed that the usable model is TRELLIS v1's
gaussian head**, which is a different thing.

**The settled facts (from RA2 session 2 + the session-12 import-graph audit):**

1. **TRELLIS.2 = mesh-only.** The v1 gaussian/RF decoders were dropped; O-Voxel ↔
   mesh; PBR texturing needs `nvdiffrast`/`nvdiffrec` (NVIDIA **non-commercial**).
   **Correctly off-thesis for a splats-only product.** Known since session 2. Do not
   revisit unless we want it purely as an *internal* geometry prior to distill (the
   retired R-T1 bet), which needs a 1-line MIT fork patch to `o_voxel/__init__.py`.
2. **TRELLIS v1 has a native gaussian head that is clean and MIT.**
   `TrellisImageTo3DPipeline.run(image)['gaussian']` imports **zero**
   nvdiffrast/nvdiffrec/diffoctreerast **as long as you don't import
   `trellis.utils.render_utils` / `postprocessing_utils`** (use our own gsplat
   preview renderer — already the plan). Verdict from audit 12: *"clean as-is, no
   fork needed."*
3. **The real Windows risk for v1 is its sparse-attention stack** (spconv /
   flash-attn / xformers), **not** nvdiffrast. Prebuilt wheels target
   cu124/torch2.5/py3.10 while Box A is cu128/torch2.11/py3.12, so we either find
   matching wheels, use `ATTN_BACKEND=xformers` + `SPCONV_ALGO=native`, or build via
   vcvars like we did for gsplat (risk R-T9 — surmountable, never attempted).

**Why it isn't wired:** TripoSplat won the L2 decision on cleaner deps + measured
metrics + native gaussians (DECISIONS #2, session 14). TRELLIS v1 was kept as the
documented runner-up; the v1-vs-TripoSplat A/B was deferred and never run.

**The strategic point:** TRELLIS v1's gaussian head is the **strongest open
alternative** if we want a *better* single-image L2 (structured latents → better
global geometry/identity). But note **it is still single-image conditioning**, so by
itself it does not solve the multi-view deficit — it raises the L2 ceiling, it does
not change the architecture. The architectural unlock (multi-view consistency) is a
**multi-view diffusion** stage (MV-Adapter / current SOTA), not TRELLIS v1.

---

## 8. Gaps ledger — decided/claimed vs. actually implemented (corrections) ⚠️

The founder noted "master repo for GS KHR and some others are not implemented to my
knowledge." Here is the verified truth, including the things that **are** built:

| Item | Believed | **Verified reality** |
|---|---|---|
| **glTF `KHR_gaussian_splatting` export** | not implemented | ⚠️ **IMPLEMENTED** (session 27, `astel_splat_io/gltf.py`: `write_gltf`/`read_gltf`, RC schema, golden+round-trip tests). **Caveats:** exports **SH degree 0** only; RC schema may churn before Q2-2026 ratification; it's a **library function, not auto-emitted into each `.astel`** package; **no headless engine/glTF-viewer round-trip test** (only in-repo round-trip). |
| **Unity package + UE5 plugin** | unsure | ✅ **code-complete** (session 27, `plugins/unity/`, `plugins/unreal/AstelPlugin/`) with physics auto-setup from L5/L6 + NUnit/struct tests — **but never compiled against a real Unity/UE5** (no licensed CI runners). Logic is unit-tested in pure code only. |
| **Python SDK / TS SDK / MCP server** | unsure | ✅ **implemented + unit-tested** (session 27, `packages/sdk-python`, `packages/sdk-ts`, `mcp_server.py` with 3 FastMCP tools). MCP `[mcp]` extra optional. Not yet published to PyPI/npm. |
| **USD / USDZ export with splat payload** | listed in §1.5 | ❌ **never built.** |
| **Higher-order SH (degree 1–3)** | implied "photorealistic" | ❌ **not in the format at all** — `SplatCloud` carries only band-0 `colors_dc`; .ply/.spz/.sog/gltf all SH0. Spark *viewer* supports SH0–3; we never feed it. |
| **Multi-view diffusion (MV-Adapter)** | part of §4 text pipeline | ❌ **never built.** Text path is single-image. |
| **TRELLIS (v1 GS head)** | "tried, unusable" | ❌ never installed/wired; the *usable* v1 head was conflated with the mesh-only v2 (§7). |
| **L1 dense cloud** | core layer | ❌ no producer ever writes `l1.ply`. |
| **Real PGSR multi-view L3 losses** | DECISIONS ✅ "technique settled" | ❌ never implemented (the largest geometry-accuracy lift). |
| **Capture/video reconstruction in the product** | M2/M6 | 🟡 COLMAP runner real + DTU-validated **offline**; MapAnything never installed; **neither wired into the API producer**; video → static placeholder. |
| **MPM physics (Warp) / GPU L4 inverse-render** | §8 features | ❌ web sandbox is single-rigid-body; L4 is a CPU estimate. |
| **CI** | "green CI" in retros | ❌ **no CI ever ran.** `.github/workflows/ci.yml`+`gpu.yml` are **authored** (session 29) but the repo is **local-only** (git log: `Beta`,`V0.1`,`V0.2`,`V0.6×4` — no remote). "All gates green" = a **manual** ritual. |
| **Fine-tuning / custom models** | §6 "later" | ❌ not started (correctly — gated on ≥10k telemetry-labelled generations + a >$1k/mo founder decision). |

---

## 9. The honest backlog — three tracks (from `18-post-m6-roadmap.md`)

- **Track N — Launch hardening (infra, blocks any launch):** N1 **push to a GitHub
  remote so CI actually runs** (authored, never executed); N2 `.astel`
  deserialization hardening (zip-bomb/path-traversal/`validate_untrusted()`); N3
  production deploy + load validation; N4 monitoring/SLIs/rollback runbook; N5 engine
  CI runners (licensed).
- **Track G — GPU-real (closes the honest scaffolding gaps):** G1 real 4DGS video
  L7; **G2 text→multiview bridge (the quality unlock — see §10)**; G3 live LOD
  streaming in the viewer; G4 scene generation end-to-end; G5 MPM sandbox + GPU L4
  inverse-render.
- **Track T — Fine-tuning (deferred, founder-gated):** T1 domain-adapt the L2 prior
  on accepted generations; T2 multi-view diffusion model; T3 metric-scale head; T4
  small L6 material model. Start only after Track N + ≥10k labelled generations + a
  measured recurring deficit + a written cost approval (>$1k/mo).

---

## 10. What to do next to fully realize photorealistic text → Gaussian splats

The product is feature-complete and honest; the missing thing is *visual fidelity*,
and the path to it is now unambiguous because the diagnosis, the levers, and most of
the machinery already exist. **The single highest-leverage program is to convert the
L3 stage from a frozen distillation into a real, information-adding refinement driven
by genuinely multi-view-consistent supervision, and to teach the whole format to
carry view-dependent color.** Concretely, in priority order: **(1)** finish the
half-landed **MV-enhance `detail_transfer`** fix (repair the two failing test asserts,
re-run `out/verify_mv.py`, confirm the asset is no longer dark and gains real detail)
so the already-built densified refine finally has *external* targets to chase — this
is the cheapest experiment and turns the existing `densify.py`/`refine.py` engine from
"correct but starved" into a measurable win; **(2)** in parallel, build the real
**text→multi-view-diffusion bridge** (MV-Adapter or the current SOTA — re-verify at
build time, it is the principled fix for the single-image hallucination that
img2img can only approximate) so text yields several *3D-consistent* conditioning
views instead of one, optionally pairing it with a head-to-head against the
**TRELLIS v1 gaussian head** as a stronger single-image L2 (it is MIT, nvdiffrast-clean,
and Windows-feasible via xformers/vcvars — the only real cost is the sparse-attention
install, never attempted); **(3)** then undertake the **SH-degree milestone** —
extend `SplatCloud`, .ply/.spz/.sog/glTF writers, and the Spark viewer to carry SH
degree 2–3 with golden round-trip tests (large blast radius, do it as its own
milestone), because view-dependent specular is what removes the "painted-clay" look
even after geometry is correct; and **(4)** finally, reimplement the **PGSR
multi-view geometric/photometric losses** (the largest geometry-accuracy lift, fully
specified in RA8 §2 but never built) and replace the post-hoc CPU L4 with a real
**deferred-PBR-on-gsplat inverse render** so relighting is measured, not approximated.
Throughout, exploit the **idle second 4090** (run the generator on one GPU and the
densified refine on the other) and **enforce the RA9 quality floors in CI** (push to
a remote first — N1) so each of these wins is regression-gated and the Truth Meter's
numbers become real invariants rather than point measurements. Do **not** chase
fine-tuning (Track T) yet: every lever above uses permissive off-the-shelf checkpoints
and our own code, and fine-tuning is correctly gated behind ≥10k telemetry-labelled
generations and a founder cost decision. In short — the research is done, the layers
are built, the GPU has headroom; photorealism is now an **engineering program of
four well-scoped steps (multi-view targets → multi-view conditioning → SH → PGSR/PBR)**,
not an open research question.
