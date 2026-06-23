# 20 — The Photorealism Program (committed plan)

> **Written 2026-06-21. Decision authority: founding engineer per CLAUDE.md §10.2
> ("decide and document").** This is the binding program to take Astel's text/image
> → Gaussian-splat output from "honest but painted-clay" to **photorealistic, highly
> detailed, and provably surface-accurate** — the level where it beats textured mesh
> on the content that matters. It executes on the diagnosis in
> [`19-master-state-and-photorealism.md`](19-master-state-and-photorealism.md).
> Status legend: ✅ built · 🟡 partial/opt-in · 🔨 to build · ⚠️ founder-gated.

---

## 0. The goal, stated honestly (the critique up front)

"Surpassing every mesh in quality and visuals" is a **winnable claim — but not
uniformly.** As the engineer, I will not let us ship hype we can't defend:

- **Where splats genuinely and provably beat mesh** (this is our ground): real
  captured objects/scenes, organic and fuzzy matter (hair, fur, foliage, fabric),
  **reflective/specular/translucent materials**, and anything where **view-dependent
  radiance** matters. A textured mesh fakes these with shaders and never matches a
  splat's per-view appearance. This is the splat killer feature and we lean into it.
- **Where a clean mesh can still look sharper**: hard-surface, low-texture, CAD-like
  objects with crisp edges (a splat's gaussians soften 90° creases). We answer this
  with the surface-aligned 2DGS + PGSR geometry path and the SDF/watertight proof —
  not by pretending the failure mode doesn't exist.
- **The defensible headline**: *"photoreal, relightable, and we publish the measured
  error — splats that look like the real thing and prove they're faithful."* The
  **Truth Meter** is the weapon no mesh competitor will copy.

So the program targets three things at once: **photorealism** (view-dependent
appearance), **detail** (high-frequency texture + density where it's needed), and
**solidity/accuracy** (geometry that the SDF/Truth-Meter can certify). All three, or
it's not "mesh-surpassing."

---

## 1. The thesis — four capabilities, in dependency order

The current ceiling is `SDXL (1 image) → TripoSplat (single-image, SH0) → frozen
distillation`. Photorealism needs four things, and **they have a strict dependency
order** — doing them out of order wastes effort (e.g. SH degree 3 learns nothing
without multi-view targets, because self-renders are SH0 and contain no
view-dependence to fit):

1. **(KEYSTONE) Multi-view-consistent, high-resolution conditioning.** Replace the
   single image with N (~6–8) 3D-consistent orbit views. *Without this, the refine
   and SH are starved — proven on Box A (densify lost 20.1 vs 23.0 dB on self-render
   targets).* Everything else depends on it.
2. **A refinement that actually adds information.** Un-frozen, **densified** 2DGS
   optimization against those multi-view targets, with perceptual + multi-view
   geometric/photometric losses, anti-aliasing, and diffusion guidance for genuinely
   unseen regions (confidence-gated — never hallucinate over measured data, §1.4).
3. **View-dependent appearance, then relightability.** SH degree 3 (kills the
   painted-clay look) → then a real per-gaussian PBR/BRDF inverse-render (true
   relighting). This is the single biggest *visual* win once geometry is right.
4. **Budget & scale.** Hero assets at 1M splats (Box A), cinematic 5M on the cloud
   tier. Detail needs both density *and* high-res supervision.

---

## 2. The committed architecture — the "Astel Hero Pipeline"

```
TEXT
 │  Generation Spec (LLM, offline ok)  →  canonical single-object prompt
 ▼
 SDXL / FLUX  →  1 hero image            ── [Tier-0 best-of-N + image QA: ✅ built]
 ▼
 MULTI-VIEW DIFFUSION  (MV-Adapter on SDXL; re-verify SOTA at build)   ── 🔨 P1
 │   → N ~6–8 orbit views, 3D-consistent, 1024px+, high-frequency detail
 ▼
 FEED-FORWARD L2 INIT  (TripoSplat 262k; A/B TRELLIS-v1 GS head)       ── ✅ / 🔨 A/B
 │   → cheap previewable gaussians, geometry seed
 ▼
 splat_clean  (connected-components floater removal)                   ── ✅ built
 ▼
 REAL L3 REFINE  (gsplat 2DGS, UN-FROZEN + ADC densify 262k→1M)        ── 🟡 engine built, 🔨 wire+losses
 │   loss = L1+D-SSIM  + LPIPS perceptual  + PGSR multi-view geo/photo  ── 🔨 PGSR (P2)
 │        + SDS diffusion guidance (unseen regions, confidence-gated)   ── 🔨 (P2)
 │        + Mip-Splatting anti-aliasing                                 ── 🔨 (P2)
 │   appearance = optimize SH degree 0→3                                ── 🔨 format milestone (P3)
 ▼
 L4 RELIGHT  (deferred-PBR-on-gsplat inverse render: albedo/rough/metal/env)  ── 🔨 P5
 ▼
 L5 SDF→watertight (solidity proof)  +  TRUTH METER (measured error)   ── ✅ built
 ▼
 EXPORT  .ply/.spz/.sog/.astel/glTF-KHR(SH3)  +  1M/5M LOD tiers        ── ✅ / 🔨 SH3 in writers (P3)
```

The **capture path** (real photos/video → MapAnything/COLMAP → L1 → the *same* P2
refine) is the strongest "beats mesh" proof and converges on the identical L3 engine
— it is folded into P6, not a separate pipeline.

---

## 3. Binding decisions

| # | Decision | Rationale | License / Cost |
|---|---|---|---|
| **D20.1** | **Conditioning becomes multi-view.** Adopt a multi-view diffusion stage (default **MV-Adapter** on SDXL; **re-verify the current best MV-diffusion / MV-LRM at build time** per §10.1) producing N≥6 consistent ~1024px views. The single-image path stays as a fast preview fallback. | The single-image deficit is the #1 cause of hallucinated backs + identity drift; consistent multi-view is the only principled fix (RA2, doc 11 D-T1.1). | MV-Adapter permissive (LICENSE_AUDIT ✅); runs on Box A — **no new spend**. |
| **D20.2** | **L3 becomes a real un-frozen densified refine, supervised by the multi-view targets.** Make `ASTEL_L3_REFINE` the default **once it beats distillation on the eval corpus** (not before — it regresses on self-renders, measured). | The frozen distillation cannot add information; the densified engine is built + GPU-validated and only needs real external targets. | Our code; Box A — no spend. |
| **D20.3** | **Reimplement the PGSR multi-view loss set** (edge-aware single-view normal, FB-reprojection geometric, 7×7 patch-NCC photometric, exposure affine) on top of 2DGS + **LPIPS** + **Mip-Splatting AA** + **SDS guidance** (confidence-gated). | PGSR is the published surface-accuracy leader (DTU 0.52 mm); the math is fully specified (RA8 §2) and was decided ✅ but never built. | Our code on gsplat (Apache); no NC. No spend. |
| **D20.4** | **Ship SH degree 3.** Extend `SplatCloud` + .ply/.spz/.sog/glTF writers + the Spark viewer to carry SH 0–3; optimize SH coeffs in the refine against the MV targets. Do it as **its own milestone with golden round-trip + coordinate-system tests** (large blast radius). | View-dependent radiance is the splat killer-feature and the cure for "painted clay"; the format is currently SH0-only (the structural root cause). Spark already supports SH0–3. | Our code; no spend. |
| **D20.5** | **Hero budget = 1M splats on Box A; cinematic 5M = cloud H100 tier (deferred).** | 1M fits Box A comfortably (densified refine peaked 1.2 GB); 5M needs the 32–48 GB tier (CLAUDE.md §6). | 1M: no spend. **5M/cloud: ⚠️ >$1k/mo founder gate** (deferred, see §6). |
| **D20.6** | **Real relightable L4** (deferred-PBR-on-gsplat: albedo/metallic/roughness + jointly-optimized env), driven by real 2DGS normals, replacing the CPU achromatic estimate. | True relighting is a flagship differentiator (Relight Studio); recipe settled (RTR-GS, RA8 §6). | Our code; no spend. Optional 3dgrut (Apache) for validation only. |
| **D20.7** | **L2 generator: keep TripoSplat as the spine; A/B the TRELLIS v1 gaussian head** as a stronger init in parallel (not on the critical path). | TripoSplat is clean + shipping; TRELLIS v1 (MIT, nvdiffrast-clean, §7 of doc 19) may give better global geometry — but it's still single-image, so it raises the init ceiling, it doesn't replace D20.1. | Both MIT; v1 install risk = sparse-attention on Windows (R-T9, surmountable). No spend. |
| **D20.8** | **Lead the public "beats mesh" claim with the CAPTURE path; drive TEXT to photorealism behind it.** | Real-photo → relightable splat is where we *provably* and *immediately* beat mesh; text is harder (hallucination) and matures via D20.1–D20.6. Honest positioning (§0). | No spend; needs MapAnything install (Apache ckpt) + wiring (P6). |
| **D20.9** | **Every win is regression-gated.** Enforce the RA9 per-layer floors in CI (`eval_targets.yaml`) — requires pushing the repo to a remote so CI actually runs (roadmap N1). | "All gates green" is currently a manual ritual; mesh-surpassing claims must be CI invariants, not point measurements. | No spend (free GitHub Actions for CPU gates; self-hosted runner for GPU). |

**No decision above requires new spend on Box A.** The only founder-gated cost item
is the cinematic 5M / cloud tier (D20.5) and eventual fine-tuning (Track T) — both
explicitly deferred (§6).

---

## 4. Milestone plan (sequenced by dependency × ROI)

Each milestone ends green-CI + a measured before/after on the fixed eval corpus + a
turntable montage (`render_preview`). Order is **strict** where noted.

### P1 — Multi-view conditioning (the keystone) 🔨
- **Build:** `text_to_multiview.py` — base image → MV-diffusion → N consistent
  ~1024px orbit views with known camera poses. Wire as `external_targets` into the
  existing `run_l2_to_l3` (the seam already exists). **Finish the half-built
  `mv_enhance.py` `detail_transfer` path in parallel** as the no-new-weights interim
  + fallback (repair the 2 failing test asserts first — see memory
  `splat-quality-diagnosis-and-qa-gates`).
- **Exit gate:** MV views pass a 3D-consistency check (cross-view reprojection
  error below threshold); the densified refine supervised by them **beats
  distillation** on held-out PSNR/LPIPS on the corpus (reverses the measured 20.1 vs
  23.0 dB self-render result). *This unblocks P2–P5.*

### P2 — Real refine becomes the default 🔨 (engine ✅, losses 🔨)
- **Build:** PGSR loss set (D20.3) + LPIPS + Mip-Splatting AA + confidence-gated SDS;
  enable un-frozen ADC (built) to grow 262k→1M against the P1 targets; tune
  `grad_threshold`/`percent_dense`/`interval`/`λ_perceptual`.
- **Exit gate:** beats current distillation by a measured margin on held-out
  PSNR/LPIPS *and* Chamfer-vs-target; floaters gone; ADC fires; make
  `ASTEL_L3_REFINE` the default (D20.2).

### P3 — SH degree 3, the appearance milestone 🔨 (depends on P1/P2)
- **Build:** schema bump across `astel_splat_io.SplatCloud` + ply/spz/sog/gltf
  writers + Spark viewer; optimize SH 0→3 in the refine; golden round-trip +
  SH-rotation-across-coordinate-systems tests (the classic silent-corruption bug).
- **Exit gate:** visible specular/glints on reflective corpus assets; golden tests
  green; legacy SH0 `.astel` still load; glTF-KHR export carries SH3.

### P4 — Highly-detailed pass 🔨
- **Build:** 1024px+ targets (+ optional diffusion upscaler on the MV views); 1M
  default budget; ADC tuned for high-frequency texture; LOD tiers regenerated.
- **Exit gate:** LPIPS + edge-sharpness detail metrics hit RA9 targets; 1M hero asset
  renders crisp at all LOD levels.

### P5 — Real relightable L4 🔨
- **Build:** deferred-PBR-on-gsplat inverse render (D20.6) replacing the CPU L4
  estimate, behind the existing `LayerAppearance` contract; feed Relight Studio the
  real decomposition.
- **Exit gate:** relight PSNR vs held-out HDRIs ≥ 27 (RA9); albedo carries no baked
  shadow (Meshy's sin — hard gate).

### P6 — Prove it + capture lead + cinematic scale 🔨 / ⚠️
- **Build:** wire the capture path (MapAnything/COLMAP → L1 → the P2 refine) so real
  photos → relightable splats (D20.8); enforce RA9 floors in CI (D20.9, needs the
  remote push, N1); cinematic 5M on the cloud tier (⚠️ founder-gated, §6).
- **Exit gate:** published blind-eval (the built harness) beats Meshy-free / Tripo /
  raw TRELLIS on the corpus; Truth Meter numbers are CI invariants.

---

## 5. How we prove "surpasses mesh" (measurement, not vibes)

We already have the substrate (synthetic + DTU eval, blind-eval corpus, Truth
Meter); P6/D20.9 make it binding. The claims and their proofs:

| Claim | Proof |
|---|---|
| Photorealistic | Held-out NVS **PSNR ≥ 30 / LPIPS ≤ 0.12** on the product corpus (RA9), with SH3 view-dependence. |
| Highly detailed | LPIPS + high-frequency edge-sharpness vs source; 1M-splat hero tier. |
| Surface-accurate ("mesh-quality solid") | **Chamfer ≤ 1% of bbox diagonal** in measured regions + **100% watertight** SDF surface (RA9 hard gate) + COM/inertia sanity. |
| Relightable (beats baked-in mesh texture) | **Relight PSNR ≥ 27** under novel HDRIs; no shadow leak into albedo. |
| Honest (no competitor matches) | Truth Meter publishes measured error + provenance heatmap per asset, gated in CI. |
| Beats mesh on real capture | Blind side-by-side vs Meshy/Tripo on the capture corpus (D20.8). |

---

## 6. Cost & licensing gates (the only founder decisions)

- **Cinematic 5M / cloud H100 pool (D20.5):** crosses the >$1k/mo §10.2 threshold.
  **Deferred** — everything in P1–P5 runs on Box A at zero new spend; surface a
  written cost estimate before provisioning cloud GPUs.
- **Fine-tuning our own L2/MV models (Track T):** **deferred** until ≥10k
  telemetry-labelled generations + a measured recurring deficit + a written cost
  approval. The program above deliberately uses only permissive off-the-shelf
  checkpoints (SDXL/FLUX/MV-Adapter/TripoSplat/TRELLIS-v1) + our own code, so we get
  to photorealism **without** training a model.
- **LLM key (Generation Spec / L6):** modest (~$0.02–0.035/gen), already double-gated
  offline; optional, not on the photorealism critical path.
- **Licensing:** every adopted model is MIT/Apache (verified, LICENSE_AUDIT). No NC
  code/weights/data enters the shipped path. SDS guidance uses our local diffusion
  model — no external dependency.

---

## 7. Risks & honest caveats

- **MV-diffusion consistency is the make-or-break.** If MV-Adapter's cross-view
  consistency is too weak, the refine averages to mush (the exact failure seen with
  raw SDXL img2img). Mitigation: P1's consistency exit-gate; the `detail_transfer`
  fallback; and re-verifying whether a *native* multi-view feed-forward reconstructor
  (GS-LRM/LaRa-class — re-verify SOTA) is a better keystone than diffusion+optimize.
- **SH3 blast radius** touches the archival format and the viewer — a botched bump
  breaks every existing `.astel`. Mitigation: P3's golden round-trip tests + additive
  schema (legacy SH0 must still load).
- **Hard-surface sharpness** (§0) — splats soften crisp edges. Mitigation: 2DGS +
  PGSR edge-aware normals; position the claim on organic/captured/reflective content.
- **Scale-dependent λdist** (memory `l3-2dgs-decision`) — the depth-distortion weight
  is per-scene; build the dimensionless normalized λdist before the refine
  generalizes across object sizes.
- **Box A's idle second 4090** — exploit it: generator on GPU0, refine on GPU1 (the
  JobManager currently serialises with Semaphore(1); relax for the two-stage pipeline).

---

## 8. Recommended immediate next action

**Start P1.** Concretely, in this order: (1) repair the two failing `mv_enhance`
test asserts and re-run `out/verify_mv.py` to confirm `detail_transfer` no longer
darkens the asset (the cheapest possible signal that *any* external target helps);
(2) in parallel, stand up the MV-diffusion stage (MV-Adapter on the already-installed
SDXL — re-verify it's still the best plug-and-play option) and wire its views as
`external_targets`; (3) run the P1 exit-gate A/B on the corpus. The moment the
densified refine beats distillation on real external targets, the architecture is
unblocked and P2→P5 are execution, not research. Push the repo to a remote early
(D20.9 / N1) so every step is regression-gated from the start.

> **Bottom line:** the research is done, the layers are built, the GPU has headroom,
> and the path is four ordered capabilities — **multi-view targets → real refine →
> SH3 → relightable PBR**, proven on the Truth Meter and led publicly by the capture
> path. No new model training, no new spend on Box A. This is an engineering program,
> not an open question.

---

## 9. Session 2026-06-23 — extreme-8K + densification experiments (measured)

Goal this session: "complete all graphical enhancements" — push the MV path to its
quality ceiling (more views, more splats, crisper 8K). All runs on Box A GPU 1
(server kept live on GPU 0), prompt = engraved knight's-helmet. Drivers:
`pipelines/gpu/out/mv_extreme.py` (kept). Findings, honest:

**WIN — multi-ring 12-view reconstruction.** Adding a **+30° elevation ring** (2× 6-view
MV-Adapter calls, same prompt/seed → 12 views) to the equatorial ring lifted
**geom_qa 0.674 → 0.713** with no ghosting. Per-ring fit PSNR (ring0 25.4 / ring1 23.8 dB,
gap 1.6 dB) confirms the elevated ring is correctly framed — the `ortho_cameras`
elevation convention generalises (also covered by the CPU camera tests). The extra
ring gives real crown coverage the single equatorial ring lacks. **This is the one
proven enhancement and should graduate**: make the T2MV path do a 2-ring spec by
default (cost: one extra MV-Adapter call). Asset: `out/mv_extreme/recon12.ply`.

**NEGATIVE — "more splats" via aggressive densification does NOT help.** Lowering the
ADC `grad_threshold` 3.5e-5 → 1.8e-5 with init 300k did not grow the cloud (stayed
~300k: prune balances clone/split) and geom_qa *dropped* to 0.62. The cloud size is
prune-limited, not threshold-limited; sparsity is not the 8K problem.

**NEGATIVE — anisotropy regularisation collapses the fit in the ADC pipeline.** The 8K
"splat-soup" streaks are elongated (needle) gaussians, so a scale-anisotropy penalty
is the textbook fix — but here every nonzero weight collapsed the cloud (245k → 12–58k)
and either melted the engraving (rounded, blurry) or left it *more* streaky (sparser).
Root cause: a scale penalty fights the densify/prune dynamics — it shrinks needles and
the opacity-prune then deletes them, instead of rounding them. (Also surfaced and fixed
a real bug while prototyping: a `smax/smin` ratio penalty has an **unbounded gradient as
`smin→0`** — the exact needle case — which blows up even at λ=0.001; the bounded linear
hinge `relu(smax−r·smin)` still collapsed via the prune interaction.) The knob was
**reverted** from `mv_reconstruct` (didn't graduate → deleted per §1.7); package gates
stay green (181 passed).

**CONCLUSION — the 8K extreme-zoom ceiling is real and representation-level, not a render
knob.** At ~250k splats from 1024px source views, extreme 8K pixel-peep is fundamentally
streak-or-blur: the streaks *are* the high-frequency engraving. At normal / 2K viewing
the 12-view asset is genuinely museum-grade (clean polished steel, crisp engraving, no
body floaters). The real levers for crisp 8K are **pipeline-level**, in priority order:
1. **Higher-resolution source views** (>1024 — SDXL/MV-Adapter native cap; needs a tiled
   / upscaled multi-view generator) so there is true high-frequency signal to fit.
2. **A surface representation for extreme zoom** (2DGS surfels + normal/detail maps, or a
   textured-surface proxy *derived* from the splats) — gaussians soften at pixel-peep by
   nature; this is the documented hard-surface failure mode (§0).
3. SH3 view-dependent appearance for the steel glints (still the deliberately deferred
   format-chain item; orthogonal to sharpness).

**Remaining known artifacts on the 12-view asset:** (a) a floater/mush cloud in the open
neck cavity — a visual-hull limitation (no view looks *into* the concavity; a bottom/low
ring or a bounded-volume prune would carve it); (b) weak crown apex (the +30° ring is not
a true top-down view).
