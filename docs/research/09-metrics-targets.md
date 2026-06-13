# RA9 — Per-Layer Accuracy Metrics & CI-Gated Targets

*Session 2, 2026-06-13. Defines, per layer L0–L7, the metric, the reference it is measured
against, the measurement protocol, the initial target (justified from the deep-read paper numbers
in [08-deep-reads.md](08-deep-reads.md)), and the CI-failing regression. This is the substrate for
spec §10.3 ("measure everything; regressions fail CI") and the **Truth Meter** (spec §8.4).*

**Conventions used below**
- **Held-out views**: every capture/generation reserves N test views (default **1 in 8**, min 3)
  never seen by the optimizer; PSNR/SSIM/LPIPS computed on these.
- **LPIPS**: AlexNet backbone, lower = better. **SSIM**: standard, higher = better.
- **Chamfer**: bidirectional mean point-to-point distance (mm for object-scale), measured
  against the stated reference, **not** against ground-truth scans we don't have — against
  **L1 metric points** (our own measured layer) for the product, and against dataset GT only in
  the **benchmark corpus** runs.
- **CI gate**: each metric has (a) a **fixed-corpus absolute floor** (build fails if below) and
  (b) a **regression delta** vs the last green commit on the same asset (build fails if it
  regresses beyond the noise band). Both are needed: floors catch "never good enough," deltas
  catch silent erosion.
- **Targets are launch-realistic, not paper-peak** — paper numbers are on curated benchmarks
  (DTU/T&T/Mip-NeRF360) with dense views; our in-the-wild corpus is harder. Floors are set
  **below** paper numbers with the gap noted; benchmark-corpus runs are held to paper-adjacent
  numbers.

---

## Two test corpora (both in CI)

1. **Benchmark corpus** (apples-to-paper): a handful of DTU scans, Tanks&Temples scenes,
   Mip-NeRF360 scenes, TensoIR/Stanford-ORB relight sets. Used to prove our reimplementations
   match published math. **Held to paper-adjacent numbers.**
2. **Product corpus** (spec §M2/M3, the blind-eval harness corpus — 20 text / 20 image / 10
   capture): in-the-wild assets, the real bar. **Held to launch floors.** Targets here are the
   ones that gate releases.

---

## L0 — Seed / Sparse Point Cloud

| Field | Value |
|---|---|
| **Metric(s)** | (a) Pose accuracy: rotation geodesic err (°), translation err (% of scene extent); (b) sparse reprojection RMSE (px); (c) **per-point confidence calibration** (does reported confidence predict error?) |
| **Reference** | Benchmark: dataset GT poses. Product: GLOMAP/COLMAP BA solution as pseudo-GT (front-end init scored against the refined solve). |
| **Protocol** | Run MapAnything front-end → compare poses to reference; reprojection RMSE on matched sparse points; confidence calibration via reliability-diagram ECE. |
| **Initial target** | Two-view rel ≤ **0.20** (MapAnything reports 0.18, arXiv:2509.13414); single-image calibration angular err ≤ **2.0°** (paper 1.18°); pose rot err ≤ **2°** median on dense orbits post-BA. |
| **CI-fail** | Reprojection RMSE > 1.5× green baseline; calibration ECE > 0.1; any pose median rot err regressing > 0.5°. |

## L1 — Dense Cloud (metric-scaled, normals, semantics)

| Field | Value |
|---|---|
| **Metric(s)** | (a) **Metric-scale accuracy**: abs-rel depth error; (b) **scale-CI coverage**: does the reported confidence interval contain truth at its stated rate (e.g. 90% CI covers 90% of the time)?; (c) normal angular error (°); (d) point completeness/accuracy (Chamfer halves). |
| **Reference** | Benchmark: GT metric depth/scale. Product: consensus (MapAnything + MoGe-2 + SfM/EXIF) cross-check + any known-size reference object in scene. |
| **Protocol** | abs-rel = mean(|d_pred − d_gt|/d_gt) on held-out depth; **CI coverage** = fraction of assets where the reported scale CI brackets the cross-check scale, over the corpus; binned to verify calibration. |
| **Initial target** | abs-rel ≤ **0.08** images-only (MapAnything multi-view 0.057; we add a margin for object-centric/in-the-wild), ≤ **0.05** with intrinsics+poses. **Scale-CI coverage ≥ 0.85 at nominal 0.90** (honest-but-slightly-conservative is acceptable; over-confident is a hard fail). |
| **CI-fail** | abs-rel > floor on benchmark; **scale-CI coverage < 0.80** (under-covering = the Truth Meter lying = spec §10.4 violation, hardest gate); normal err regressing > 2°. |

## L2 — Coarse Gaussians (TRELLIS v1 GS head)

| Field | Value |
|---|---|
| **Metric(s)** | (a) Identity/shape fidelity vs conditioning: CLIP-image similarity (image→3D) and render-vs-input LPIPS; (b) held-out NVS PSNR/SSIM/LPIPS; (c) splat count within requested budget. |
| **Reference** | The conditioning image(s); for benchmark, GT renders. |
| **Protocol** | Render L2 from input viewpoint → LPIPS/CLIP-sim vs conditioning; NVS metrics on held-out angles. L2 is a *preview* tier — bar is "judge shape/identity," not final quality. |
| **Initial target** | Render-vs-input LPIPS ≤ **0.20**; CLIP-image-sim ≥ **0.85**; NVS PSNR ≥ **22** (coarse). Speed: ≤ **30 s** on 4090 (preview economics, spec §7). |
| **CI-fail** | CLIP-sim regressing > 0.03; LPIPS regressing > 0.02; budget overrun (MCMC count off by > 1%). |

## L3 — Refined Surface Gaussians (the hero layer)

| Field | Value |
|---|---|
| **Metric(s)** | (a) **Geometric: Chamfer vs L1 metric points** (product) and **Chamfer vs GT** (benchmark); (b) **normal consistency** (mean angular error of rendered normal vs depth-gradient normal, the 2DGS `L_n` quantity); (c) NVS **PSNR/SSIM/LPIPS** on held-out views; (d) splat count = requested budget. |
| **Reference** | Product: **L3 vs L1** (does the refined surface stay faithful to measured points — spec §3 "measured geometric error vs L1"). Benchmark: DTU/T&T GT. |
| **Protocol** | Sample L3 surface points (median-depth render fused), bidirectional Chamfer to L1 cloud (measured-region only; generated regions excluded and tracked separately via provenance). Normal-consistency from the 2DGS render. NVS on held-out. |
| **Initial target (benchmark)** | DTU mean Chamfer ≤ **0.9 mm** at first (2DGS-only baseline reports 0.80; arXiv:2403.17888), **target ≤ 0.6 mm** once PGSR multi-view losses land (PGSR 0.52; arXiv:2406.06521). T&T F1 ≥ **0.35** → **0.50** with PGSR terms. Mip-NeRF360 PSNR ≥ **27** (paper 27.25). |
| **Initial target (product)** | **Chamfer(L3,L1) ≤ 1% of object bounding-box diagonal** in measured regions (metric, scale-aware — this is the spec's "geometric error vs L1" headline number, reported per asset). Held-out PSNR ≥ **26**, LPIPS ≤ **0.18**. Normal mean-angular-err ≤ **15°**. |
| **CI-fail** | Benchmark Chamfer > floor or regressing > 5%; product Chamfer(L3,L1) > 1.5% bbox; PSNR regressing > 0.5 dB; **any L3 splat placed in a measured region that contradicts L1 beyond 2σ without being flagged generated** (provenance-integrity gate, spec §10.4). |

## L4 — Appearance / Lighting (deferred PBR on gsplat)

| Field | Value |
|---|---|
| **Metric(s)** | (a) **Relighting PSNR/SSIM/LPIPS** under novel HDRIs (the decomposition test); (b) NVS PSNR under original lighting; (c) **albedo/material plausibility** (no baked shadows in albedo — measured via albedo-vs-shaded correlation or a held-out-light delta). |
| **Reference** | Benchmark: TensoIR / Shiny-Blender / Stanford-ORB relight GT (RTR-GS protocol). Product: round-trip — re-render under the *estimated* env and compare to input (self-consistency), plus the SH-bake export vs full-PBR render delta. |
| **Protocol** | Relight under K held-out HDRIs → PSNR vs GT (benchmark). Product self-consistency: |input − render(albedo,material,env_est)| held-out-view PSNR. Export check: PBR-render vs SH-bake-render LPIPS (must be small = faithful one-way bake). |
| **Initial target** | Relight PSNR ≥ **27** (RTR-GS 28.9 on ORB, 30.1 TensoIR; arXiv:2507.07733 — we set floor below, real assets harder); NVS PSNR ≥ **30**; **albedo contains no high-freq shadow** (shadow-leak metric below threshold). |
| **CI-fail** | Relight PSNR regressing > 0.5 dB; **shadow leaking into albedo above threshold** (Meshy's historical sin — spec L4 hard gate); SH-bake export delta LPIPS > 0.1 (export unfaithful). |

## L5 — Collision & Solidity (SDF → convex proxies, watertight print surface)

| Field | Value |
|---|---|
| **Metric(s)** | (a) **Watertightness** (boolean: closed manifold, no holes, no non-manifold edges); (b) **printability**: min wall thickness vs nozzle, overhang fraction, self-intersection count; (c) collision-proxy fidelity: convex-hull-set Hausdorff/volume-IoU vs L3 surface; (d) mass-property sanity: volume vs SDF volume, COM inside hull. |
| **Reference** | L3 surface (the SDF source) and the print constraints (nozzle/material profile). |
| **Protocol** | trimesh manifold/watertight check on marching-cubes output; SDF-erosion wall-thickness map; overhang from face normals vs build direction; CoACD proxy vs L3 volume-IoU; inertia tensor positive-definite + COM containment. |
| **Initial target** | **100% watertight** on all corpus assets (hard binary — a print file that isn't watertight is a defect, no tolerance); min wall thickness reported and ≥ profile min or flagged; convex-set volume-IoU ≥ **0.85** vs L3; COM inside convex hull **always**. |
| **CI-fail** | **Any non-watertight print mesh** (hard fail, zero tolerance); volume-IoU regressing > 0.05; mass-property check failing (non-PD inertia, COM outside hull); self-intersections > 0 in print mesh. |

## L6 — Physics-Material & Semantic (LLM/VLM + MPM)

| Field | Value |
|---|---|
| **Metric(s)** | (a) **Material-classification accuracy** vs a labeled subset (rigid/soft/cloth/fluid-adjacent); (b) density estimate within plausible range (e.g. wood 400–900, steel ~7850 kg/m³); (c) **MPM stability** (no NaN/explosion over a fixed sim-step budget); (d) part-segmentation IoU vs hand labels. |
| **Reference** | Hand-labeled subset of the corpus (materials, densities, parts). |
| **Protocol** | Run L6 LLM/VLM pass → compare class + density to labels; run a fixed Physics Sandbox scenario (drop on floor, N steps) → assert energy-bounded, no explosion; SAM-class part masks IoU vs labels. |
| **Initial target** | Material class accuracy ≥ **0.80** on labeled subset; density within 2× of reference material; **MPM sim completes N steps with bounded energy 100% of the time** (stability is non-negotiable for the sandbox demo); part-seg IoU ≥ **0.6**. |
| **CI-fail** | Any MPM explosion/NaN on the fixed scenario (hard); class accuracy regressing > 5%; density estimate outside plausible band without a low-confidence flag (provenance honesty). |

## L7 — Dynamics (video → 4DGS, M6)

| Field | Value |
|---|---|
| **Metric(s)** | (a) Per-frame held-out NVS PSNR/SSIM/LPIPS across time; (b) temporal consistency (flicker / warp error between adjacent rendered frames vs optical flow); (c) deformation plausibility (no tearing). |
| **Reference** | Held-out time-steps and held-out views of the dynamic capture. |
| **Protocol** | Reserve test frames in time **and** view; PSNR per held-out (t,view); temporal warp error via optical-flow-warped previous frame. |
| **Initial target (M6)** | Held-out-frame PSNR ≥ **25** (dynamic, harder than static); temporal warp error below static-scene baseline + margin. **Deferred to M6 — placeholder targets, revisit when L7 lands.** |
| **CI-fail** | (M6) PSNR regressing > 0.5 dB; temporal warp error regressing > 10%. |

---

## Cross-cutting CI gates (all layers)

1. **Provenance integrity (spec §10.4, "sacred"):** every gaussian from L0→export carries a
   measured↔generated scalar; CI asserts (a) the channel is never dropped by any stage, (b) no
   generated splat sits in a measured region without the flag, (c) the Truth Meter heatmap
   round-trips through export. **Hard fail.**
2. **Scale-CI honesty:** L1 scale-CI under-coverage (< 0.80 at nominal 0.90) is a **hard fail** —
   the system must never be over-confident about metric scale (spec §1.3 "honesty over hype").
3. **Export golden-files (spec §10.5):** .ply/.spz/.sog/glTF round-trip → reload headless →
   per-splat attribute match within tolerance; **SH-rotation golden test** across coordinate
   systems (Unity/UE/USD) — the classic silent-corruption bug (RA5 §4). Hard fail on mismatch.
4. **Resource budgets (spec §10.3):** every stage logs wall-time, VRAM peak, $-estimate; CI
   flags > 20% regression on any (soft warn) or > 50% (hard fail) — keeps preview economics
   (spec §7) intact.

## Measurement protocol notes

- All metrics computed by a single `auriga eval` harness over the two corpora; results written to
  a per-commit JSON the Truth Meter and CI both read (one source of truth).
- Floors and deltas live in a versioned `eval_targets.yaml` (not hardcoded) so tightening a
  target is a reviewed change, not a silent edit.
- **GPU-dependent numbers are deferred** (user decision 2026-06-13): the floors above are derived
  from paper-reported numbers (cited) as *initial* targets; first green GPU run on the corpus
  **replaces** them with measured baselines, after which the regression-delta gates take over.

## Sources (paper numbers behind the targets)
- 2DGS DTU 0.80 / T&T F1 0.32 / Mip360 PSNR — arXiv:2403.17888
- PGSR DTU 0.52 / T&T F1 0.52 / PSNR 27.25 — arXiv:2406.06521
- MapAnything rel 0.057–0.042 / calib 1.18° — arXiv:2509.13414
- RTR-GS relight PSNR 30.1/26.2/28.9 — arXiv:2507.07733
- TRELLIS GS-head render-loss (L1+D-SSIM+LPIPS) — arXiv:2412.01506
- PhysGaussian MPM stability/filling — arXiv:2311.12198
