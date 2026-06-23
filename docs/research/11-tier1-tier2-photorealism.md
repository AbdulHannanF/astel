# Tier 1 / Tier 2 — Production-grade, photorealistic splats (plan + decisions)

> Status: **decision + foundation landed** (this session). The Adaptive Density
> Control engine (`densify.py`) and the densified refine loop (`refine.py`) are
> implemented and CPU-gated; the heavy model integration (TRELLIS.2) and the
> view-dependent appearance work (SH, BRDF) are scoped here as the next steps,
> several of which are Box-A (GPU) activations.

## Why this exists — the root cause (carried from the Tier 0 diagnosis)

The text→3D path is `prompt → SDXL (1 image) → TripoSplat (single-image,
feed-forward, 262k, **SH degree 0**) → "L3 refine"`. The "L3 refine" in
`generative.py` renders the L2 cloud's **own** output and re-fits a 2DGS surfel
cloud to reproduce it with **frozen positions** (`means_lr_scale = 0.0`) and a
**fixed count**. It is a *distillation*: it can only ever lose information relative
to L2. So the entire quality ceiling is `SDXL image × single-image TripoSplat`,
and there is no stage that adds detail, fixes geometry, or recovers
view-dependent appearance.

Three deficits cap photorealism, and Tier 1/2 each remove one:

| Deficit | Consequence | Fix |
|---|---|---|
| Single-image conditioning | Hallucinated back/sides; identity drift | **Tier 1**: multi-view generator (TRELLIS.2) |
| Distillation, frozen, fixed-count | No detail can be added; geometry locked | **Tier 1**: densified refine vs external multi-view targets |
| SH degree 0; lighting baked in colour | Flat "painted-clay" look; no specular; can't relight | **Tier 2**: SH degrees + per-gaussian BRDF (L4) |

## Verified model landscape (June 2026 — re-checked per CLAUDE.md §10.1)

- **TRELLIS / TRELLIS.2 (Microsoft Research)** — **MIT licensed, full commercial
  use**, native **Gaussian-splat** output via its structured-latent (SLAT) gaussian
  head, ~15–30 s, "cinematic" fidelity; 4B weights public. TRELLIS.2 is the newer,
  "native and compact structured latents" iteration. This is the strongest
  license-clean **splat** generator and is on-thesis for a splats-only product.
- **Hunyuan3D (Tencent)** — broad quality, top open option, but **mesh-oriented**
  (off-thesis for splats-only; licence is a Tencent community licence, must be
  re-read before adoption). Not selected.
- **DreamGaussian / SDS** — score-distillation sampling from a frozen diffusion
  model + progressive densification: the recipe for injecting generative detail
  into the *unseen* regions during refinement. Adopted as the optional guidance
  path on top of the refine loop.
- **Adaptive Density Control** — the Kerbl-2023 clone/split/prune/opacity-reset,
  with 2025 refinements (AbsGS, Improving-ADC-3DGS, DC4GS, metric-driven scoring).
  Implemented here as the densification engine.

### Decision

**D-T1.1 — Generator: adopt TRELLIS.2 as the Tier-1 multi-view front end**
(MIT, native gaussian head). Keep TripoSplat as a fast fallback / A-B baseline; do
not rip it out. Vendoring mirrors the existing `external/TripoSplat` pattern.
Runner-up: a "render L2 + per-view SDXL img2img" pseudo-multi-view path (uses the
already-installed SDXL, but per-view inconsistency limits the gain) — kept as a
no-new-weights fallback behind the same target seam.

**D-T1.2 — Refiner: real densified multi-view optimisation**, not distillation.
Unfrozen positions + ADC + perceptual loss, supervised by external multi-view
targets. Implemented (`refine.py`, opt-in).

**D-T2.1 — Appearance: SH degrees (view-dependent) then per-gaussian BRDF (L4)**.
Staged because SH has a cross-cutting blast radius (see Risks).

## What landed this session (CPU-gated green)

- **`densify.py`** — Adaptive Density Control: `clone_mask` / `split_mask` /
  `prune_mask`, `densify_and_prune` (parent→`split_n` children, jittered + shrunk),
  `reset_opacity`, and a stateful `DensityController` (gradient accumulation +
  densify/opacity-reset schedule). VRAM cap; never empties the cloud. 11 unit tests.
- **`refine.py`** — `refine_with_densification`: unfrozen positions, ADC every
  `interval` iters within `[warmup, stop)`, periodic opacity reset, and
  `L1 + D-SSIM + surface-reg + λ·perceptual` loss. `gradient_loss` (dependency-free
  edge/sharpness proxy; LPIPS plugs into the same `perceptual` callable) and
  `build_optimizer` are CPU-tested; the full gsplat loop has a GPU-guarded test.
- **`generative.run_l2_to_l3`** — opt-in `densify` flag / `ASTEL_L3_REFINE` env and
  an `external_targets` parameter (multi-view images on the train rig). Default path
  unchanged (zero risk). `metrics` now carry `densify`, `external_targets`, and the
  full densify history.

## Verified on Box A (2×4090, this session)

All 163 GPU+CPU tests green via `run-python.cmd`. Measured A/B:

- **Engine proof (synthetic, target carries new info):** random init **8.87 → 31.77
  dB (+22.9)**, count adapted 6000→10448, ADC fired. The densified refine recovers
  detail/geometry decisively when the target has information the init lacks.
- **Real 262k TripoSplat, supervised by L2 SELF-renders:** frozen distillation
  **23.05 dB** vs densified **20.14 dB** — distillation **wins**. Moving positions
  + changing count drifts away from the exact configuration that produced the
  self-render targets, so against self-renders densify *hurts*. Peak VRAM 1.2 GB
  (huge headroom — modest densification was a `grad_threshold` tuning effect, not a
  memory limit).
- **Conclusion (load-bearing):** the densified engine is correct and powerful but
  **starved of new information without external multi-view targets**. Keep
  `ASTEL_L3_REFINE` OFF until step 1 lands; it only beats distillation once a
  stronger generator supervises it. This is the empirical case for TRELLIS.2.

## Remaining steps (ordered)

1. **Vendor TRELLIS.2** under `external/` + `models/`, wrap as `l2_trellis.py`
   mirroring `l2_triposplat.py` (typed, opacity/scale convention fix, SplatCloud
   out, select via `ASTEL_L2_MODEL`). Generate **multi-view-consistent images** and
   feed them as `external_targets` to the refine — **the unlock**: only with these
   does the densified refine exceed L2 (verified that self-renders do not suffice).
   Then tune `grad_threshold`/`percent_dense`/`interval`/`λ_perceptual` and make
   densify the default once it beats distillation on the eval corpus.
2. **SDS guidance** (optional) — add a diffusion-guided term for genuinely unseen
   regions, gated by the confidence channel (never silently hallucinate over
   measured data, CLAUDE.md §1).
3. **Tier 2 — SH degrees**: extend `GaussianParams` (+ `astel_splat_io.SplatCloud`,
   export, `.spz`/`.sog` packers, web viewer) to carry SH up to degree 2–3; let the
   refine optimise SH coefficients (gsplat `rasterization_2dgs` accepts SH coeffs +
   `sh_degree`). Biggest single photorealism win after multi-view; biggest blast
   radius — do it as its own milestone with golden-file export tests.
4. **Tier 2 — per-gaussian BRDF (L4)**: replace the post-hoc DC-colour un-lighting
   with an optimised albedo/roughness/metallic fit against the multi-view targets,
   so relighting is real, not approximate.

## Risks / honesty

- **Densify on self-render targets regresses PSNR** (measured: 23.05 → 20.14 dB on
  the real 262k cloud). It is correct but information-starved without external
  multi-view supervision; do not enable `ASTEL_L3_REFINE` in production until step 1.
  The end-to-end loop itself is GPU-validated (Box A) and the ADC engine has 11
  component unit tests, so the engine is sound — treat it as production-proven only
  once it has real multi-view targets to chase.
- **Per-view SDXL pseudo-multi-view is inconsistent across views** → can blur rather
  than sharpen. The principled fix is a true multi-view-consistent generator
  (TRELLIS.2), which is why D-T1.1 prefers it over the no-new-weights fallback.
- **SH/BRDF touch the archival format and the web viewer** — coordinate the schema
  bump across `astel_splat_io`, exporters, and `apps/web` in one milestone with
  golden-file round-trip tests, or older `.astel` packages break.
- **VRAM** — densification grows the cloud; `max_gaussians` caps it. Cinematic
  budgets need the 32–48 GB recommended tier (CLAUDE.md §6).
