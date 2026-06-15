# Session 13 retro (2026-06-15)

**L3 surface-aligned A/B RESOLVED — 2DGS beats raw 3DGS on real DTU geometry.**
The L3 representation decision (DECISIONS #1), open and 🟡 since 2026-06-13, is now
✅ on a measured basis. No founder gate touched — no Anthropic API key, no spend.

Mode: Opus, inline (no subagents this session). All work on Box A (2× RTX 4090).

## 1. What shipped

- **`pipelines/gpu/src/astel_gpu/l3_refine.py`** (new, typed, `mypy --strict` + `ruff`
  clean): gsplat-native 2DGS refine.
  - `render_2dgs_colors` — RGB-only surfel render for eval/PSNR.
  - `render_2dgs_train` — `render_mode="RGB+ED"` render returning colors + rendered
    normals + depth-derived `surf_normals` + distortion map (surf_normals/distort
    require depth rendering in gsplat — discovered the hard way).
  - `surface_reg_loss` — pure, CPU-testable: `λn·(1 - <render_n, surf_n>) + λd·distort`.
  - `optimize_2dgs` — mirrors `smoke_refit.optimize` (same Adam param groups, same
    `spatial_lr_scale` metric-coords trick, same soft-clamps) + the surface losses.
- **`capture_eval`** gained `--representation {3dgs,2dgs}` + `--lambda-normal/--lambda-dist`,
  so both arms share an IDENTICAL init cloud, DTU ObsMask/Plane protocol, and held-out
  PSNR split. `representation`/lambdas recorded in metrics + quality report.
- **4 new CPU tests** (`test_l3_refine_cpu.py`) for the pure loss seam. Suite **49 passed**
  (45 prior + 4). Gates green: `ruff`, `mypy --strict` (29 files), `pytest`.

## 2. Two gsplat-1.5.3 quirks found & fixed

1. `rasterization_2dgs` does NOT auto-expand per-gaussian colors to per-camera
   `[C, N, D]` when `sh_degree is None` (the expansion is commented out in
   rendering.py, unlike `rasterization`). Fixed: we broadcast colors to `[C, N, 3]`
   ourselves (`_colors_per_cam`).
2. `surf_normals` and `distort` are `None` unless `render_mode ∈ {ED, RGB+ED}`, and
   `distloss=True` *requires* a depth render mode. Hence the split colors-only vs
   training render functions.

(Also: cmd.exe `%*` in `run-python.cmd` mangles multi-line `-c` strings — use a temp
`.py` file or single-line `;`-joined `-c` for ad-hoc GPU checks.)

## 3. Measured A/B (Box A, 200k gaussians, 3000 iters, no densification, seed 20260614)

| Arm | overall mm | accuracy mm | completeness mm | held-out PSNR |
|---|---|---|---|---|
| 3DGS (raw baseline) | 8.76 | 11.52 | 6.00 | **21.41** |
| 2DGS λn=0.05, λd=0 | 9.48 | 13.06 | 5.90 | 20.59 |
| 2DGS λn=0.05, λd=1.0 | 27.11 | 30.24 | 23.99 | 8.12 |
| **2DGS λn=0.05, λd=1e-4** | **8.53** | **10.91** | 6.15 | 20.47 |
| 2DGS λn=0.05, λd=3e-4 | 9.07 | 11.91 | 6.23 | 20.21 |

Fresh 3DGS (8.76) reproduces the session-10 baseline (8.73) → the comparison is fair.
**Normal consistency alone slightly hurts geometry; a scale-appropriate distortion term
flips it** — 2DGS then beats 3DGS on overall + accuracy AND emits real surfel normals
(needed for L4 BRDF / L5 SDF), for ~1 dB less PSNR. The right trade for a
geometry-accurate splat product. **Decision: L3 = 2DGS + normal + distortion. GOF
runner-up not needed, stays unimplemented.**

## 4. Honest gaps / carried forward

- **λdist is scene-scale-dependent** (1.0 → 27 mm collapse at ~600 mm depths; 1e-4
  optimal). λdist=1e-4 is DTU-scan1-specific. A dimensionless scale-normalized λdist is
  future work before the 2DGS default generalizes across scenes.
- Both arms ran WITHOUT densification (MCMC/ADC) — this isolates representation +
  regularization, not the fully-productionized L3 (DECISIONS #19's PGSR/MCMC stack).
- Single scan (scan1). A multi-scan DTU corpus number is still owed (M2 carryover).
- **Still nothing committed** — sessions 7–13 GPU work remains in the working tree on
  the single "Beta" commit. Flagged again; awaiting founder go-ahead.

## 5. Next (session 14)

(a) **L2→L3 wiring** (M3 step 4): image → TripoSplat L2 → 2DGS L3 surfelization via
self-rendered views (generated objects have no GT scan, so refine against the L2's own
multi-view renders; honest self-consistency report). (b) Formally **close DECISIONS #2**
(TripoSplat as L2 prior — evidence in hand; the multi-model PSNR bake-off needs a
multi-view generative corpus that doesn't exist yet, recorded as deferred). (c) Then the
**Generation Spec LLM stage** scaffolded on fixtures (M3 step 5), founder adds API key last.
