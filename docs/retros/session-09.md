# Session 9 retro (2026-06-14)

**The Truth Meter has its first REAL-WORLD geometry number.** A gaussian fit to
real DTU photographs, measured against a real structured-light scan, covers the
object's surface to **≈ 3.85 mm** (completeness) at metric scale. The synthetic
baseline is now backed by a real-world one — the core M2 deliverable.

Mode (per founder directive — Opus plans/decides/verifies, Sonnet implements,
Haiku small tasks):
- **Opus**: planning, the public-dataset pivot, all the pipeline code
  (`colmap_io`, `colmap_runner`, `dtu`, `capture_eval`, chunked Chamfer), every
  GPU run, the diagnosis of two real bugs (§3), and all docs/verification.
- **Sonnet ×1** (background research-scout): surveyed public capture datasets
  with licenses + verified download URLs; recommended DTU. Output drove the
  dataset decision (DECISIONS.md session-9 §).

## 1. The pivot: public datasets instead of waiting on filmed video

The M2 real-world numbers were gated on the founder filming the CORPUS.md
C01–C10 orbit videos. Founder directive: source real capture data from the
internet instead. The Sonnet scout found **DTU MVS** is the only license-clean
option shipping real structured-light GT geometry **in metric mm** *plus* a
single-scene download. T&T / CO3D are non-commercial (rejected for a commercial
venture); Mip-NeRF 360 has no GT scan. DTU's license is unstated → adopted
**internal-benchmark-only** (no redistribution, no shipped-model training, no
product/marketing claims without clearance). All recorded in DECISIONS.md.

DTU's `pos_NNN.txt` projection matrices are in the SAME frame as the GT scan, so
fitting with them lands the cloud directly in the GT metric frame — **the first
real Chamfer needs no registration** (the clean path). Honest limitation: this
uses DTU's lab-calibrated poses, so it validates splat-fitting geometry, NOT the
casual-phone / pose-free / scale-from-monocular-depth story (still founder-gated
or a pose-free public set).

## 2. What landed (all measured on Box A, RTX 4090; all gates green)

New, license-clean, typed, tested modules in `pipelines/gpu`:
- `colmap_io.py` — COLMAP binary model reader (cameras/images/points3D → K,
  world→cam viewmats, L0 cloud). Convention already matches gsplat.
- `colmap_runner.py` — SfM pipeline driver, **pinned to the verified COLMAP 4.1
  flag surface** (e.g. `--FeatureExtraction.use_gpu`, renamed from the old
  `SiftExtraction`). Built + unit-tested; not yet run on real images (§5).
- `dtu.py` — pose parse, **RQ projection decomposition** (round-trip tested incl.
  the real pos_001 matrix), GT-free object-centre via camera-axis convergence,
  binary/ASCII PLY reader, image+pose scan loader.
- `capture_eval.py` — the DTU eval: load scan → GT-free init → fit → masked,
  opacity-filtered Chamfer vs GT → honest `astel.quality-report/v0`.
- `metrics.chamfer_distance_chunked` — VRAM-safe Chamfer (the GT is **2.88M
  points**; naive `cdist` would need a 2.88M×N matrix).
- Generalized `RenderInputs` for non-square images (DTU 1600×1200).

**The number (scan1, 200k gaussians, 3000 it, 49 views @ 400×300, ~4 min):**
train PSNR 5.6→**23.3 dB**; **completeness (GT→fit) ≈ 3.85 mm**, accuracy
(fit→GT) ≈ 18.9 mm. Completeness is the trustworthy surface-coverage number;
accuracy is inflated by unmasked turntable/background in the GT + floaters
(documented). Good-completeness / poor-accuracy asymmetry = the same story as the
synthetic baseline, now on real data → motivates surface-aligned L3 + masking.

CI: `pipelines/gpu` ruff · mypy (23 files) · **38 pytest** (CPU lane 27 + full
suite via launcher 38); CPU `api` stack untouched.

## 3. The two bugs Opus caught (why verification matters)

1. **First run gave PSNR 6 dB — no convergence.** Before assuming a pose bug, a
   CPU diagnostic confirmed the poses were CORRECT (object centre projects to
   ~(823, 620), dead-centre in all 49 views; cameras a consistent 608 mm out).
   Root cause: **scene-scale / learning-rate mismatch** — `optimize()`'s
   `lr=5e-3` is right for the synthetic ~unit-scale scene, but DTU is in **mm**
   (object ~100 mm), so steps were ~1000× too small. Fix: 3DGS-style
   `spatial_lr_scale` (per-param-group lr; default 1.0 leaves synthetic/smoke
   unchanged). 6 → 23 dB.
2. **`gt_longest_axis` was 492 mm — the whole scanned scene, not the object.**
   The DTU `stl_total` cloud includes the turntable/background; the object is the
   dense central blob (~1.75M of 2.88M points within ±100 mm). The naive
   full-GT-bbox made the scale metric a circular artifact (both clouds clipped to
   the same box → fake 0.0002 "error"). Fixed to a box-around-centre proxy
   applied symmetrically, and scale reported honestly as *inherited, not
   estimated* (`relative_error: null`).

## 4. Honest gaps / carried forward

- **`accuracy` is background-inflated.** The proper fix is DTU's voxel **ObsMask**
  (`.mat`, needs scipy) to isolate the object, plus background masking / full-scene
  modelling so PSNR isn't background-capped. Box proxy is the documented stand-in.
- **PSNR is TRAIN** (fit on all views), not held-out.
- **COLMAP SfM not yet run on real images** — `colmap_runner` is built + tested,
  but the first number used DTU's metric poses (Path 2). Running COLMAP on the 49
  DTU images (pose-from-images + pose-accuracy vs DTU GT) is the immediate next
  step and finally closes the functional-SfM smoke deferred since session 8.
- **Nothing committed.** Sessions 7–9 GPU work is all in the working tree (single
  "Beta" commit + untracked). Flagged to the founder; awaiting the go-ahead to
  organize clean commits.

## 5. Next (session 10)

(a) Run `colmap_runner` on scan1 → registered-pose count + pose error vs DTU GT
(the SfM front-end validation). (b) Parse the DTU ObsMask (+scipy) for a
protocol-correct accuracy number. (c) Then the L3 2DGS-vs-3DGS+GOF A/B on this
real scan — the surface-regularized fit must beat the 18.9 mm accuracy baseline.
(d) TRELLIS import-graph check + R-T1 distillation for the generative path.
