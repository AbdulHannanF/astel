# Session 10 retro (2026-06-14)

**The M2 capture-path gaps from session 9 are closed.** The SfM front-end is
validated to sub-millimetre on real images, and the geometry number is now
DTU-protocol-correct (ObsMask/Plane) with held-out PSNR. The capture path has a
real, defensible baseline; the surface-aligned L3 is the next move.

Mode: Opus throughout (plan/decide/implement/verify) — the work was tightly
coupled (protocol fidelity + numerical correctness), so it stayed on one agent.

## 1. COLMAP SfM front-end validated (closes the session-8 deferral)

Ran the session-9 `colmap_runner` on the 49 real DTU scan1 images: GPU SIFT →
exhaustive match → mapper → undistort (~55 s) → **49/49 registered**, 26,921
sparse points. Then `capture_sfm` aligns COLMAP's scale-free camera centres to
DTU's metric GT centres via **Umeyama** similarity and reports the residual:
**pose RMSE 0.886 mm** (median 0.76, max 1.54) over all 49 cameras (~608 mm from
the object → ~0.15% error). The SfM front-end recovers the real rig to
sub-millimetre. This is the *pose-from-images* validation; `capture_eval` uses
DTU's GT poses to isolate splat geometry from pose error — complementary.

## 2. DTU-protocol geometry + held-out PSNR (replaces the box proxy)

Rewrote `capture_eval` to follow DTU's own eval (`PointCompareMain.m`), read
straight from the Matlab source rather than guessed:
- **ObsMask** (`.mat`, 1 mm voxels) masks which fitted gaussians count for
  accuracy; **Plane** (`.mat`) + ObsMask define the observable object volume for
  completeness; per-point distances **capped at 60 mm**.
- **Held-out PSNR**: fit on 42 train views, measure on 7 unseen test views.
- One documented deviation: we intersect ObsMask with the plane for
  completeness (DTU's leaderboard uses above-plane only) because Astel
  reconstructs the *object*, not the full scene — scoring whole-scene coverage
  would unfairly penalise object-only modelling.

**Measured (scan1, 200k gaussians, 3000 it, ~168 s):** held-out PSNR **21.5 dB**;
**accuracy 11.36 mm, completeness 6.10 mm, overall 8.73 mm** vs the real scan.

## 3. What the numbers say

- Accuracy fell from the box-proxy's 18.9 mm to **11.36 mm** once the ObsMask
  excluded out-of-volume floaters — the box proxy was over-counting background.
- 11.36 mm accuracy is still high: raw 3DGS (no densification, no surface
  regularization) leaves floaters even inside the object volume. This is the
  concrete real-world baseline the surface-aligned L3 (2DGS/PGSR) must beat —
  the same story as the synthetic + session-9 results, now protocol-correct.
- Held-out 21.5 dB (vs 23.3 train) is the honest generalization figure;
  background-capped because object-only gaussians don't model the background.

## 4. New code (all gates green: ruff · mypy 24 files · 43 pytest)

- `capture_sfm.py` — COLMAP↔DTU pose accuracy via Umeyama.
- `colmap_runner.main()` — CLI for the SfM pipeline.
- `dtu.py` — `umeyama`, `load_obsmask`/`load_plane`, `points_in_obsmask`,
  `points_above_plane` (all CPU-unit-tested: Umeyama round-trip, mask logic).
- `metrics.nn_distances` — public per-query NN (the protocol filters/caps before
  averaging, so it needs the raw distances, not just the Chamfer mean).
- `capture_eval` rewritten (ObsMask protocol + held-out split); `scipy` added.

## 5. Honest gaps / carried forward

- **The L3 A/B is now the headline next step** — a surface-aligned fit (2DGS or
  3DGS+GOF) on this exact scan must beat 8.73 mm overall / 11.36 mm accuracy.
- PSNR is background-capped (object-only fit); full-scene modelling would lift it
  but isn't needed for the object geometry number.
- One DTU scan so far; a multi-scan corpus number would harden the baseline.
- **Still nothing committed** — sessions 7–10 GPU work is all in the working tree
  on top of the single "Beta" commit. Flagged repeatedly; awaiting go-ahead.
- The stale "Download COLMAP 4.0.4" background task (5 h, zombie shell, no active
  process, redundant — we run 4.1) should be cleared from the UI.

## 6. Next (session 11)

(a) **L3 surface-aligned A/B** on DTU scan1 (2DGS surfels vs 3DGS+GOF) — beat the
8.73 mm baseline; this finally resolves the long-open L3 representation decision
on real content. (b) A few more DTU scans for a corpus number. (c) Then M3: the
TRELLIS import-graph/license check + R-T1 distillation de-risk (generative path),
which needs an Anthropic API key for the Generation Spec stage — cost estimate
first.
