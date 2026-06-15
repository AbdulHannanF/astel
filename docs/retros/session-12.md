# Session 12 retro (2026-06-15)

**M3 step 3a — TripoSplat spike graduated into typed, tested production L2 module;
inf-opacity defect fixed and measured.** No founder gate touched — no Anthropic API
key used, no spend incurred.

Mode: Opus planning/review; one Sonnet agent implemented the module and tests; one
Haiku agent drafted this retro (per the founder's model directive).

## 1. Step 3a — Graduate and fix L2_TripoSplat, PASS

The spike `triposplat_spike.py` from session 11 step 2 graduates into production:
- New module `pipelines/gpu/src/astel_gpu/l2_triposplat.py` (typed, `mypy --strict`
  clean, `ruff` clean): converts vendored TripoSplat `Gaussian` into `astel_splat_io.cloud.SplatCloud`
  and exports via `write_ply` instead of upstream's `Gaussian.save_ply`.
- **The defect fix**: upstream `save_ply` writes opacity as `log(get_opacity/(1-get_opacity))`,
  which saturates to `inf` when sigmoid-activated `get_opacity` hits exactly 1.0 in
  fp16 (~11% of points, session 11). The new module reads activated `get_opacity` (in
  [0,1]), clamps to [1e-6, 1-1e-6], recomputes logit — matching how `astel_gpu.export.to_splat_cloud`
  handles other layers. All other fields (xyz, f_dc, log-scale, rotation) come from
  `_get_ply_data(transform)`, which applies the correct coordinate transform.
- Conversion split into pure GPU-free `splat_cloud_from_fields(...)` plus thin
  `gaussian_to_splat_cloud(...)` adapter — unit-testable on CPU without weights.

## 2. Tests: opacity clamp verified

New CPU tests `pipelines/gpu/tests/test_l2_triposplat_cpu.py`:
- (a) opacity clamp produces finite logits even when activated opacity contains 1.0
  and 0.0.
- (b) adapter test with minimal fake `Gaussian` whose `_get_ply_data` returns
  inf-laden opacity; we ignore that field and yield all-finite opacity.

## 3. Measured results (Box A, 2× RTX 4090)

building_stone_house example, 65,536 gaussians, 20 steps:
- Gates: `ruff` clean; `mypy --strict` clean (27 source files); `pytest` 45 passed
  (43 prior + 2 new), via `run-python.cmd` MSVC launcher.
- Wall-time **11.1 s**, peak VRAM **4.59 GB**.
- **Decisively: n_nonfinite_opacity_logit = 0, n_nonfinite_xyz = 0** — inf-opacity
  defect eliminated, vs spike's ~11% inf via upstream save_ply.
- Wrote `out_triposplat/l2.ply` (3.67 MB) + `l2-metrics.json`.

## 4. Honest gaps / carried forward

- This is the **graduate half** of M3 step 3 only. The **bake-off scoring half**
  (input-view reconstruction PSNR/SSIM/LPIPS for TripoSplat, ± TRELLIS-v1 comparison)
  is NOT done — so DECISIONS #2 (L2 prior choice) is NOT formally resolved, though
  TripoSplat's lead is reinforced.
- `l2_triposplat` is **NOT yet wired** into `astel_gpu.produce` or the API — that is
  M3 step 4 (L2→L3 wiring), gated on the L3 surface A/B.
- **The L3 surface-aligned A/B (M2 carryover) is still open** — 2DGS vs 3DGS+GOF on
  DTU scan1, must beat 8.73 mm overall. Gating next step.
- **Still nothing committed** — sessions 7–12 GPU work remains in working tree on top
  of single "Beta" commit. Awaiting founder go-ahead.
- Generation Spec LLM stage (M3 step 5) deferred pending Anthropic API key + spend
  cap; adapter built on cached fixtures first regardless.

## 5. Next (session 13)

(a) **L3 surface-aligned A/B on DTU scan1** (beat 8.73 mm). (b) **L2→L3 wiring** —
feed `l2_triposplat` gaussians into surface-aligned L3 refine. (c) **Bake-off scoring**
to formally resolve DECISIONS #2.
