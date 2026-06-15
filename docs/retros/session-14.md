# Session 14 retro (2026-06-15)

**Generative L2→L3 wired end-to-end; DECISIONS #2 resolved (L2 = TripoSplat); R-T1
retired.** M3 step 4 done. No founder gate touched — no Anthropic API key, no spend.

Mode: Opus, inline (no subagents). All GPU work on Box A (2× RTX 4090).

## 1. What shipped

- **`pipelines/gpu/src/astel_gpu/generative.py`** (new, typed, gates green): the full
  generative path, image → splat L3, end-to-end.
  - `run_l2_to_l3`: TripoSplat L2 (native gaussians) → `normalize_params` (unit frame) →
    render an orbit of synthetic views as the distillation target → `optimize_2dgs`
    (the session-13 2DGS L3) → surface-aligned surfel L3 with real normals.
  - `normalize_params` — pure CPU seam (center on centroid, scale to unit radius;
    means+scales transformed, colors/opacity/quats invariant).
  - `build_generative_quality_report` — honest report for a GENERATED asset.
- **`export.gaussian_params_from_splat_cloud`** — inverse of `to_splat_cloud`
  (SplatCloud → training-space params), so a generated L2 cloud can feed the L3 refiner.
- **2 new CPU tests** (`test_generative_cpu.py`); suite **51 passed** (49 prior + 2).
  Gates green: `ruff`, `mypy --strict` (31 files), `pytest`.

## 2. Why distillation (the honest design)

A single-image generator produces an object never photographed from all angles — there
is no GT to refine against. So the L3 is **distilled from the L2 generator's own orbit
renders**: render L2 (3DGS) from 24 views, split train/held-out, refine a 2DGS surfel
cloud to reproduce them. The reported number is held-out **self-consistency /
distillation fidelity**, NOT accuracy vs reality. The quality report keeps
`geometric_error` and `scale` honestly `None` and `generated_ratio = 1.0` — the
confidence channel never implies measured reality.

## 3. Measured (Box A, building_stone_house, 65,536 gaussians, 24 views, 1500 iters)

- L2 65,536 gaussians → L3 65,536 surfels.
- **Held-out self-consistency PSNR 23.13 dB**; refine 20.3 s; peak VRAM 4.93 GB.
- L2/L3 PLYs both fully finite (xyz/opacity/scale). L3 carries real surfel normals.

## 4. Decisions closed

- **DECISIONS #2 → L2 prior = TripoSplat** (MIT code+weights, 4.6–4.9 GB / ~11 s, cleanest
  deps, native gaussians, published Elo > TRELLIS.2, now proven end-to-end). The deferred
  piece — a multi-model PSNR/SSIM/LPIPS head-to-head vs the TRELLIS-v1 head — needs a
  multi-view generative corpus (none exists) + a TRELLIS-v1 install (cu128 wheel risk).
  Committed now on the evidence in hand per §10.2.
- **R-T1 RETIRED**: the TRELLIS.2-mesh→surfel distillation (the single riskiest bet) is
  off the critical path. TRELLIS.2 stays only as an optional future fidelity upgrade.

## 5. Honest gaps / carried forward

- Distillation: no densification, 1500 iters → 23 dB is good, not hero-tier.
- Generative L3 uses normal-only reg (λdist=0; its scale-tuning is per-scene, session 13).
- **Not yet wired into the API `produce` path or `.astel` packaging** — the
  `astel_gpu.generative` CLI exists; the API still runs the stub/smoke producer. That
  integration (plus `.astel` package with l2+l3+report) is the next step.
- Single test image; no multi-asset generative corpus yet.
- **Still nothing committed** — sessions 7–14 GPU work remains in the working tree on the
  single "Beta" commit. Flagged again; awaiting founder go-ahead.

## 6. Next (session 15)

**Generation Spec LLM stage** (M3 step 5), scaffolded on cached fixtures: model-agnostic
adapter + structured prompt→spec JSON (Haiku 4.5 default, structured outputs + prompt
caching), fixture/mock backend with cached-fixture tests, token-cost logging hook. **No
paid call until the founder provides an Anthropic API key + spend cap** — the adapter and
all tests run offline first, the founder wires the key + a live test at the very end.
