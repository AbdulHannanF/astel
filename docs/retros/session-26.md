# Session 26 retro (2026-06-18)

**M4 closed for real — photorealism fixed, two latent gaps made live, one hard
hang bounded.** The task: re-audit M4 for anything *unplugged / fake / simulated*,
make it real, and specifically verify whether a **photorealistic** splat now
generates instead of the blurry one seen before — all before M5. Opus end-to-end
(audited, implemented, measured on the GPU box, gates re-run). The headline:
**the generator was always capable of photorealism; our pipeline was throwing it
away.** Three real defects fixed, all measured.

## 0. Verification first (never trust a summary)

Re-ran every gate sessions 23–25 claimed, on disk, before changing anything:
astel_appearance **25**, astel_format **28**, astel_solid **37**, pipelines/gpu
**87**+3skip, services/api **67**+1skip, @astel/manifest **15**, apps/web
**tsc -b**·eslint·**43** + production build. All real — the prior retros were
honest. The M4 world-awareness data (L4/L5/L6) and the studios (Relight, Physics,
Truth Meter) are genuine, honestly-caveated implementations, not fakes.

## 1. Photorealism — root-caused and fixed (the headline)

**Diagnosis (measured on Box A, creature_butterfly).** The generative path
(`image → TripoSplat L2 → 2DGS L3`) shipped a **blurry, low-detail** asset for two
compounding reasons:

1. **L2 starved at 1/4 its budget.** `run_l2_to_l3` hard-coded
   `num_gaussians=65536`, but TripoSplat's native max is **262144**
   (`_NUM_GAUSSIANS_MAX`; its own examples use 262144). 65k is below even the
   CLAUDE.md §3 "lowpoly-splat" (100k) tier. Rendered side-by-side, the **262k L2
   is genuinely photorealistic** (wing veins, body hairs, iridescence); the 65k L2
   is a soft blob.
2. **The L3 distillation *degraded* the L2.** `optimize_2dgs` ran 1500 Adam iters
   with a full-rate position LR (5e-3) against **256px** self-renders. Positions
   drifted off-surface into **floaters** (visible speckle) that both lost the L2's
   crispness and inflated the cloud's bounding radius. The pipeline shipped the
   *worse* of L2 and L3.

**Fix.**
- L2 budget → **262144** (`DEFAULT_NUM_GAUSSIANS`).
- Distillation supervision **256 → 512px** (`DEFAULT_IMAGE_SIZE`): at 256px a 262k
  cloud is starved of detail to fit.
- L3 refine is now a **surfelization, not a re-fit**: new `means_lr_scale`
  (default **0.0**) freezes the already-excellent L2 positions while
  scales/opacity/colour/quats adapt to flatten into 2DGS surfels (real normals for
  L4/L5 preserved). Iters **1500 → 600** (the init is already good).

**Measured.** The `produce` path now ships a **262,144-splat** L3 (4× prior),
refine ~34s, self-consistency 21.9 dB (vs 19.4 at full-drift), and the floaters
are gone. Visual proof: input vs **OLD 65k (blurry)** vs **NEW 262k (photoreal)**
side-by-side; confirmed on a second object (pirate ship — planking/sails/rigging
resolved). `produce.py` / `gpu_producer` defaults updated so the **API path ships
this automatically** (`refine_iters` default was overriding to 1500 — aligned).

**New tool (graduated).** `astel_gpu.render_preview` — load a `.ply`, normalise,
render a turntable + contact-sheet montage with the 3DGS rasterizer (matches the
web viewer). Pure camera seam CPU-tested; this is the "is it photorealistic?"
QA/thumbnail utility.

**Honest ceiling:** L3 quality now tracks the **TripoSplat L2 generator** at 262k.
Beating that needs a stronger generator / multi-view diffusion / true densification
beyond the generator's output — future work, not this fix.

## 2. L6 binding was latent in production — now live (the "unplugged" finding)

The session-25 articulation fix (hinge→revolute + region indices) and the L6↔L5
mass join **only ever ran in tests**: the API physics-material stage wrote
`l6.json` to the store **after** the producer had already packaged, so
`write_layer_stack` never saw it. Every shipped `.astel` carried **no L6 layer**
and **no `l6-mass.json`**.

**Fix.** The physics-material stage reasons over the **Generation Spec**, not the
produced asset, so it now runs **before** produce. The resulting `l6.json` path is
threaded (`produce_artifacts_dispatch(..., l6_json_path=)` → GPU CLI `--l6-json`),
staged into the producer's out-dir (`_stage_l6_json`), and bound by
`write_layer_stack`. Billing-neutral (it prices delivered artifacts, and a refine
keyed on a preview still skips it). **Proven end-to-end:** a full produce run with
a staged `l6.json` now yields a package binding
`l6: physics_material articulation=[('revolute', 0, 1)]` (hinge→revolute + region
indices body=0/wing=1) **and** `l6-mass.json` (2 regions, 9.76 kg, honest
"not-segmented" + "scale ungrounded" caveats). This was impossible before today.

## 3. CoACD packaging hang — bounded (the "not working" finding)

The L5 convex-decomposition (`astel_solid.convex_decompose`) ran CoACD with
default params and **no timeout**; on a detailed/thin watertight mesh its MCTS ran
**> 30 CPU-minutes without terminating** — and the GPU producer invokes it via a
`subprocess.run` with no timeout, so a real generation **hung in packaging
forever** (I hit this on the very first baseline run this session).

**Diagnosis (probe).** CoACD voxel-remeshes the input to a manifold at
`preprocess_resolution` (default 50 → ~286k working triangles), then runs MCTS at
~30 s/iteration over many iterations. The **input** face count is irrelevant — and
a vertex-cluster decimation I first tried *backfired* (welding made the mesh
non-manifold, forcing an even finer remesh).

**Fix.** CoACD now runs in a **spawned subprocess with a 45 s wall-clock cap**
(the only way to interrupt a C++ extension); on timeout/err it's terminated and we
fall back to a single **scipy convex hull**, with the path recorded honestly in
`ConvexSet.method`. Collision-grade fast params (`preprocess_resolution=30`,
`resolution=512`, `threshold=0.1`, modest MCTS). **Measured:** a convex-friendly
L-shape → `method=coacd`, 2 hulls, **7.8 s**; pathological thin-featured meshes
(insect wings, ship rigging) → bounded scipy fallback. The producer can no longer
hang.

## 4. Honesty fix

`gpu_producer._gpu_conditioning` still claimed "today's text path runs the
prompt-independent smoke-refit" — false since session 22 (text → SDXL/FLUX →
TripoSplat → 2DGS is wired). Corrected the docstring.

## 5. Gates — all green (Opus-run)

- `astel_appearance`: ruff · mypy --strict (13) · **25**
- `astel_format`: ruff · mypy · **28**
- `astel_solid`: ruff · mypy (9) · **37** (CoACD now subprocess-bounded)
- `pipelines/gpu`: ruff · mypy (40 incl. tests) · **94**+3skip (+7: render_preview 4,
  produce-l6-staging 3)
- `services/api`: ruff · mypy --strict (26) · **71**+1skip (+4: l6-json dispatch 2,
  `_l6_json_artifact_path` 2)
- `@astel/manifest`: typecheck · eslint · **15**
- `apps/web`: **tsc -b** · eslint · **43** + production build

## 6. Honest gaps / next

- **CoACD falls back to a single scipy hull for thin-featured objects** (bounded,
  honest `method` label). Multi-hull for those needs a real mesh decimator
  (`fast_simplification`/open3d — a new dep) or a coarser collision isosurface.
- **L6 only flows when a Generation Spec + physics fixture/key exist** (text path,
  founder gate R-O2); the offline default still ships no L6, honestly.
- Per-region volume is still **not-segmented** (mean-density mass, flagged).
- Photorealism tracks the **TripoSplat L2 ceiling** at 262k; image-modality scale
  stays ungrounded (no VLM size estimate on that path).
- No live-browser screenshot (no Playwright harness) — studios covered by unit
  tests + clean production build.

**M4 is closed** (world-awareness real + bound, generation photorealistic, no
hangs). **Next: M5 pipeline-readiness** — Unity/UE5 plugins (the direct consumer
of the now-live L5 collision + L6 mass/material/articulation), KHR_gaussian_splatting
glTF export, SDK + MCP — or the text→multiview bridge (mission modality #1).
