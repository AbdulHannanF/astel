# Session 16 retro (2026-06-15)

**M3 integration — part 1: the GPU producer now emits the full `.astel` artifact
contract AND runs the real generative image path end-to-end.** Closed the
session-15 "integration" gap on the GPU/asset side: a GPU generation now produces
the identical layer-stack a stub generation does, and an uploaded image actually
drives TripoSplat L2 → 2DGS L3 through the production API→subprocess seam.

Mode: Opus, inline. On the 2×4090 box (`THREADRIPPER-48`); real CUDA runs.
No founder gate touched (no API key, no spend, nothing committed).

## 1. What shipped

**GPU side (`pipelines/gpu`):**
- New `astel_gpu.packaging` — a **torch-free, CPU-testable** `write_layer_stack`
  seam that takes the refined L3 `SplatCloud` and emits the full contract:
  `l0.ply` (strided seed subsample), `l3.ply`, `l3.spz`, `l3.sog`,
  `package.astel` (via `astel_format.build_minimal_package`, L0+L3 bound with
  per-gaussian provenance), `quality-report.json` — plus `l2.ply` for the
  generative path. Honest typed package report: geometric error `None`+reason
  (no GT), ungrounded identity scale (`ci_method="gpu-no-estimate"`), 0% measured.
- `astel_gpu.produce` rewritten to dispatch by modality: **image + `--image`** →
  real `run_l2_to_l3` (TripoSplat→2DGS) emitting l2+l3+generative report;
  otherwise the render-then-refit **smoke** path. Both converge on
  `write_layer_stack`, so the artifact contract is identical to the stub.
- Added `astel-format` as a GPU dep (pure-python: pydantic/jsonschema; no CUDA).

**API side (`services/api`):**
- `gpu_producer.produce_artifacts_dispatch` gained `capture_id`: for the image
  modality it resolves the uploaded `source*` image from the store and passes
  `--image` to the GPU CLI (the documented local-fs seam; S3 would download
  first). `main.create_generation` threads `body.capture_id` through. The stub
  default path is byte-for-byte unchanged.

## 2. Measured on the box (real CUDA)

- **Smoke/text path** (8k gaussians, 300 iters): full 7-artifact contract, L0=334
  + L3=8000 splats all finite, self-consistency PSNR **41.8 dB**, 2.4 s, 0.17 GB.
- **Generative image path** (`creature_butterfly.webp`, 500 refine iters): L2
  TripoSplat **65,536 gaussians, 11.1 s, 4.59 GB, 0 non-finite** (opacity fix
  holds) → L3 2DGS **65,536 surfels, 8.1 s, 4.93 GB**, held-out self-consistency
  PSNR **18.14 dB** (500 iters; cf. session-14's 23.13 dB at 1500). Full
  8-artifact contract (incl `l2.ply`), all PLYs finite, `package.astel`
  round-trips with honest `chamfer=None`, `measured_fraction=0.0`.
- **REAL API→GPU end-to-end** (no mocking, `ASTEL_PRODUCER=gpu`):
  `produce_artifacts_dispatch` invoked the live `run-python.cmd` subprocess and
  landed all 7 artifacts (8000 splats) in a `LocalArtifactStore`. The production
  seam works, not just the unit-mocked logic.

## 3. Gates

- GPU: ruff ✅ · mypy --strict ✅ (33 files) · pytest **54 CPU + 2 GPU = 56** (5
  new CPU packaging tests). `astel_format.*` added to the mypy override list
  (no py.typed marker, matching the existing `astel_splat_io` treatment).
- API: ruff ✅ · mypy --strict ✅ (19 files) · pytest **30 passed + 1 skipped**
  (2 new gpu_producer tests: image-modality threads `--image`; no-capture omits it).

## 4. Honest gaps / carried forward

- **Generation Spec LLM stage is still NOT wired into the API text path.**
  `astel_llm` is torch-free and belongs in the API env; session 17 wires
  `build_generation_spec` into the text modality (offline `FixtureAdapter` with
  graceful degrade on cache miss; stores `generation-spec.json`; threads
  `target_scale` into the report's scale field). Still gated on the founder's
  Anthropic key (R-O2) for arbitrary prompts.
- **Text modality has no prompt conditioning yet** — it runs the self-consistency
  smoke (honestly labelled). Real text→multiview→L2 is a generative-model stage
  not built (needs a current multi-view diffusion checkpoint; survey is M3 research,
  build is post-integration).
- Generative `geometric_error`/`scale` are honestly `None` (generated objects have
  no GT scan / metric grounding). Self-consistency PSNR is the only number, flagged.
- The web Truth Meter / Layer Inspector haven't been re-pointed at a real GPU
  generation in-browser yet (the contract matches the stub's, so it *should* just
  work — unverified live this session).
- **Still nothing committed** — sessions 7–16 remain in the working tree on the
  single "Beta" commit. Flagged again; awaiting founder go-ahead.

## 5. Next

(a) **Session 17 — wire `astel_llm.build_generation_spec` into the API text
path** (offline fixtures, graceful degrade, store the spec, thread scale into the
quality report). (b) **Founder**: Anthropic API key + spend cap to light up live
spec calls (R-O2). (c) Then **M4** — L4 relighting, L5 collision/SDF + print path,
L6 physics-material LLM pass (reusing the `astel_llm` adapter).
