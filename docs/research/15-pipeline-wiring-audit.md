# 15 — Pipeline Wiring Audit (post-mortem on the silent text→smoke fallback)

**Date:** 2026-06-15
**Trigger:** The text modality silently falls back to the GPU "render-then-refit
smoke" (a procedurally generated cloud unrelated to the prompt) instead of any
real text→geometry pipeline. This was disclosed only as a caveat buried in a
retro, not surfaced structurally in the API response or the package's typed
quality report. This document is a complete wiring matrix so no equivalent gap
can hide again.

Scope: `services/api/src/astel_api/{main,producer,gpu_producer,
generation_spec_stage,physics_material_stage,billing,schemas,storage,engine}.py`,
`pipelines/gpu/src/astel_gpu/{produce,generative,l2_triposplat,l3_refine,
packaging,export,smoke_refit}.py`, `libs/astel_llm`, `libs/astel_solid`,
`libs/astel_format`, `libs/astel_splat_io`, `apps/web/src/{components/
GenerationDock.tsx,App.tsx}`.

---

## 1. WIRING MATRIX

Legend: ✅ real (does what the layer name says, conditioned on the real input) ·
🟡 honest-stub/placeholder (explicitly flagged as fake/unconditioned) ·
❌ unplugged/missing (no code path produces this layer for this cell at all).

Two producer paths exist, selected by `ASTEL_PRODUCER` env var
(`services/api/src/astel_api/gpu_producer.py:139-141`):
- **CPU-stub** = `astel_api.producer.produce_artifacts` (default, no GPU)
- **GPU** = `astel_gpu.produce` via subprocess (`ASTEL_PRODUCER=gpu`)

| Stage / Layer | Text · CPU-stub | Text · GPU | Image · CPU-stub | Image · GPU | Video · CPU-stub | Video · GPU |
|---|---|---|---|---|---|---|
| **L0 Seed cloud** | 🟡 strided subsample of procedural torus-knot — `services/api/src/astel_api/producer.py:188-200`, `producer.py:74-185` (synth, prompt-independent) | 🟡 strided subsample of smoke-refit cloud (also prompt-independent) — `pipelines/gpu/src/astel_gpu/packaging.py:61-67`, `produce.py:159-205` | 🟡 same procedural torus-knot, image ignored — `producer.py:188-200` (stub never reads `capture_id`) | ✅ subsample of TripoSplat L2 cloud — `packaging.py:61-67` fed by `produce.py:112-156` → `generative.py:146-243` | 🟡 same procedural torus-knot, video ignored | 🟡 **same smoke-refit path as text** — `produce.py:223-227`: `modality=="image" and image is not None` is the ONLY branch off the smoke default; video always hits `_produce_smoke` (`produce.py:159-205`) |
| **L1 Dense cloud** | ❌ no `l1.ply` ever written; `_ARTIFACT_LAYER` maps `"l1.ply"` (`billing.py:76`) but no producer emits it | ❌ same — `packaging.write_layer_stack` (`packaging.py:167-249`) never writes `l1.ply` | ❌ none | ❌ none — generative path stops at L2→L3, no L1 | ❌ none | ❌ none |
| **L2 Coarse gaussians** | ❌ no `l2.ply`; stub goes straight to L3 (`producer.py:294-369`) | ❌ smoke path has no L2 concept (`_produce_smoke`, `produce.py:159-205`) — only `_produce_generative` writes `l2.ply` | ❌ none | ✅ `l2.ply` = raw TripoSplat output — `generative.py:125-126,135-145`, written by `packaging.write_layer_stack(..., l2_cloud=...)` (`packaging.py:202-204`) | ❌ none | ❌ none |
| **L3 Refined surface gaussians** | 🟡 procedural torus-knot written as `l3.ply` (prompt-independent) — `producer.py:294-334`, report says so explicitly (`producer.py:228-238`) | 🟡 smoke-refit cloud (real gsplat optimization, but target is a synthetic torus-knot, not the prompt) — `produce.py:159-205`, `smoke_refit.py` | 🟡 same procedural torus-knot, capture ignored | ✅ 2DGS-refined L3 distilled from TripoSplat L2 — `generative.py:146-243` (`optimize_2dgs`, `l3_refine.py`) | 🟡 procedural torus-knot, video ignored | 🟡 **smoke-refit cloud — identical code path as text**, capture never consumed |
| **L4 Appearance/lighting** | ❌ no L4 anywhere in either pipeline. `_ARTIFACT_LAYER` has no `l4.ply` entry beyond the placeholder line `billing.py:82` (`"l4.ply": "L4"`), but nothing ever writes `l4.ply` |||||| 
| **L5 Collision/solidity** | ❌ stub producer never calls `astel_solid` — `producer.py` has no solidify step | ✅ best-effort SDF→watertight→`.stl`+mass via `_try_solidify` — `packaging.py:120-165`, called from `write_layer_stack` (`packaging.py:210-215`) for ALL GPU paths (smoke + generative) | ❌ same as text stub | ✅ same as text·GPU (shared `write_layer_stack`) | ❌ same as text stub | ✅ same as text·GPU (shared writer; runs on the smoke cloud) |
| **L6 Physics-material LLM** | 🟡/✅ `run_physics_material_stage` — real LLM call when `ASTEL_LLM_LIVE=1` + API key present, else honest "skipped" `physics-material.json` — `services/api/src/astel_api/physics_material_stage.py:43-99`, gated `main.py:309-311` (text only, `modality != "text"` short-circuits at `physics_material_stage.py:56`) | same — L6 stage runs in the API layer regardless of `ASTEL_PRODUCER`, so text·GPU also gets it (it's a separate API-side stage, not part of `astel_gpu.produce`) | ❌ `modality != "text"` guard (`physics_material_stage.py:56`) — image never gets L6 | ❌ same guard | ❌ same guard | ❌ same guard |
| **L7 Dynamics** | ❌ nowhere. No `4dgs`/dynamics module exists in `pipelines/gpu/src/astel_gpu/` (confirmed via directory listing) |||||| 
| **Generation-Spec LLM stage** | 🟡/✅ `run_generation_spec_stage` — real when live-gated, else honest "skipped" `generation-spec.json` — `generation_spec_stage.py:50-97`, gated `main.py:299-304`, `modality != "text"` short-circuit at `generation_spec_stage.py:62` | same (API-side, independent of `ASTEL_PRODUCER`) | ❌ `modality != "text"` guard | ❌ same | ❌ same | ❌ same |
| **Export .ply** | ✅ real INRIA-layout PLY (`l0.ply`,`l3.ply`) via `astel_splat_io.ply.write_ply` — `producer.py:312-320` | ✅ same, via `packaging.py:194-204` | ✅ same | ✅ same + `l2.ply` | ✅ same | ✅ same |
| **Export .spz** | ✅ real, spec-conformant — `producer.py:326-329`, `astel_splat_io.spz.write_spz` | ✅ `packaging.py:219-220` | ✅ | ✅ | ✅ | ✅ |
| **Export .sog** | ✅ real but best-effort quantization (documented codebook caveat in `astel_splat_io.sog`) — `producer.py:331-334` | ✅ `packaging.py:221-222` | ✅ | ✅ | ✅ | ✅ |
| **Export .astel package** | ✅ schema-valid, `build_minimal_package` — `producer.py:340-359`; honesty fields (`QualityReport`) all `None`+`reason` for stub | ✅ `packaging.py:227-244` | ✅ same | ✅ same | ✅ same | ✅ same |
| **Billing** | ✅ real, deterministic from delivered artifacts — `billing.py:177-254`, `main.py:312-315`; LLM line only when spec stage actually spent (`main.py:135-148`) | ✅ same | ✅ | ✅ | ✅ | ✅ |

Notes on the matrix:
- "GPU" columns assume `ASTEL_PRODUCER=gpu` is set; if unset, GPU columns are
  unreachable and every modality silently runs the CPU stub
  (`gpu_producer.py:139-141` — there is no error/log if the env var is merely
  absent, which is itself a soft "is the GPU pipeline even on?" ambiguity, see
  §2).
- The **only** modality/producer cell that performs prompt- or
  capture-conditioned generation is **Image · GPU** (`generative.py`). All five
  other cells produce a prompt/capture-independent procedural or self-consistency
  cloud for L0/L3.

---

## 2. SILENT FALLBACKS

Ranked roughly by how likely each is to surprise a caller who only looks at
top-level API/billing fields (not prose caveats).

### 2.1 Video silently aliases to the text smoke path (NEW finding, same class as the text gap)
- **Location:** `pipelines/gpu/src/astel_gpu/produce.py:208-227` (`produce()`); the
  only branch is `if modality == "image" and image is not None and image.is_file()`.
  Any other modality — including `"video"` — falls through to `_produce_smoke`
  (`produce.py:159-205`), the exact same code path as text.
- **Trigger:** `POST /v1/generations` with `modality=video` (+ optional
  `capture_id` of an uploaded video) and `ASTEL_PRODUCER=gpu`.
- **Surfaced today?** Partially. `apps/web/src/components/GenerationDock.tsx:56-61`
  has an honest `HINT["video"]` string ("Video capture ... is on the M6 roadmap
  and not wired yet ... produce the placeholder preview"). But:
  - The **API response** (`GenerationResource`, `schemas.py:158-168`) carries no
    field distinguishing "video produced a real reconstruction" from "video
    produced the text-identical smoke." A programmatic client (SDK/MCP — CLAUDE.md
    §7 API-first) gets no signal.
  - `quality-report.json`'s `modality` field for this path is whatever string was
    passed in (`"video"`), but the caveats text (`produce.py:104-108`) literally
    says **"the text modality runs a render-then-refit smoke"** — i.e. the
    GPU-side caveat copy was written for the text case and is reused verbatim for
    video without being updated, which is itself a smaller honesty bug (a video
    submitter reads a caveat about "the text modality").
  - The uploaded video capture (`capture_id`) is NEVER resolved for video — only
    `modality == "image"` triggers `_resolve_capture_image`
    (`gpu_producer.py:95-96`). So a video upload is stored (real bytes, real
    `capture_id`) but **100% unused**, same as image-modality-on-CPU-stub.

### 2.2 CPU-stub ignores `capture_id` for ALL modalities, with no per-response flag
- **Location:** `services/api/src/astel_api/producer.py:294-369` —
  `produce_artifacts(task_id, modality, prompt, store)` has no `capture_id`
  parameter at all; `gpu_producer.produce_artifacts_dispatch` only forwards
  `capture_id` to the GPU path (`gpu_producer.py:125-141`).
- **Trigger:** any generation where `ASTEL_PRODUCER` is unset/not `"gpu"` (the
  default in dev/CI) and the request includes `capture_id` from a prior
  `POST /v1/captures`.
- **Surfaced today?** Only in source comments (`producer.py` module docstring
  lines 1-24, `schemas.py:113-116` docstring on `CreateGenerationRequest
  .capture_id`) — nothing in the JSON response. `GenerationResource` has no
  "producer: stub|gpu" / "capture_consumed: bool" field. A caller cannot
  distinguish "my photo was reconstructed" from "my photo was silently dropped"
  without comparing the returned geometry to their upload by eye.

### 2.3 `ASTEL_PRODUCER` env var typos/absence are invisible
- **Location:** `gpu_producer.py:139-141`:
  ```python
  if os.environ.get("ASTEL_PRODUCER") == "gpu":
      return _run_gpu_producer(...)
  return produce_artifacts(...)
  ```
- **Trigger:** any value other than the exact string `"gpu"` (e.g. `"GPU"`,
  `"true"`, `"1"`, a typo) silently routes to the CPU stub. No warning is logged.
- **Surfaced today?** No. This is an operational footgun: a deploy that intends
  GPU-backed generation but has the env var misconfigured will serve placeholder
  geometry for every request, with `quality-report.json.origin` correctly saying
  `"stub"` (`producer.py:212`) — but nothing *alerts* on that; it's discoverable
  only by reading the per-asset report.

### 2.4 `package.astel`'s typed `QualityReport` has no `origin`/`status` field at all
- **Location:** `libs/astel_format/src/astel_format/models.py:414-427` —
  `QualityReport` has `geometric_error`, `scale_confidence`, `hallucination`,
  optional `view_metrics`/`stage_telemetry`, and `caveats: list[str] | None`.
  There is **no enum field** like `origin: "stub"|"measured"|"generated"` —
  unlike the separate `astel.quality-report/v0` dict (`producer.py:211-212`,
  `produce.py:65-66`, `generative.py:90-91`) which DOES have a top-level
  `"origin"` key.
- **Trigger:** any consumer that loads `package.astel`'s embedded quality report
  (the manifest-v0 schema-validated one — Unity/UE5 plugins, USD pipelines, the
  print path) rather than the web-only `quality-report.json` dict.
- **Surfaced today?** No — only via free-text `caveats` strings (e.g.
  `producer.py:284-290`, `packaging.py:109-116`). Free text is exactly the kind
  of signal that gets dropped when a downstream tool only checks structured
  fields — this is the structural root cause that let the text→smoke gap stay
  prose-only instead of being a field a dashboard could alert on.

### 2.5 Generation-Spec / L6 LLM stages: cache-miss "skipped" is silent w.r.t. billing only, not API
- **Location:** `generation_spec_stage.py:68-79`, `physics_material_stage.py:68-79`.
  On `FixtureMissingError` both write an honest `{"status": "skipped", "reason":
  ...}` artifact (`generation-spec.json` / `physics-material.json`) and return
  early.
- **Trigger:** any *new* (non-fixture-cached) text prompt when
  `ASTEL_LLM_LIVE`/`ANTHROPIC_API_KEY` are not both set (the default).
- **Surfaced today?** Billing is honest (no `LLM_SPEC`/`L6` line item is charged —
  `main.py:135-148`, `billing.py:74-90` only maps `l6.json`, not
  `physics-material.json`, to a billable code). But `GenerationResource` itself
  has no field reflecting "spec stage ran / skipped" — a caller has to fetch and
  parse `generation-spec.json` to find out the prompt was never actually turned
  into a structured spec, which silently degrades the L6/scale-confidence
  pipeline for that asset (the quality report's `scale` block stays at the
  un-grounded `1.0/1.0/1.0` identity from `producer.build_package_quality_report`
  / `packaging.build_package_quality_report` rather than the LLM estimate from
  `apply_spec_scale_to_report`, `generation_spec_stage.py:100-128`).

### 2.6 `produce_artifacts_dispatch` swallows GPU subprocess failures into the generic except in `main.py`
- **Location:** `_run_gpu_producer` (`gpu_producer.py:63-122`) uses
  `subprocess.run(..., check=True)`, which raises `CalledProcessError` on any
  non-zero exit (e.g. CUDA OOM, missing checkpoint, gsplat JIT build failure).
  This propagates up to `main.py:285-319`'s
  ```python
  try:
      produce_artifacts_dispatch(...)
      ...
  except Exception:
      logger.exception("artifact production failed for %s", task_id)
  ```
- **Trigger:** any GPU-side crash (very plausible given gsplat JIT/MSVC build
  fragility noted in project memory).
- **Surfaced today?** The generation row is still created and returned with
  `status: QUEUED` and **zero artifacts** and `billing: None` (since
  `_build_and_store_billing` is never reached) — `main.py:321-332`. The SSE
  engine (`InProcessStubEngine`) then runs its **simulated** L0-L3 progress
  (`engine.py:53-120`) to completion regardless, ending in
  `status: SUCCEEDED` / "Asset ready" with fabricated `_STAGE_TARGETS` metrics
  (`engine.py:29-38`) — **even though no artifacts exist**. A client following
  the documented happy path (submit → SSE "Asset ready" → fetch `l3.ply`) gets a
  404 on the artifact fetch after being told the asset is ready. This is the
  single largest "looks-done-but-isn't" gap in the audit: the SSE progress
  stream is fully decoupled from whether `produce_artifacts_dispatch` actually
  succeeded.

### 2.7 `InProcessStubEngine` progress/metrics are independent of real production outcome (generalization of 2.6)
- **Location:** `engine.py:53-120`, `_STAGE_TARGETS` (`engine.py:29-38`). The SSE
  endpoint (`main.py:356-384`) runs this simulation for *every* generation,
  whether the producer dispatch (already completed synchronously in
  `create_generation`, `main.py:284-319`) succeeded, partially succeeded, or threw.
- **Trigger:** always, in the default (non-Temporal) engine
  (`get_engine`, `main.py:75-86`, default `InProcessStubEngine`).
- **Surfaced today?** No. The `metrics` in the terminal `ProgressEvent`
  (`StageMetrics(splats=48_000, psnr_db=31.2, chamfer_mm=0.9, ...)`,
  `engine.py:35-37`) are hardcoded targets unrelated to what
  `produce_artifacts_dispatch` actually produced (which could be a 65536-gaussian
  TripoSplat cloud, an 8000-gaussian smoke cloud, or nothing at all on failure).
  The web UI's `ProgressRail` (`GenerationDock.tsx:251-257`) displays
  `metric.splats` / `metric.chamfer_mm` straight from this fabricated
  `ProgressEvent` — so the "48k splats" / "0.9mm" shown to the user on the
  progress rail is **always** the same hardcoded number, never the real
  per-task `quality-report.json` count.

---

## 3. MISSING / NOT WIRED (mission requires, nothing implements)

| Mission requirement (CLAUDE.md) | Status | Where it would plug in |
|---|---|---|
| **§4 Text pipeline**: prompt → Generation Spec → text-to-multiview → feed-forward gaussian (L2) → L3 refine w/ MV-diffusion guidance | ❌ Spec stage exists (`generation_spec_stage.py`, real when live-gated) but stops there. Text→multiview generator: **does not exist** anywhere in `pipelines/gpu/src/astel_gpu/`. | A new module e.g. `astel_gpu/text_to_multiview.py` producing views consumable by `l2_triposplat.run_l2`-equivalent or a dedicated text-conditioned L2 model; wired into `produce.py` as a third branch (`modality == "text"` with a real spec) before falling back to `_produce_smoke`. |
| **§3 L1 — Dense Cloud** (metrically-scaled, normals, semantic logits; SfM-derived or VLM-estimated) | ❌ No `l1.ply` writer in either producer. `_ARTIFACT_LAYER["l1.ply"] = "L1"` exists in `billing.py:76` as dead config. | `astel_gpu/packaging.write_layer_stack` would need an `l1_cloud` param analogous to `l2_cloud` (`packaging.py:177,202-204`); for the capture path, `capture_sfm.py`/`colmap_runner.py` already produce SfM point clouds that are the natural L1 source but aren't threaded into `produce.py` at all (only `capture_eval.py`/`synthetic_eval.py` use them, for offline evals). |
| **§3 L4 — Appearance/Lighting** (per-gaussian albedo/roughness/metallic/specular/emissive + env-light decomposition) | ❌ Not implemented. No module name matches (`grep` for "L4"/"appearance"/"BRDF"/"relight" across `pipelines/gpu` and `services/api` returns nothing besides the dead `billing.py:82` mapping `"l4.ply": "L4"`). | Would sit between L3 (`l3_refine.py` output) and L5 (`packaging._try_solidify`) in `write_layer_stack` (`packaging.py:167-249`); needs a BRDF-decomposition pass producing per-splat material channels, written as `l4.ply`/`l4.json`. |
| **§3 L7 — Dynamics** (4DGS deformation field for video input) | ❌ Not implemented. No `4dgs`/`deformable` module exists. Video modality (see §2.1) doesn't even reach a static-but-real reconstruction, let alone dynamics. | Would be a video-specific branch in `produce.py` (parallel to `_produce_generative`), e.g. `_produce_dynamic(...)`, using pose-free reconstruction (DUSt3R/VGGT-class — also not present) feeding a 4DGS optimizer; output `l7.ply`/keyframe buffers wired into `write_layer_stack` and `_ARTIFACT_LAYER["l7.ply"] = "L7"` (already present, `billing.py:86`, also dead). |
| **§4 Video pipeline**: frame selection → pose-free reconstruction/SfM → static (as photos) or dynamic (4DGS) | ❌ Entirely unwired for the GPU producer (§2.1). `capture_sfm.py`/`colmap_runner.py`/`colmap_io.py` exist and are real (used in `capture_eval.py` for offline DTU-style evals) but are never called from `produce.py`. | `produce.py` needs a `modality == "video"` (or generic "capture") branch that: extracts frames from the uploaded video capture (`_resolve_capture_image`-equivalent for video, currently absent — §2.1), runs `capture_sfm`/`colmap_runner` for L0/L1, then L3 refine via `l3_refine.optimize_2dgs` (already real, used by the image path). |
| **§4 "All pipelines converge at L3 and share L4-L6"** | 🟡 Partial — L3 and L5/L6 paths do converge (`write_layer_stack` is shared, L6 stage is modality-agnostic in code though gated to text), but since L1/L2/L4/L7 don't exist for text/video, "convergence" is currently "everything but image converges on the same prompt-independent L3". | N/A — depends on the above being built first. |
| **§5 Engine: Temporal** | 🟡 `TemporalTaskEngine` (`engine.py:200-294`) and `services/api/src/astel_api/temporal/*` exist and look real (workflows, activities, devserver), but `get_engine` (`main.py:75-86`) defaults to `InProcessStubEngine` unless `settings.engine == "temporal"`. Given §2.6/2.7, the in-process engine's decoupling from actual production result is itself a correctness gap independent of Temporal. | If Temporal is the intended default, `Settings.engine` default (`config.py`) should be checked — but that's a config concern, not a missing-code one; not deep-dived here per scope. |

---

## 4. RECOMMENDATIONS

1. **Add a structured `origin`/`status` enum to the typed `QualityReport`**
   (`libs/astel_format/src/astel_format/models.py:414-427`), mirroring the
   `astel.quality-report/v0` dict's existing `"origin": "stub"|"measured"` field
   (`producer.py:211-212`, `produce.py:65-66`). Make it a closed enum, e.g.
   `origin: Literal["measured-capture", "generated-conditioned",
   "generated-unconditioned", "stub"]`, and make `build_minimal_package`
   (`libs/astel_format/src/astel_format/builder.py`) require it as a positional
   arg (no default) so every call site is forced to state honestly which bucket
   it's in. `"generated-unconditioned"` is exactly the bucket the text-smoke and
   video-smoke paths belong to — distinct from `"generated-conditioned"` (image
   path) and `"measured-capture"` (future SfM path).

2. **Add a top-level `conditioning` field to `GenerationResource`**
   (`schemas.py:158-168`): something like
   `conditioning: Literal["prompt", "image", "video", "none"]` reflecting what,
   if anything, the L3 geometry was actually conditioned on for *this specific
   task*. Populate it from the producer's return dict (`produce_artifacts*`
   already returns a dict that could carry this — `producer.py:365-369`,
   `produce.py:151-156,200-205`). This single field would have made the original
   text gap visible in the API response without reading any prose.

3. **Make `_produce_smoke`'s caveats modality-aware** (`produce.py:95-108`): the
   current caveat text hardcodes "the text modality runs a render-then-refit
   smoke" even when `modality == "video"` (or any future non-image modality)
   triggers the same fallback. Interpolate `modality` into the caveat and add an
   explicit statement "{modality} input was NOT used to condition this geometry."

4. **Resolve and pass video captures into the GPU producer**, even if the
   immediate effect is only "extract first frame and treat as image input" as an
   interim honest improvement over "always smoke" — and regardless, emit
   `conditioning: "none"` (per #2) until true video reconstruction lands, so the
   interim behavior is structurally flagged.

5. **Decouple SSE progress/metrics from fabricated targets (§2.6/2.7).** At
   minimum: after `produce_artifacts_dispatch` returns (or raises) in
   `create_generation` (`main.py:284-319`), persist its outcome (success +
   real splat count from the returned dict, or failure) on the `Generation` row;
   have `InProcessStubEngine.run` (`engine.py:61-120`) read that outcome and (a)
   emit `TaskStatus.FAILED` with a real error message if production failed —
   never claim "Asset ready" with zero artifacts — and (b) report the *actual*
   `splats`/`psnr_db`/etc. from `quality-report.json` in the terminal
   `ProgressEvent.metrics` instead of the hardcoded `_STAGE_TARGETS`
   (`engine.py:29-38`).

6. **Warn-log on `ASTEL_PRODUCER` misconfiguration** (`gpu_producer.py:139-141`):
   if the env var is set but not exactly `"gpu"` (e.g. truthy-looking values),
   log a warning that the stub is being used despite an apparent intent to use
   GPU. Also log at INFO which producer path was actually taken per request —
   currently there is no log line distinguishing stub vs GPU dispatch at all.

7. **Surface Generation-Spec/L6 "skipped" status in `GenerationResource`**
   (complements #2): add `spec_stage: "ok" | "skipped" | "not-applicable"` and
   `l6_stage: "ok" | "skipped" | "not-applicable"` fields so a caller knows
   without fetching `generation-spec.json`/`physics-material.json` whether the
   reported `scale` in the quality report is an LLM estimate or the ungrounded
   identity default.

8. **CI check for "dead" `_ARTIFACT_LAYER` entries** (`billing.py:74-89`): `l1.ply`,
   `l4.ply`, `l7.ply`, `print.3mf`/`print.stl` are mapped to billable layer codes
   that no producer ever emits. A simple test asserting "every key in
   `_ARTIFACT_LAYER` is either produced by at least one producer path or
   explicitly listed in a `NOT_YET_IMPLEMENTED` allowlist with a tracking issue"
   would prevent silent drift between the pricing schedule (which markets L1/L4/L7
   as purchasable add-ons, `billing.py:50-60`) and what the system can deliver.
