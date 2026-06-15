# 16 — Dead/Unused Code Audit (2026-06-15)

*Method: static grep-based import/reference tracing across all first-party
Python (`libs/`, `pipelines/`, `services/`, `experiments/`), TypeScript
(`apps/`, `packages/`), and config (`pyproject.toml`, `package.json`) files.
Excluded: `.venv/`, `node_modules/`, `pipelines/gpu/external/` (vendored
TripoSplat), `dist/`, `build/`, `__pycache__/`, `data/`, `.git/`,
`pipelines/gpu/out/` (generated artifacts, now gitignored per the pending
`.gitignore` diff — not source). Read/Grep only; nothing deleted or modified.
Builds on the wiring gaps already identified in
`docs/research/15-pipeline-wiring-audit.md`, which is corroborated below
rather than re-derived from scratch.*

## TL;DR

The codebase is **unusually clean** for its size — almost everything that
exists is imported by production code, a test, or both. The two genuinely dead
items are:

1. `pipelines/gpu/src/astel_gpu/triposplat_spike.py` — explicitly superseded by
   `l2_triposplat.py` per session-12 notes, zero importers, no test. **DELETE.**
2. Three unused keys in `services/api/src/astel_api/billing.py`'s
   `_ARTIFACT_LAYER` table (`l1.ply`, `l4.ply`, `l7.ply`) — forward-looking
   placeholders for layers that don't exist yet (already flagged in doc 15).
   **KEEP** (harmless, documents the intended L1/L4/L7 wiring), but don't add
   more until the layers they map exist.

One duplication is real and worth a small refactor: `stable_seed()` is
byte-identical in `pipelines/gpu/src/astel_gpu/produce.py` and
`services/api/src/astel_api/producer.py`. Everything else flagged below is
either (a) a documented, tested, not-yet-consumed forward-built artifact
(`@astel/manifest`), (b) a CLI tool used only via `run-python.cmd`/docs (real,
just not imported by other modules), or (c) a transitive dependency of
`diffusers`'/TripoSplat's stack that doesn't show up as a direct first-party
import.

## DISPOSITION TABLE

| Path (+ line) | What it is | Evidence it's unused | Recommendation |
|---|---|---|---|
| `pipelines/gpu/src/astel_gpu/triposplat_spike.py` (97 lines, whole file) | M3-step-2 install spike: one-shot single-image → TripoSplat gaussians CLI, explicitly marked "Not wired into `astel_gpu.produce` yet" in its own docstring | `grep -rn "triposplat_spike"` across first-party `.py` returns **zero** importers. `docs/retros/session-12.md:12` confirms "The spike `triposplat_spike.py` from session 11 step 2 graduates into production" as `l2_triposplat.py`. No test file (`test_triposplat_spike*`) exists, unlike every other module in `astel_gpu`. `l2_triposplat.py`'s own docstring says "Graduates `triposplat_spike.py` ... into a typed module and fixes [its issues]". | **DELETE**. Graduation is complete and documented; the spike is pure leftover. (Its `.pyc` in `__pycache__` will also go away on next clean.) |
| `services/api/src/astel_api/billing.py:76` — `"l1.ply": "L1"` in `_ARTIFACT_LAYER` | Dict entry mapping a not-yet-produced artifact filename to its layer ID, used by `_layers_present()` (`billing.py:98`, a set-comprehension filter) | `grep -rn "l1.ply\|\"L1\"" pipelines services` — no producer ever writes `l1.ply`. `docs/research/15-pipeline-wiring-audit.md:208` already flags this exact line as "dead config". | **KEEP-WITH-REASON**. Harmless: `_layers_present` silently skips keys whose artifact never appears (`if n in _ARTIFACT_LAYER`), so this is a forward stub for when L1 (dense cloud) gets a writer — not a bug, just unused today. Don't expand the table further until the corresponding layer exists. |
| `services/api/src/astel_api/billing.py:82` — `"l4.ply": "L4"` in `_ARTIFACT_LAYER` | Same table, L4 (appearance/lighting) placeholder | No `l4.ply` writer anywhere in `pipelines/gpu` or `services/api` (confirmed via grep for "L4"/"appearance"/"BRDF"/"relight" — doc 15 §3). | **KEEP-WITH-REASON**. Same as above — forward stub for L4, not yet implementable until BRDF decomposition exists (M4). |
| `services/api/src/astel_api/billing.py:86` — `"l7.ply": "L7"` in `_ARTIFACT_LAYER` | Same table, L7 (dynamics/4DGS) placeholder | No `l7.ply` writer; no 4DGS/deformable module exists anywhere (doc 15 §3). | **KEEP-WITH-REASON**. Forward stub for M6 dynamics layer. |
| `pipelines/gpu/src/astel_gpu/produce.py:46-52` `stable_seed()` vs. `services/api/src/astel_api/producer.py:62-65` `stable_seed()` | Two byte-identical functions: blake2b(task_id)[:4] → int, including identical docstrings | Each module imports nothing from the other (`astel_gpu` and `astel_api` are separate packages with no shared dependency edge today — `astel_api` deliberately keeps torch/gsplat out of its import graph per `gpu_producer.py`'s docstring). Both have their own callers (`produce.py:_produce_smoke`-family; `producer.py:produce_artifacts`). | **KEEP-WITH-REASON, but flag for a future shared helper**. The duplication is small (7 lines) and currently *load-bearing isolation*: `astel_api` must not import `astel_gpu` (torch/gsplat). If a third copy appears, extract into a tiny dependency-free shared lib (e.g. a new `libs/astel_common` with zero deps) rather than importing across the API/GPU boundary. Not urgent at 2 copies. |
| `pipelines/gpu/src/astel_gpu/produce.py:54` `build_quality_report(*, count, modality, psnr_db, n_views)` vs. `services/api/src/astel_api/producer.py:203` `build_quality_report(*, count, modality)` vs. `pipelines/gpu/src/astel_gpu/synthetic_eval.py:86` `build_synthetic_quality_report(...)` | Three "build the `astel.quality-report/v0` dict" functions, each with a different `origin` value (`"measured"` render-then-refit smoke / `"stub"` / synthetic-eval-specific) and different honest-unmeasured-field sets | Each is used by its own caller only (`produce.py`'s smoke path, `producer.py.produce_artifacts`, `synthetic_eval.run_synthetic_eval`). `synthetic_eval.py:98` explicitly docstrings the difference from `produce.build_quality_report`. | **KEEP-WITH-REASON**. Not true duplication — each encodes a genuinely different honesty story (different fields are `None` vs. measured per CLAUDE.md §10.4's "no silent hallucination" rule), and the differences are deliberate and documented. A shared "skeleton dict with these 6 top-level keys" helper *could* reduce repetition, but conflating the three risks accidentally leaking a measured field into a stub report — current explicitness is arguably the safer design. Low-priority simplify candidate only. |
| `experiments/task-engine-spike/` (10 tracked files, 33 KB; `main.py`, `src/{activities,query_progress,shared,starter,worker,workflows}.py`, `pyproject.toml`, `README.md`, `uv.lock`, `.python-version`) | M0-era hands-on Temporal-on-Windows spike (`AssetPipelineWorkflow`, 3-stage stub activity) | `grep -rn "task-engine-spike"` — only self-references plus doc mentions (`docs/research/10-task-engine-spike.md`, `docs/architecture/ARCHITECTURE.md`, `docs/NEXT_STEPS.md`, `docs/retros/session-02.md`, `infra/docker-compose.yml` — the compose reference is to a Temporal *service*, not this dir). The spike's findings graduated into `services/api/src/astel_api/temporal/{workflows,activities,worker,devserver,shared}.py` + `engine.py`'s `TemporalTaskEngine`, which **is** wired into `main.py` (`get_engine`, gated by `settings.engine == "temporal"`). The spike's `bin/temporal.exe` (553 MB), `data/temporal.db`, and `logs/` mentioned in its own README were never committed (only 33 KB of source is tracked). | **GRADUATE (done) → DELETE the experiment dir**. The decision (Temporal as task engine) and the implementation pattern both graduated into `services/api/src/astel_api/temporal/`. The experiment's only remaining value is historical/narrative, which `docs/research/10-task-engine-spike.md` and `docs/retros/session-02.md` already preserve. Tracked size is small (33 KB) so this is low-stakes either way — but per CLAUDE.md's binding rule ("graduate or get deleted"), it has graduated and should go. |
| `pipelines/gpu/src/astel_gpu/colmap_runner.py` (CLI, `run_sfm`/`main`) | COLMAP SfM pipeline driver (feature extraction → matching → mapping → undistortion) | No first-party `.py` module imports it (`grep -rn "colmap_runner" src tests` → only `tests/test_colmap_runner_cpu.py`). However it's a documented standalone CLI: `pipelines/gpu/README.md:118` (`-m astel_gpu.colmap_runner --image-dir DIR --work-dir DIR`), exercised on real DTU images per `docs/retros/session-09.md`/`session-10.md`, and is the SfM front-end `capture_sfm.py` consumes output from (file-based handoff via `colmap_io.load_colmap_model`, not a Python import). | **KEEP** — real, tested (CPU unit tests), documented CLI entry point; "no importer" is expected for a CLI tool invoked via `run-python.cmd -m`. |
| `pipelines/gpu/src/astel_gpu/capture_sfm.py` (CLI, `main`) | COLMAP↔DTU pose-accuracy validator (Umeyama alignment) | Only imported by `tests/test_capture_eval_cpu.py`(no — actually only its own functions `colmap_io`/`dtu` are imported elsewhere; `capture_sfm` itself has no Python importer). Documented CLI: `README.md:121` (`-m astel_gpu.capture_sfm --colmap-model-dir DIR --pos-dir DIR`), used in session-10 to validate the session-9 `colmap_runner` output against DTU ground truth. | **KEEP** — same as above, real validated CLI tool, not a Python-import dependency by design. |
| `pipelines/gpu/src/astel_gpu/smoke_refit.py` | gsplat render-then-refit smoke (`run_smoke`, `optimize`, `render_views`, `RenderInputs`, `d_ssim_loss`, `make_trainable`, `DEFAULT_ITERS`, etc.) | Heavily imported: `capture_eval.py`, `generative.py`, `l3_refine.py`, `produce.py`, `synthetic_eval.py`, plus its own `tests/test_smoke_refit.py` (GPU-gated) and CLI script (`astel-gpu-smoke` in `pyproject.toml:51`). | **KEEP** — this is a core shared module, not a spike despite the "smoke" name; it's the common Adam/SSIM/render machinery every refine path mirrors or imports directly. |
| `pipelines/gpu/src/astel_gpu/synthetic.py` / `synthetic_eval.py` | Procedural ground-truth synthetic scene + eval harness (first REAL Chamfer-vs-GT check) | `synthetic.py` imported only by `synthetic_eval.py`; `synthetic_eval.run_synthetic_eval` imported by `tests/test_synthetic_eval.py` (GPU-gated, per `conftest.py`) and has a `main()`/`if __name__` CLI. `docs/MVP_TESTING.md:130` documents running it via `run-python.cmd -m pytest tests/test_smoke_refit.py tests/test_synthetic_eval.py`. | **KEEP** — real eval harness, used by tests + documented as part of the GPU test suite (CLAUDE.md §10.5 "golden-file / integration tests on a fixed corpus"). |
| `pipelines/gpu/src/astel_gpu/capture_eval.py` | DTU-based Chamfer/PSNR eval for the capture path (`run_capture_eval`, `split_train_test`) | Imports `smoke_refit`, `dtu`, `capture_sfm`-adjacent `dtu` helpers, `gaussians`, `export`; imported by `generative.py` (`split_train_test`) and `tests/test_capture_eval_cpu.py`; has CLI `main()`. Referenced extensively in `docs/research/DECISIONS.md` and session 9-10 retros as the M2 accuracy story. | **KEEP** — real, wired, documented as the flagship "reality first" accuracy proof (CLAUDE.md §9 M2). |
| `experiments/` (top level) | Only contains `task-engine-spike/` | See above — sole entry, see disposition above. | See `task-engine-spike` row. |
| `packages/manifest/` (`@astel/manifest`, TS types + reader/writer for `.astel` manifest, 10 vitest tests passing per session-21 retro) | TS mirror of `libs/astel_format` (Python) — manifest types, schema validation, reader/writer | `grep -rn "@astel/manifest" apps/web/src` → **zero** — `apps/web` does not currently import this package; `apps/web/src/lib/{report.ts,layers.ts}` define their *own* parallel `QualityReport`/`LayerId` types instead. `pnpm-workspace.yaml` includes `packages/*` so it builds/tests in CI (`docs/MVP_TESTING.md:123`, `docs/retros/session-21.md:33`: "10 vitest ✓ · tsc ✓ · eslint ✓"). `docs/architecture/ARCHITECTURE.md:17` documents it as "Shared TS packages (@astel/manifest; @astel/sdk reserved)". | **KEEP-WITH-REASON**. Not dead — it's a deliberately-built, tested, documented dual-language (Python+TS) format contract that's part of the architecture spec (mirrors `libs/astel_format`). It's *not yet consumed* by `apps/web`, which has grown its own lighter-weight duplicate types (`apps/web/src/lib/report.ts`, `layers.ts`). **Follow-up worth scheduling** (not this audit's call to make): either (a) have `apps/web` import `@astel/manifest`'s types for `QualityReport`/layer IDs to remove the parallel definitions, or (b) if `apps/web`'s simpler types are intentionally decoupled from the full manifest schema, document why in `ARCHITECTURE.md` so the next session doesn't flag this as drift. |
| `pipelines/gpu/pyproject.toml` deps: `transformers`, `accelerate`, `sentencepiece`, `protobuf`, `safetensors`, `tqdm`, `torchvision`, `huggingface-hub` | Declared runtime deps of `astel-gpu` | None of these appear as direct `import` statements in `pipelines/gpu/src/astel_gpu/*.py` (only `diffusers` is directly imported, in `text_to_image.py`). | **KEEP** — all are transitive runtime deps of `diffusers.FluxPipeline` (T5 encoder needs `transformers`+`sentencepiece`+`protobuf`; device/dtype placement commonly needs `accelerate`) and of the vendored TripoSplat (`torchvision.ops.deform_conv2d`, `safetensors.torch.load_file`, `tqdm.auto` per `docs/research/14-triposplat-triage.md` §2). Not unused — just not directly imported by first-party code, which is normal for a model-loading stack. No action. |
| `pipelines/gpu/src/astel_gpu/env_check.py` | `torch`/CUDA/GPU-visibility printer (`main()`) | No Python importer (expected — it's a diagnostic CLI). | **KEEP** — documented (`README.md:87-89`) and actively used by `scripts/setup-gpu-env.ps1:109-117` as part of the GPU box bring-up flow. Real and wired into tooling. |
| `services/api/src/astel_api/producer.py` vs. `gpu_producer.py` | Stub (CPU, procedural) vs. GPU-subprocess producer | Both real: `gpu_producer.produce_artifacts_dispatch` (called from `main.py`) imports and calls `producer.produce_artifacts` for the default (non-`ASTEL_PRODUCER=gpu`) path, and shells out to `pipelines/gpu` for the GPU path. `producer.py` also has its own importer in `tests/test_artifacts.py`. | **KEEP** — this is intentional dispatch (env-var-gated), not duplication; the stub *is* the fallback/default implementation, by design (`gpu_producer.py`'s own docstring: "byte-for-byte unchanged behaviour" for the default path). |

## Commented-out code / unreachable code

Searched for comment lines that look like dead code (`#`/`//` followed by
`def`/`class`/`return`/`import`/`from`/control-flow keywords with code-like
punctuation) and for `if False:`/`if 0:` blocks across all first-party
`.py`/`.ts`/`.tsx` files.

**Result: none found.** The only comment block matching the code-shaped
pattern (`services/api/src/astel_api/main.py:293-297`) is prose explaining
*why* the Generation Spec stage runs after `produce`, not commented-out code.
The four `raise NotImplementedError(...)` sites (`astel_splat_io/spz.py:264,291`,
`sog.py:306`, `provenance.py:137`) are intentional unsupported-format guards
with messages, not stubs-to-delete.

## Declared-but-unused dependencies — pyproject.toml / package.json

Checked every `pyproject.toml` (7, excluding `experiments/`) and both
`package.json` files (`apps/web`, `packages/manifest`) plus the root.

- **`libs/astel_eval`, `libs/astel_format`, `libs/astel_solid`,
  `libs/astel_splat_io`, `libs/astel_llm`**: every declared dependency
  (`numpy`, `pydantic`, `jsonschema`, `referencing`, `scipy`, `scikit-image`,
  `pillow`, optional `anthropic`) has a corresponding import. No unused deps.
- **`pipelines/gpu`**: see the transitive-deps row above (`transformers`,
  `accelerate`, `sentencepiece`, `protobuf`, `safetensors`, `tqdm`,
  `torchvision`, `huggingface-hub` — all justified as `diffusers`/TripoSplat
  transitive runtime needs, not first-party imports). `astel-splat-io`,
  `astel-format`, `astel-solid` all confirmed imported (`l2_triposplat.py`,
  `produce.py`/`packaging.py`/`export.py`, `packaging.py` respectively).
- **`pipelines/stub`**: only dep is `numpy`, used in `make_sample_splat.py`. OK.
- **`services/api`**: all of `fastapi`, `uvicorn`, `sqlalchemy`, `aiosqlite`,
  `pydantic`, `pydantic-settings`, `sse-starlette`, `python-multipart`,
  `temporalio`, `astel-splat-io`, `astel-format`, `astel-llm`, `alembic` have
  confirmed importers (`main.py`, `db.py`, `config.py`, `schemas.py`,
  `temporal/*`, `producer.py`/`gpu_producer.py`, `generation_spec_stage.py`/
  `physics_material_stage.py`, `migrations/`). No unused deps.
- **`apps/web/package.json`**: `@sparkjsdev/spark` (splat rendering — used in
  `SplatScene.ts`), `react`/`react-dom` (App.tsx etc.), `three` (Viewport/
  SplatScene). All used. Dev deps (`vitest`, `jsdom`, `@testing-library/*`,
  eslint stack) match the `*.test.tsx`/`vitest.setup.ts` files present.
- **`packages/manifest/package.json`**: `ajv`/`ajv-formats` used by
  `schema_validation`-equivalent in `src/schemas/index.ts` (schema compilation);
  package itself flagged above as not-yet-consumed by `apps/web` but its own
  deps are all used by its own source/tests.

No unused dependencies found in any manifest.

## Summary counts

- **DELETE**: 2 (`triposplat_spike.py`; `experiments/task-engine-spike/`)
- **GRADUATE**: 0 new (the one candidate, the task-engine spike, has *already*
  graduated into `services/api/src/astel_api/temporal/` — its disposition is
  DELETE-the-source now that graduation is complete)
- **KEEP / KEEP-WITH-REASON**: 13 rows (forward-looking config table entries,
  intentional small duplication at a package boundary, distinct
  honesty-driven `build_quality_report` variants, CLI-only tools, transitive
  deps, and the not-yet-consumed `@astel/manifest` package)
