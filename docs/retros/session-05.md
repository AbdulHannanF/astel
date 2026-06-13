# Session 5 retro (2026-06-13)

Mode: Opus (founder) planned + reviewed; 3 parallel subagents (2 Opus for the
heavier algorithmic/architectural slices, 1 Sonnet for docs) executed; Opus
verified all gates afterward. No GPU. Closes out `docs/OPEN_ISSUES.md` (the
end-of-session-4 verification pass) and lands the next P3 slice of M2's CPU
spine.

## P1 — `astel_eval` suite 8m30s → ~4s

`libs/astel_eval/src/astel_eval/bradley_terry.py::_fit_strengths` used a
pure-Python O(n^2) double loop with `max_iter=10_000, tol=1e-10`. On the
fully-separated gate corpus (Astel wins all 50 cases) the Bradley-Terry MLE
diverges and never hits `tol`, so every fit ran the full 10k iterations.

Fix: vectorized the Zermelo update with numpy broadcasting, added a symmetric
0.5/0.5 smoothing-tie prior on every compared pair (gives separated data a
finite stable fixed point — correctness fix, not just speed), and switched to
relative-change early-stop with `max_iter=200`. Public API unchanged; all 36
tests pass with the same gate conclusions (Astel still strictly ahead). Local
wall time: 48.6s → 3.98s (the "8m30s" in OPEN_ISSUES was a slower CI-class box;
same ~12x speedup either way).

## P2 — README.md + ARCHITECTURE.md de-staled

- `README.md`: status line + phases list now say "R closed, M1 closed, M2
  spine landed (CPU)"; quickstart fixed from wrong `npm install`/`npm run dev`
  to `pnpm install` + `pnpm run up` (with `dev:all`/`dev` alternatives).
- `docs/architecture/ARCHITECTURE.md`: added the `libs/` package group
  (`astel_format`, `astel_splat_io`, `astel_eval`) to the monorepo layout;
  listed all 6 CI jobs (web/manifest/api/pipeline-stub/libs/license-gate);
  rewrote the artifact-flow section (LocalArtifactStore + producer.py are
  real, not "no artifacts until M2"); rewrote the TaskEngine section
  (TemporalTaskEngine exists since session 3, `ASTEL_ENGINE=stub|temporal`);
  refreshed verified-checks counts (web 15, manifest 10, api 17+1skip,
  astel_format 16, astel_splat_io 11, astel_eval ~36).

## P3 — full layer-stack artifacts + `.astel` packaging + `/v1/captures` upload

`services/api/src/astel_api/producer.py` now writes the **full per-task
artifact set** on submit: `l0.ply` (strided subsample of L3, divisor 24),
`l3.ply`, `l3.spz`, `l3.sog` (best-effort — `astel_splat_io.write_sog` works
on the stub cloud via uniform-quantile codebooks/no spatial sort; higher
quantization error than reference k-means SOGS, documented inline, never
silently swallowed), `package.astel` (via
`astel_format.builder.build_minimal_package`, with a fully-typed
`QualityReport` — every metric explicit `None` + a reason, e.g.
`ci_method="stub-no-estimate"` for the degenerate `scale_confidence` interval
the JSON Schema requires to be positive), and the existing
`quality-report.json` (`astel.quality-report/v0`, Truth Meter's illustrative
dict — kept as a **second, deliberately distinct** report shape; the two are
not conflated).

New `POST /v1/captures` (multipart `UploadFile`) stores raw bytes under a
`capture-<uuid>` namespace in the existing `ArtifactStore` (member name always
`source<ext>` with a sanitized extension — no traversal surface from user
filenames), returns a `CaptureRef`. `CreateGenerationRequest.capture_id`
(optional) threads through to a new nullable `Generation.capture_id` column.
Web: `GenerationDock` now uploads a dropped Image/Video file to `/v1/captures`
before submitting, shows inline upload status, and passes the resulting
`capture_id` into `start()` → the generation request.

**Gates** (re-verified by founder after subagents reported):
api ruff/mypy --strict/pytest 25+1skip; web eslint+tsc+vitest 18 (was 15);
astel_format 16; astel_splat_io 11; astel_eval 36 (~4s).

### Honest gaps carried forward
- **Captures are uploaded but not consumed.** The producer still emits the
  procedural stub splat regardless of an attached image/video. Consuming the
  capture (real reconstruction) is the M2 GPU path.
- **No DB migrations** — `init_db` is `create_all`; the new `capture_id`
  column requires a fresh SQLite file (gitignored dev/test DBs already
  regenerate). A real migration tool is owed before the schema changes again
  with persistent data at stake.
- `.sog` remains best-effort/partial per `astel_splat_io`'s own docs — not a
  new gap, just still open.
- Capture upload flow was verified via automated tests (api integration test +
  new vitest test with mocked fetch), not a live browser round-trip — the
  next live-browser session should exercise an Image-modality drop end-to-end.

## Next (session 6, still no GPU required unless founder green-lights scaling)

- Live browser pass: drag-drop an image, confirm `/v1/captures` round-trip and
  `.astel`/`.spz`/`.sog` artifacts list/serve correctly from a running `astel up`.
- Decide and stub a DB migration story (Alembic) before the schema grows again.
- When founder is ready: GPU smoke tests on the 2x4090 box (CUDA-in-WSL sanity
  → gsplat reference train → MapAnything orbit test → TRELLIS import check →
  R-T1 distillation experiment) — this is the actual M2 capture-path start.
