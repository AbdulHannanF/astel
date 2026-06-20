# MVP Testing Guide (build plan M0–M6 complete)

> **Read this first if you want to test what Astel can do today.** It answers
> the one question that matters for a hands-on test: *"If I type a text prompt,
> do I get a model of what I described?"* — and tells you how to drive each path.

_Gate counts last re-verified: 2026-06-19 (session 29, end of M6). Text→model
verified live on Box A (session 22). All gates green — see [§ Gates](#gates)._

---

## TL;DR — can I give it a text prompt and get a model?

**Yes — on the GPU box.** As of session 22 the text→3D path is wired and
verified end-to-end:

| Input | What you get today | Is the geometry *yours*? |
|---|---|---|
| **Text** prompt | A **real generated 3D Gaussian asset** of what you described (text → SDXL/FLUX reference image → TripoSplat L2 → 2DGS L3), incl. the intermediate `text-reference.png` you can inspect | ✅ **Yes** — on the GPU box (`ASTEL_PRODUCER=gpu`). On the CPU stub it's still a placeholder. |
| **Single image** | A **real generated 3D Gaussian asset** of the object in the image (image → TripoSplat L2 → 2DGS L3) | ✅ **Yes** — on the GPU box. |
| **Video** | Placeholder preview only (the upload is stored, not yet reconstructed) | ❌ Not wired through the product. M6 added a CLI static-reconstruction path (`_produce_video`, runs when handed a frame), but the API does not yet extract/pass a video frame, so an uploaded video still yields the placeholder, flagged `conditioning: "none"`. End-to-end video recon = roadmap G1. |

**How text→3D works:** your prompt is wrapped into a single-object studio-shot
prompt → a local text-to-image model renders a reference image → TripoSplat
turns that image into 3D Gaussians (it does its own background removal) → 2DGS
refines the surface → the full layered `.astel` asset is produced. The default
image model is **SDXL base 1.0** (open, no Hugging Face login, no spend); set
`ASTEL_T2I_MODEL=black-forest-labs/FLUX.1-schnell` to use the Apache-licensed
FLUX upgrade (needs one free `hf auth login`).

**Honesty:** a generated asset has no real-world ground truth, so the quality
report keeps `geometric_error`/`scale` null and `generated_ratio: 1.0`, and the
API response carries a structured `conditioning` field (`prompt` / `image` /
`video` / `none`) so you always know what the geometry was actually derived
from. The CPU stub (no GPU) still returns a placeholder, reported honestly as
`conditioning: "none"`.

On the **default CPU stub** (any box, no GPU) every modality returns the
procedural placeholder — good for exercising the UI/SSE/billing plumbing. For
real generated models, use the GPU box per §2.

---

## 1. Run the app — one command

```powershell
# from the repo root
pnpm install          # first time only
pnpm run up           # boots API (FastAPI) + web (Vite) together
# open http://localhost:5173
```

`pnpm run up` **auto-detects the GPU**: on the 2×4090 box (nvidia-smi +
`pipelines/gpu/.venv` + `run-python.cmd` all present) it runs the **real
generative producer** (`ASTEL_PRODUCER=gpu`), so every Text/Image generation is
a real, prompt-conditioned splat. On a box with no GPU it falls back to the CPU
**stub** (procedural placeholder geometry) — good for exercising the UI, layer
inspector, Truth Meter, SSE progress, and billing without a GPU. Force either
path with `pnpm run up -- -Gpu` or `pnpm run up -- -Stub`.

Generation is **asynchronous**: clicking Generate returns immediately and the
Layer Stack streams **real** per-stage progress over SSE while the job runs in
the background (no blocked request, no fake replay). When it finishes the viewer
loads the per-task `l3.ply` and the Truth Meter shows the honest origin pill
(**GENERATED** on the GPU path, **STUB** on the CPU fallback) with provenance
and caveats; the credit ledger prices the run (preview vs refine).

- The dock shows an honest one-line hint under each modality tab telling you
  what that input actually produces today.
- `pnpm run up -- -Temporal` uses the durable Temporal engine instead of the
  in-process async job engine (optional; needs the Temporal dev server).
- `pnpm run up -- -BindHost 0.0.0.0` exposes the stack on the LAN so a laptop can
  drive generation on this box's 4090s — see
  [REMOTE_ACCESS.md](REMOTE_ACCESS.md).

## 2. The real generative path, by hand (GPU box, `THREADRIPPER-48`)

On the box, `pnpm run up` already runs the real producer (§1) — just use the web
app's **Text** or **Image** tab. To drive the producer directly (no web app),
which is how spikes/CI run it — note it **must** go through `run-python.cmd` so
gsplat's JIT compiler has the MSVC build env:

```powershell
cd pipelines\gpu
.\run-python.cmd -m astel_gpu.produce `
  --task-id demo-001 --modality image `
  --image external\TripoSplat\static\example_inputs\creature_butterfly.webp `
  --out .\.demo-out --refine-iters 1500
```

You'll get the full artifact contract in `.demo-out\`: `l0.ply`, `l2.ply`,
`l3.ply`, `l3.spz`, `l3.sog`, `package.astel`, `quality-report.json`, plus a
`l2l3-metrics.json` sidecar. The quality report's PSNR is **held-out
self-consistency / distillation fidelity** (a generated object has no real
scan), and `geometric_error`/`scale` are honestly `None` — never faked.

## 3. The text path's structured spec (optional, founder-gated)

The text pipeline runs prompt → **Generation Spec** (object class, parts,
materials, target scale w/ confidence). It is **offline by default** and never
spends:

- Default: replays cached fixtures from `ASTEL_LLM_FIXTURES_DIR`. With no
  fixture for your prompt it writes an honest `"skipped"` note (this is what you
  see out-of-the-box, since no fixtures ship).
- Live (real Anthropic calls, ~$0.02–0.035/gen): set **both**
  `ASTEL_LLM_LIVE=1` **and** `ANTHROPIC_API_KEY`. A key alone never triggers
  spend — both gates are required (founder gate R-O2).

This spec does **not** drive geometry today; it threads an LLM **size estimate**
into the Truth Meter's scale field (honestly flagged `method: llm-estimate`).

## 4. Quality / billing you can inspect per generation

Every generation writes, into its artifact dir:
- `quality-report.json` — the Truth Meter feed (origin, provenance %, caveats).
- `credit-ledger.json` — per-layer credits (preview L0–L2 cheap, L3 the main
  spend); `GET /v1/pricing` publishes the schedule.
- `generation-spec.json` — the text-path spec (or a `"skipped"` note).

## Gates

Run these to re-verify the build is green (all pass as re-run 2026-06-19):

| Component | Command (from its dir) | Result |
|---|---|---|
| API | `uv run ruff check . && uv run mypy && uv run pytest -q` | ruff ✓ · mypy 27 ✓ · **72 passed, 1 skipped** |
| Web | `pnpm test && pnpm run lint` (lint = `eslint . && tsc -b`) | **73 passed** ✓ · eslint ✓ · tsc -b ✓ |
| `@astel/manifest` | `pnpm test && pnpm run build` | **15 passed** ✓ |
| `@astel/sdk` | `pnpm test && pnpm run build` | **9 passed** ✓ |
| libs (`astel_*`, ×9) | `uv run ruff check . && uv run mypy && uv run pytest -q` in each | **342 passed** (appearance 25 · dynamics 40 · eval 36 · format 34 · llm 24 · lod 53 · scene 56 · solid 37 · splat_io 37) |
| GPU pipeline | `uv run ruff check . && uv run mypy && uv run pytest -q` | ruff ✓ · mypy 43 ✓ · **112 passed, 3 skipped** |
| `astel-sdk` (Python) | `uv run ruff check . && uv run mypy && uv run pytest -q` | ruff ✓ · mypy ✓ · **11 passed** |
| `tools/loadtest` | `uv run ruff check . && uv run mypy && uv run python load_test.py --self-test` | ruff ✓ · mypy ✓ · self-test ✓ |

The 3 skipped GPU tests are environmental: 2 run a real gsplat kernel (they skip
cleanly unless launched through `run-python.cmd` on a CUDA box, so a plain
`uv run pytest` is green everywhere) and 1 needs cached FLUX.1-schnell weights.
Run the kernel tests for real with:
`.\run-python.cmd -m pytest tests\test_smoke_refit.py tests\test_synthetic_eval.py`.

> There is no CI runner yet (the top launch blocker) — these gates are run by
> hand. See [LAUNCH_CHECKLIST.md](LAUNCH_CHECKLIST.md).
