# MVP Testing Guide (end of M3)

> **Read this first if you want to test what Astel can do today.** It answers
> the one question that matters for a hands-on test: *"If I type a text prompt,
> do I get a model of what I described?"* — and tells you how to drive each path.

_Last verified: 2026-06-15 (session 22). Text→model verified live on Box A. All
gates green — see [§ Gates](#gates)._

---

## TL;DR — can I give it a text prompt and get a model?

**Yes — on the GPU box.** As of session 22 the text→3D path is wired and
verified end-to-end:

| Input | What you get today | Is the geometry *yours*? |
|---|---|---|
| **Text** prompt | A **real generated 3D Gaussian asset** of what you described (text → SDXL/FLUX reference image → TripoSplat L2 → 2DGS L3), incl. the intermediate `text-reference.png` you can inspect | ✅ **Yes** — on the GPU box (`ASTEL_PRODUCER=gpu`). On the CPU stub it's still a placeholder. |
| **Single image** | A **real generated 3D Gaussian asset** of the object in the image (image → TripoSplat L2 → 2DGS L3) | ✅ **Yes** — on the GPU box. |
| **Video** | Placeholder preview only (the upload is stored, not yet reconstructed) | ❌ Not wired (M6). Flagged `conditioning: "none"`. |

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

## 1. Run the app (CPU, default — works on any box)

This is the full product surface with the **stub** producer. Good for testing
the UI, the layer inspector, the Truth Meter honesty channel, SSE progress, and
the billing/credit ledger. The geometry is the procedural placeholder.

```powershell
# from the repo root
pnpm install          # first time only
pnpm run up           # boots API (FastAPI) + web (Vite) together
# open http://localhost:5173
```

- Type anything in the **Text** tab → Generate → watch L0→L3 stream → the
  viewer loads the per-task `l3.ply`, the Truth Meter shows a **STUB** pill and
  a caveat saying the geometry is a placeholder, and the credit ledger prices
  the run (preview vs refine).
- The dock now shows an honest one-line hint under each modality tab telling you
  what that input actually produces today.

`pnpm run up -Temporal` uses the durable Temporal engine instead of the
in-process stub engine (optional; needs the Temporal dev server).

## 2. Run the **real** generative path: image → 3D (GPU box, `THREADRIPPER-48`)

This produces a genuine generated Gaussian asset from a single image.

```powershell
# 1) point the API at the GPU producer (subprocess; keeps torch out of the API env)
$env:ASTEL_PRODUCER = "gpu"
pnpm run up

# 2) in the web app, switch to the Image tab, drop a single clean object photo,
#    and Generate. The API resolves your upload and runs:
#       image -> TripoSplat L2 (native gaussians) -> 2DGS L3 (surfel refine)
```

Or drive the GPU producer directly (no web app), which is how CI/spikes run it —
note it **must** go through `run-python.cmd` so gsplat's JIT compiler has the
MSVC build env:

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

Run these to re-verify the build is green (all pass as of 2026-06-15):

| Component | Command (from its dir) | Result |
|---|---|---|
| API | `uv run ruff check . && uv run mypy && uv run pytest -q` | ruff ✓ · mypy 23 ✓ · **51 passed, 1 skipped** |
| Web | `pnpm test && pnpm exec tsc --noEmit && pnpm lint` | **18 passed** ✓ · tsc ✓ · lint ✓ |
| `@astel/manifest` | `pnpm test && pnpm typecheck && pnpm lint` | **10 passed** ✓ |
| libs (`astel_*`) | `uv run pytest -q` in each | **87 passed** (14+10+16+11+36) |
| GPU pipeline | `uv run ruff check src tests && uv run mypy src tests && uv run pytest -q` | ruff ✓ · mypy 34 ✓ · **55 passed, 2 skipped** |

The 2 skipped GPU tests run a real gsplat kernel; they skip cleanly unless
launched through `run-python.cmd` on a CUDA box (so a plain `uv run pytest` is
green everywhere). Run them for real with:
`.\run-python.cmd -m pytest tests\test_smoke_refit.py tests\test_synthetic_eval.py`.
