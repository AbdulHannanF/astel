# Session 22 retro (2026-06-15)

**Text→3D SHIPPED and verified live — the headline modality that was silently
missing now works end-to-end — plus a full "nothing-unplugged" audit, honesty
hardening, and a physics bug fix.** Triggered by the founder catching that text
prompts produced a placeholder, not a model.

Orchestration (founder's directive): **Opus planned + reviewed + did the
cross-cutting/critical work; Sonnet subagents did the bulk implementation and
the read-only audits** — for token efficiency.

## 1. The headline: text → real 3D model

New `astel_gpu.text_to_image` stage + `produce.py` text dispatch:
**prompt → canonicalized single-object prompt → local text-to-image model →
TripoSplat L2 (it does its own background removal) → 2DGS L3 → full `.astel`
layer stack.** Loaded via `diffusers.AutoPipelineForText2Image` (no custom CUDA
build).

- **Model decision (verified live, §10.1):** default
  `stabilityai/stable-diffusion-xl-base-1.0` — **open access (no HF login), CreativeML-OpenRAIL++-M (commercial OK), ~7 GB**, so **text→model runs fully
  local with zero founder credentials and zero spend** (honors §1 local-first).
  Opt-in upgrade `black-forest-labs/FLUX.1-schnell` (Apache-2.0, the cleanest
  license, but HF-gated → one free `hf auth login`) via `ASTEL_T2I_MODEL`.
  FLUX was the first pick but its gated repo would have re-introduced a founder
  gate — the non-gated SDXL default removes that.
- **Verified live on Box A** (prompt "a worn brass astrolabe on a wooden base",
  `--refine-iters 300`): SDXL rendered a clean astrolabe `text-reference.png`
  (download ~11 min one-time, gen ~5 s) → TripoSplat L2 65,536 gaussians (11.3 s,
  4.59 GB, 0 non-finite) → 2DGS L3 (4.2 s, held-out self-consistency **22.55 dB**)
  → L5 solid → 12-artifact contract incl. `text-reference.png`, `l0/l2/l3.ply`,
  `.spz`/`.sog`, `package.astel`, `l5.stl`. The intermediate reference image is
  kept as an artifact for transparency.

## 2. Audits — so a missed-headline gap can't hide again

- **Wiring audit** ([doc 15](../research/15-pipeline-wiring-audit.md), Sonnet):
  a full modality×producer×layer matrix. Found the **same silent-fallback class
  twice more**: video also aliased to the text-smoke path (with a copy-paste
  caveat literally saying "text modality" on a video task), and — worst — the
  stub SSE engine **always streamed "Asset ready" with hardcoded metrics even
  when production produced zero artifacts**. Root cause of the original gap: the
  honesty signal lived only in prose `caveats`, with no structured field a caller
  could check.
- **Dead-code audit** ([doc 16](../research/16-dead-code-audit.md), Sonnet):
  **deleted** `triposplat_spike.py` (graduated to `l2_triposplat.py`) and
  `experiments/task-engine-spike/` (graduated into `services/api/.../temporal/`).
  Confirmed the apparent dups are deliberate (`stable_seed` duplicated so the API
  never imports torch; the three `build_quality_report` variants each encode a
  different honesty story). No commented-out/unreachable code.

## 3. Honesty hardening (Sonnet subagent, services/api + apps/web)

- **SSE now reflects the REAL outcome:** `Generation` gained
  `produced`/`splats`/`production_error` (+ migration `d4e5f6a7b8c9`); the stub
  engine emits **FAILED** (not a fake "Asset ready") when production wrote
  nothing, and reports the real splat count on success.
- **Structured `conditioning` field** on `GenerationResource`
  (`prompt`/`image`/`video`/`none`) — the guard that would have made the text
  gap visible in the API response without reading prose. Persisted + returned.
- Producer-dispatch logging (warns on `ASTEL_PRODUCER` misconfig); web shows an
  honest conditioning badge + real progress-rail metrics; billing dead-config
  CI test (every `_ARTIFACT_LAYER` key is produced or explicitly tracked in a
  `_NOT_YET_IMPLEMENTED` allowlist).

## 4. Physics bug fixed (Opus)

Inspecting the astrolabe's L5 output caught **negative principal moments of
inertia** (physically impossible). Cause: `compute_mass_properties` computed
inertia about the origin then subtracted the parallel-axis term — a difference
of two large near-equal matrices that loses all significance (and goes negative
under the marching-cubes bias) when the COM sits far from the origin. Session 19's
centred pirate-ship masked it. **Fix:** recenter the mesh to its COM before the
second-moment integral (inertia computed directly in the COM frame, no
cancellation). Locked with a far-from-origin regression test.

## 5. Gates — all green

API ruff·mypy(25)·**59 pytest**+1skip · web **22 vitest**·tsc·lint ·
`@astel/manifest` **10** · libs **98** (24 llm + **11** solid + 16 format + 11
splat_io + 36 eval) · GPU pipeline ruff·mypy(36)·**60 pytest**+3skip.

## 6. Honest gaps / tracked follow-ups

- **Typed-package `origin` enum (audit §2.4) — NOT done, tracked.** The
  manifest-v0 `QualityReport` still has no structured `origin`; adding it is a
  *versioned schema change* (`additionalProperties:false`) feeding the M5 engine
  plugins, so it wasn't rushed. The today-relevant guard (the API `conditioning`
  field) is shipped; the v0 quality-report dict still labels generated assets
  `origin:"measured"` (corrected by caveats + `generated_ratio:1.0`) — flip to a
  proper `stub`/`generated`/`measured` taxonomy together with the typed enum and
  the web pill in one coherent pass.
- Text→3D quality: 300 refine iters here for speed (22.5 dB); the hero budget is
  ~1500. No densification yet. The reference image is single-view (TripoSplat is
  single-image) — a true multi-view text→MV stage is a later upgrade.
- L1/L4/L7 still unimplemented (tracked in the billing dead-config allowlist).

## 7. Why the gap happened (the real post-mortem)

M3 built the generative *engine* (image-first: TripoSplat) and the text
*Generation Spec* (metadata), but never the text→image bridge between them — and
that gap was recorded only as a buried retro caveat, never as a structured
signal or a flashing status. The fixes that make recurrence hard: the wiring
matrix doc, the structured `conditioning` field, the SSE-reflects-outcome change,
and the billing dead-config CI test.

## 8. Next

M4 world-awareness on the original plan: join L6→L5 for real per-region mass
(density × solid volume), bind L6 into the `.astel` manifest, the origin-enum
taxonomy pass, then L4 relighting, metric-scale L5, CoACD + `.3mf` + printability.
