# 13 — M3 Readiness (generative path) — live-verified 2026-06-14

*Prep for M3 (text/image → L2→L3 generative splats). External state re-verified
live this session per CLAUDE.md §10.1 (training knowledge is 5 months stale).
Builds on [12 — TRELLIS import audit](12-trellis-import-audit.md) and
[RISKS.md](RISKS.md). No spend incurred; no API key used.*

## 0. TL;DR for the next session

- **TRELLIS.2-4B** is current (MIT, but outputs **mesh + PBR**, not splats; ≥24 GB VRAM).
- **NEW since cutoff: TripoSplat** (VAST-AI / Tripo) — single image → **native 3D
  Gaussians**, MIT, weights on HF, ~2k LOC, reportedly beats TRELLIS.2 in blind
  studies. **This may de-risk R-T1** (our riskiest bet) by replacing the
  TRELLIS.2-mesh→surfel distillation with a direct gaussian generator.
- **First M3 task = a 3-way bake-off** (TripoSplat vs TRELLIS-v1 gaussian head vs
  TRELLIS.2-mesh-distill) on our now-built DTU eval harness — pick the L2 prior by
  measured Chamfer/PSNR, not vibes.
- **Two founder gates** before M3 LLM work: (a) an **Anthropic API key + small
  budget** (est. below — well under the $1k/mo flag), (b) confirm which generative
  prior to commit to after the bake-off.

## 1. Verified external state (June 2026)

| Model | Output | License | Notes (live-checked) |
|---|---|---|---|
| **TRELLIS.2-4B** (microsoft) | **Mesh + PBR** (O-Voxel) | MIT | Current (Dec 2025). 512³≈3 s / 1024³≈17 s on H100. **≥24 GB VRAM** (tested A100/H100). Weights on HF. Our planned use = distill its geometry prior to surfels (R-T1). |
| **TRELLIS v1** (microsoft) | **Gaussian head** | MIT | `TrellisImageTo3DPipeline` → `outputs['gaussian']` is nvdiffrast-clean (audit 12). The original L2 plan. |
| **TripoSplat** (VAST-AI) ⭐NEW | **Native 3D Gaussians** (≤262k, learned density control) | MIT | Weights `VAST-AI/TripoSplat` on HF; code on GitHub; ~2k LOC. Reportedly Elo 1137 > TRELLIS.2 992 in a blind study. ComfyUI-native. **Outputs our product representation directly.** |
| Hunyuan3D 2.1 | Mesh/textured | (verify) | Cited as a strong self-host option; not splat-native. |
| UniLat3D | — | (verify) | Mentioned as a lower-ranked alternative. |

**Windows feasibility (the practical run-it-on-Box-A risk).** Community-proven:
`sdbds/TRELLIS-for-windows` (PowerShell one-click) and `ComfyUI_TRELLIS` exist;
prebuilt **flash-attn** Windows wheels are published (e.g. bdashore3 releases),
`xformers` is the documented fallback (`ATTN_BACKEND=xformers`), and `spconv`
supports `SPCONV_ALGO=native`. **Caveat:** those prebuilts target cu124 / torch
2.5 / py3.10, while Box A runs **cu128 / torch 2.11 / py3.12** — so we either
find matching wheels or build the CUDA deps ourselves via the existing
`run-python.cmd` (vcvars) launcher, same as we did for gsplat. nvdiffrast/nvdiffrec
are **NC and avoided** per audit 12 (we don't need TRELLIS.2's texturing stage).

## 2. The generative-prior decision (L2 → L3)

DECISIONS.md #2 picked TRELLIS.2 O-Voxel → surfel distillation (R-T1, severity H).
TripoSplat changes the option space:

- **Option A — TRELLIS v1 gaussian head.** Clean MIT, proven, splat-native but
  3DGS (volumetric, no surface normals). Original plan.
- **Option B — TripoSplat (NEW).** Native gaussians, MIT, reportedly SOTA,
  lightweight. **Could replace R-T1's distillation entirely** as the L2 feed-forward
  generator. Still 3DGS, so L3 surface-alignment (2DGS/PGSR) still needed on top.
- **Option C — TRELLIS.2 O-Voxel mesh prior → distill to surfels.** Highest
  geometry fidelity potential, but the riskiest (R-T1) and needs the 1-line
  `o_voxel` fork patch (audit 12) + ≥24 GB.

**Recommendation:** keep C's *mesh prior as L3 geometry supervision*, but evaluate
**B (TripoSplat) as the L2 generator first** — if it lands within tolerance of the
DTU/synthetic Chamfer baselines, it de-risks R-T1 and shortcuts the generative
path. We can now measure all three quantitatively: the **`capture_eval`/synthetic
harness built in sessions 8–10 gives Chamfer (mm) + PSNR**, so the bake-off is a
real measurement, not a judgement call. (Generated objects have no GT scan, so
compare against held-out views (PSNR/SSIM/LPIPS) + the blind-eval corpus, and use
DTU/synthetic only for the surface-fitting half of the pipeline.)

## 3. Generation Spec LLM stage — cost (verified pricing, no key used)

The LLM layer (CLAUDE.md §5): prompt → structured Generation Spec, L6
physics-material reasoning, QA critique, user explanations. Budget target:
**<10–20k tokens/generation.**

Current pricing (per 1M tokens): **Haiku 4.5 `claude-haiku-4-5` $1 in / $5 out**
(200K ctx, supports structured JSON output + prompt caching); Sonnet 4.6
`claude-sonnet-4-6` $3/$15; Opus 4.8 `claude-opus-4-8` $5/$25.

**Plan:** Haiku 4.5 as the default for the structured Generation Spec (it's
constrained extraction, not deep reasoning) via `output_config.format` (JSON
schema); reserve Sonnet 4.6 for L6 physics reasoning only if Haiku underperforms.
**Prompt-cache** the spec schema + system prompt (1.25× write, ~0.1× read) since
it's identical across generations.

**Cost estimate** (≈15k tokens/gen, ~10k in / 5k out, system prompt cached):
- All-Haiku: ~**$0.02–0.035 / generation** (input mostly cache-read).
- With Sonnet for L6: ~$0.05–0.08 / generation.
- Monthly: ~**$50–350/mo at 1k–10k generations** — **under the $1k/mo flag**
  (CLAUDE.md §10.2) at expected closed-beta volumes.

**Founder gate (R-O2):** M3 LLM work needs an **Anthropic API key + a small spend
cap**. Cost is modest but real; per the agreement, no paid call is made until the
key + budget are approved. The adapter is built against cached fixtures first, so
all non-LLM M3 work proceeds without it.

## 4. First-M3-session plan (ordered, gated)

1. **TripoSplat triage (no GPU):** clone, import-graph + license audit (like
   audit 12), confirm MIT + nvdiffrast-clean, check torch/cu/py compat with Box A.
2. **Windows install spike:** get ONE of {TripoSplat, TRELLIS v1 gaussian head}
   importing + a single inference on Box A (flash-attn wheel or vcvars build;
   xformers fallback). This is the real feasibility gate.
3. **L2 bake-off:** run the chosen candidates on a small image set; score with the
   existing eval harness (held-out PSNR/SSIM/LPIPS) + the blind corpus; pick the L2
   prior. Resolves DECISIONS #2 on real output.
4. **L3 wiring:** feed the L2 gaussians into the session-10 surface-aligned L3
   refinement (once that A/B is also done) → measure on DTU.
5. **Generation Spec stage** (after API key): prompt→spec JSON via Haiku +
   structured outputs + caching; cached-fixture tests first; log token cost into
   the credit ledger.

Gates each step: ruff · mypy · pytest green; measured metrics logged; honest report.

## 5. Risk updates

- **R-T1 (distillation bet, H):** *potentially de-risked* — TripoSplat's native
  gaussian generator is a fallback/replacement for the TRELLIS.2-mesh→surfel
  distillation. Confirm by bake-off (step 3). Update RISKS.md after.
- **R-T7 (24 GB VRAM, M):** TRELLIS.2-4B needs **≥24 GB** = exactly the 4090
  ceiling → tight, one model per GPU, no co-resident pipeline. TripoSplat (~2k LOC,
  ≤262k gaussians) is far lighter — another reason to prefer it on Box A.
- **NEW R-T9 (Windows generative deps, M):** flash-attn/spconv/kaolin prebuilts
  don't match our cu128/torch2.11/py3.12; mitigation = vcvars build (proven for
  gsplat) or xformers fallback. Validate in step 2 before committing.
- **R-L1 (NC deps, contained):** unchanged — nvdiffrast/nvdiffrec NC, avoided;
  TRELLIS v1/v2 MIT; TripoSplat MIT (re-confirm in step 1). T&T/CO3D capture
  datasets stay non-commercial (session 9) — eval-only, never shipped.

## 6. Founder decision points (surface before M3 execution)

1. **Anthropic API key + spend cap** for the Generation Spec/L6 LLM stage (cost
   §3; modest, under the $1k/mo flag, but real — needs your key).
2. **Generative-prior commitment** — recommend "evaluate TripoSplat first, then
   decide" rather than pre-committing to the TRELLIS.2 distillation. Confirm OK.
3. No new licensing exposure so far (all candidate models MIT) — flag only if the
   bake-off forces a non-permissive option (none currently).
