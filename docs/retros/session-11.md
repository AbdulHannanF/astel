# Session 11 retro (2026-06-15)

**M3 (generative path) entered.** Cleared the first two ordered, gated steps from
[13-m3-readiness](../research/13-m3-readiness.md) §4 with measured results:
TripoSplat triaged (GO) and proven to run natively on Box A. No founder gate was
touched — no Anthropic API key used, no spend incurred.

Mode: Opus planning/review; one Sonnet agent for the triage audit, one for the
install spike (per the founder's model directive). Each agent's output was
reviewed before adoption.

## 1. Step 1 — TripoSplat triage (no-GPU), GO

Import-graph + license audit in [14-triposplat-triage](../research/14-triposplat-triage.md),
matching audit 12's method (shallow clone to `%TEMP%`, static import trace, deleted
after). Findings: `VAST-AI-Research/TripoSplat` live, **MIT code AND weights**
(confirmed in-repo, not just an HF tag); entire codebase 4 files / ~2.5k LOC;
**zero** NC/build-heavy deps — no nvdiffrast/kaolin/spconv/flash-attn/xformers/
pytorch3d anywhere. Only non-pure-Python op is `torchvision.ops.deform_conv2d`
(precompiled in the torchvision wheel). Single image → native 3D gaussians
(≤262,144, learned adaptive density), `.ply`/`.splat` export. Cleanest candidate
audited to date.

## 2. Step 2 — Windows install spike on Box A, PASS

The real feasibility gate. TripoSplat runs natively:
- `torchvision` added from the **existing cu128 index** → `0.26.0+cu128` (matches
  torch 2.11); plus `safetensors`/`tqdm`/`huggingface-hub`. **No CUDA build, no
  vcvars, no flash-attn/xformers** — attention is plain `F.scaled_dot_product_attention`.
- Weights (~3.6 GB) → gitignored `pipelines/gpu/models/triposplat`; repo vendored to
  gitignored `pipelines/gpu/external/TripoSplat`. `.gitignore` updated.
- `triposplat_spike.py` (new, under `astel_gpu`) ran one inference via the
  `run-python.cmd` launcher.

**Measured (building_stone_house example, 65,536 gaussians, 20 steps):**
wall-time **11.4 s**, peak VRAM **4.6 GB**, `l2.ply` 65,536 verts, finite/sane XYZ
bounds. Far under the 24 GB ceiling → headroom for a co-resident L3 refine.

## 3. What the numbers say

- **R-T9 resolved for TripoSplat** (no Windows generative-dep hell — it needs none).
- **R-T1 strongly de-risked**: the native gaussian generator replaces the
  TRELLIS.2-mesh→surfel distillation as the planned L2 prior, pending the bake-off.
- **R-T7 moot for TripoSplat** at 4.6 GB; the ceiling risk is now specific to the
  TRELLIS.2 fallback.
- Decision recorded: **TripoSplat is the lead L2 candidate**, to be formally
  confirmed by the step-3 bake-off (held-out-view PSNR/SSIM/LPIPS + blind corpus),
  which resolves DECISIONS #2.

## 4. Honest gaps / carried forward

- **Real defect found** (not a blocker): TripoSplat's own `Gaussian.save_ply` writes
  non-finite opacity for ~11% of points (`log(x/(1-x))` saturates at fp16 `x==1.0`).
  The production L2 wrapper must clamp opacity before export. The spike `.ply` carries
  the `inf`s as produced — documented, not silently fixed.
- `triposplat_spike.py` is a spike (uses upstream `save_ply`, no opacity sanitising,
  no unit test). It graduates into a typed, tested `l2_triposplat` module in the
  bake-off step — *not* yet wired into `astel_gpu.produce` or the API.
- `huggingface_hub` hf-xet path hung on the two largest checkpoints here;
  `HF_HUB_DISABLE_XET=1` is the reliable download path on this box.
- **The L3 surface-aligned A/B (M2 carryover) is still open** — session 10's headline
  next step. It now also gates M3 step 4 (L2→L3 wiring). Sequence it before/with the
  bake-off's L3 half.
- **Still nothing committed** — sessions 7–11 GPU work remains in the working tree on
  top of the single "Beta" commit. Flagged again; awaiting go-ahead to commit.

## 5. Next (session 12)

(a) **L2 bake-off** (M3 step 3): graduate the spike into a typed `l2_triposplat`
wrapper (opacity-clamped, astel_splat_io-clean), score TripoSplat on held-out views
(PSNR/SSIM/LPIPS) + the blind corpus; add TRELLIS-v1 head only if a comparison point
is needed. (b) **L3 A/B** (M2 carryover) — 2DGS vs 3DGS+GOF on DTU scan1, beat
8.73 mm overall. (c) Then **L2→L3 wiring** (M3 step 4). (d) **Generation Spec stage**
(M3 step 5) once the founder provides an **Anthropic API key + spend cap** (~$0.02–0.08/gen,
under the $1k/mo flag) — adapter built on cached fixtures first regardless.
