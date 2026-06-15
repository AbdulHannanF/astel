# 14 — TripoSplat Triage (M3 step 1, no-GPU session, 2026-06-14)

*Method: shallow-cloned `VAST-AI-Research/TripoSplat` (current `main`, confirmed live
on GitHub today) into `%TEMP%\astel-audit\`, statically traced every `import`/`from`
across all four Python files (the entire codebase) for `nvdiffrast`, `nvdiffrec`,
`kaolin`, `flexicubes`, `diso`, `diffoctreerast`, `spconv`, `flash_attn`, `xformers`,
`pytorch3d`. Verified license, repo identity, and weights location live via web
search/fetch (training data ~5 months stale per CLAUDE.md §10.1). Clone deleted
after analysis — nothing persisted outside `%TEMP%`. No weights downloaded
(3.78 GB on HF — inspected metadata/tree only).*

## 0. TL;DR

**Go for the install spike (M3 step 2).** TripoSplat is exactly as advertised in
doc 13: a ~2,500-LOC, two-file model with **zero** of the flagged NC/build-heavy
dependencies — not even `transformers` or `diffusers`. The only non-pure-Python op
is `torchvision.ops.deform_conv2d`, which ships precompiled in every standard
torchvision wheel (no vcvars CUDA build needed). MIT code + MIT weights, confirmed
live. This is the cleanest dependency profile of any candidate audited so far
(cleaner than TRELLIS v1's gaussian-head path in doc 12, which still required
avoiding two tainted modules).

## 1. Repo identity & license (live-checked 2026-06-14)

- **Code repo**: `github.com/VAST-AI-Research/TripoSplat` — live, default branch
  `main`, last commit `a78fa12d` dated **2026-06-02** (12 days old at audit time).
  Note: org is `VAST-AI-Research` (not `VAST-AI`) for the code repo; weights are
  under `VAST-AI/TripoSplat` on HF (and mirrored as `VAST-AI-Research/TripoSplat`
  on ModelScope) — doc 13's "VAST-AI / Tripo" framing is correct, just split across
  two org namespaces (common VAST-AI pattern, cf. `VAST-AI/TripoSG`).
- **LICENSE file** (top of repo, read directly): MIT License, "Copyright (c) 2026
  VAST". ✅
- **README** explicitly states: *"TripoSplat code and weight models are released
  under the MIT License"* — code AND weights both MIT, confirmed in-repo, not just
  inferred from an HF model-card tag.
- **Weights**: `huggingface.co/VAST-AI/TripoSplat`, total **3.78 GB**, organized as
  `background_removal/`, `clip_vision/`, `diffusion_models/`, `vae/` — matches the
  five `.safetensors` checkpoints referenced in code (`birefnet`,
  `triposplat_fp16`, `dino_v3_vit_h`, `flux2-vae`, `triposplat_vae_decoder_fp16`).
- **Paper**: arXiv 2605.16355, "Generative 3D Gaussians with Learned Density
  Control" (Yan, Cao, Wang, Liang, Guo — TripoAI/VAST, 2026).
- **Nothing has materially changed** since doc 13 — repo is live, actively
  committed (12 days ago), MIT confirmed both code and weights. No newer
  splat-native generator surfaced in this search beyond what doc 13 already flagged.
  Proceeding with full audit (not stopping early).

## 2. Import-graph audit — the entire codebase

The whole repo is **4 Python files, 2,523 LOC total** (`model.py` 1,725,
`triposplat.py` 599, `run_example.py` 39, `run_gradio.py` 160) plus static
assets/viewer HTML. Every top-level import, across all four files:

| File | Imports |
|---|---|
| `model.py` (1,725 LOC) | `typing`, `math`, `re`, `numpy`, `safetensors.torch`, `torch`, `torch.nn`, `torch.nn.functional`, `torchvision.ops.deform_conv2d` |
| `triposplat.py` (599 LOC) | `numpy`, `torch`, `torch.nn.functional`, `safetensors.torch`, `PIL.Image`/`ImageFilter`, `torchvision.transforms`, `tqdm.auto`, `model` (local) |
| `run_example.py` (39 LOC) | `triposplat.TripoSplatPipeline` (local) |
| `run_gradio.py` (160 LOC) | `time`, `pathlib`, `uuid`, `gradio`, `torch`, `triposplat.TripoSplatPipeline` (local) |

**Zero matches** for `nvdiffrast`, `nvdiffrec`, `kaolin`, `flexicubes`, `diso`,
`diffoctreerast`, `spconv`, `flash_attn`, `xformers`, `pytorch3d`, `cumesh`,
`flex_gemm` — confirms doc 13's "nvdiffrast-clean" claim and goes further: **none
of the entire NC/build-heavy watchlist appears anywhere in the codebase**, in any
module, eager or lazy. There is no `__init__.py`-based package structure at all
(flat two-file model) — no hidden eager-import taint vector like TRELLIS.2's
`o_voxel/__init__.py` (doc 12 §2).

**The only non-pure-Python op**: `torchvision.ops.deform_conv2d`, used once
(`model.py:591`) inside `_DeformableConv2d`, part of a BiRefNet-style
ASPP-Deformable background-removal backbone (`birefnet.safetensors`). This is a
**standard torchvision C++/CUDA op compiled into every official torchvision wheel**
for the matching torch+CUDA build — no separate extension, no vcvars build, no
`ninja`/`setup.py` invocation required. Confirms README's "near-zero dependencies"
and "no transformers, no diffusers, no version-conflict hell" claim — verified
true at the import-graph level, not just marketing copy.

## 3. Box A compatibility assessment (R-T9)

Declared requirements (README): `numpy`, `safetensors`, `pillow`, `tqdm`, plus
"install torch and torchvision according to your environment" — **no pinned
versions, no pinned CUDA/python**. This is a strength for Box A: there is no
stale `torch==2.5+cu124` constraint to fight, unlike the flash-attn/spconv
ecosystem flagged in doc 13's R-T9.

| Dependency | Box A (py3.12, torch 2.11+cu128, gsplat 1.5.3, RTX 4090) | Risk |
|---|---|---|
| `torch`, `torchvision` | Already installed for gsplat. `deform_conv2d` is part of `torchvision.ops` since torchvision 0.3 — present in all torch 2.11-compatible torchvision builds. | None |
| `numpy`, `safetensors`, `pillow`, `tqdm` | Pure-Python/wheel, py3.12-compatible, trivial `pip install`. | None |
| `gradio` (optional, demo only) | py3.12-compatible. Not needed for pipeline integration — only for the bundled Gradio demo. | None |
| CUDA toolchain / vcvars launcher | **Not needed.** No CUDA source files, no `torch.utils.cpp_extension`, no `load_inline`, no `ninja` in the codebase. `run-python.cmd` is unnecessary for TripoSplat itself (still needed for gsplat, which TripoSplat doesn't import). | None |
| `xformers` fallback question | **Moot** — TripoSplat doesn't use `xformers`/`flash_attn` at all; its attention (inside `LatentSeqMMFlowModel`) is presumably plain `torch.nn.functional` SDPA (would need to confirm in `model.py` internals during the install spike, but no import exists either way). | None identified |

**This is the lightest-dependency candidate audited to date** — strictly easier
than TRELLIS v1's gaussian-head path (doc 12), which is also clean but pulls in
`rembg`, `easydict`, and the lazy-import package machinery.

**VRAM estimate**: weights total 3.78 GB on disk (fp16 safetensors: BiRefNet
background-remover, DINOv3-ViT-H vision encoder, Flux.2 VAE encoder, TripoSplat
diffusion flow model, octree-gaussian VAE decoder). At fp16, runtime VRAM for all
five models loaded simultaneously plus activations for a single-image,
20-step diffusion run should comfortably fit well under the 24 GB ceiling —
likely in the **8–14 GB range** (DINOv3-ViT-H and the Flux.2 VAE are the largest
components; diffusion runs on compact latent tokens, not pixel space). This is an
estimate pending the actual install spike; doc 13's framing of TripoSplat as "far
lighter" than TRELLIS.2-4B (which needs the full 24 GB) is directionally correct
and likely conservative.

## 4. Architecture facts

- **Input modality**: single 2D image (`run_example.py` loads one image; no
  multi-view or text conditioning in this repo — text-to-3D would need a separate
  text-to-image front end, consistent with the Astel text pipeline design in
  CLAUDE.md §4).
- **Output representation**: **native 3D Gaussians**, confirmed — `decode_latent`
  returns a gaussian splat tensor via `OctreeGaussianDecoder.decode()`, exported
  to `.ply` / `.splat` (README). This is exactly Astel's product representation —
  no mesh intermediate.
- **Gaussian count**: configurable, `262144` max (`_NUM_GAUSSIANS_MAX = 262144`),
  must be a multiple of `gaussians_per_point` (decoder rounds and warns); README
  recommends 32,768–262,144 depending on hero-asset vs. background-prop use.
  "Learned density control" (the paper's title) = the octree decoder allocates
  more gaussians to detailed regions ("DeG adaptive detail" per the Comfy/Threads
  announcements found in search).
- **Pipeline stages** (from `triposplat.py`): BiRefNet background removal →
  DINOv3-ViT-H vision encoding → Flux.2 VAE encode → `LatentSeqMMFlowModel`
  (rectified-flow diffusion, default 20 steps, CFG scale 3.0, shift 3.0) →
  `OctreeGaussianDecoder.decode(latent, num_gaussians)` → gaussian tensor.
  Single `TripoSplatPipeline.run(image, seed, steps, guidance_scale, shift,
  num_gaussians, erode_radius, ...)` call, ~20 inference steps by default.
- **Approx LOC**: 2,523 total (matches README's "~2,000 LOC" claim closely enough —
  README likely excludes `run_gradio.py`/`run_example.py`).
- **Preprocessing**: integrated background removal (BiRefNet/ASPP-Deformable),
  with an `erode_radius` parameter for mask post-processing — no external
  preprocessing script needed.

## 5. Go/no-go recommendation

**GO — proceed to M3 step 2 (Windows install spike).** TripoSplat clears every
gate doc 13 set for this triage:

- ✅ Repo live, actively maintained (commit 12 days old), default branch `main`.
- ✅ Code AND weights both MIT, confirmed in-repo (not just HF tag inference).
- ✅ Weights at `VAST-AI/TripoSplat` on HF, 3.78 GB, matches expected checkpoint set.
- ✅ Nothing materially changed; no newer splat-native competitor surfaced.
- ✅ Import graph has **zero** NC or build-heavy dependencies — strictly cleaner
  than TRELLIS v1's already-clean gaussian-head path (doc 12).
- ✅ Box A compat: no CUDA extension build, no vcvars launcher needed, no
  flash-attn/xformers/spconv version-matching problem (R-T9 concern doesn't apply
  to this candidate at all). Only `torch`/`torchvision` (already present for gsplat)
  plus four trivial pure-Python packages.
- ✅ Native gaussian output, confirmed, with learned adaptive density — directly
  usable as the Astel L2 candidate per CLAUDE.md §3/§4.

### Ranked blocking risks (none severe; ordered by likelihood of surfacing in the install spike)

1. **(Low) Attention-kernel performance on Windows/sm_89** — `LatentSeqMMFlowModel`
   internals weren't traced for attention implementation (file too large for this
   no-GPU pass); if it hardcodes `torch.backends.cuda.sdp_kernel` flags or assumes
   a Linux-only fast path, may need a one-line fallback to `torch.nn.functional
   .scaled_dot_product_attention` default backend. Not a license/dependency issue
   — just a perf check, deferred to the install spike where it can be measured
   directly.
2. **(Low) Checkpoint format/dtype mismatches** — all weights are `*_fp16.safetensors`;
   confirm `safetensors.torch.load_file` + the `_place(m, device, dtype)` helper
   correctly handles fp16→bf16/fp32 casting if Box A's torch 2.11 default dtype
   policy differs. Trivial to fix if it arises.
3. **(Low) VRAM estimate unverified** — §3's 8–14 GB estimate is architecture-based,
   not measured. Even a 2× miss stays under 24 GB given TripoSplat's lightweight
   profile, but confirm in the install spike before assuming headroom for a
   co-resident L3 refinement stage on the same GPU.
4. **(Informational) Org-namespace split** — code under `VAST-AI-Research`, weights
   under `VAST-AI` (HF) / `VAST-AI-Research` (ModelScope mirror). Not a risk, just
   note for anyone re-deriving URLs later.

**No license, dependency, or architecture finding in this audit blocks proceeding.**
Recommend M3 step 2 target TripoSplat first (over TRELLIS v1 gaussian-head) given
its strictly cleaner dependency surface and lighter weight — if the install spike
succeeds, it becomes the lead candidate for the step-3 bake-off.

## 6. Cleanup

Clone (`%TEMP%\astel-audit\triposplat`) deleted at end of session. No weights
downloaded. No git commands run inside the Astel repo.
