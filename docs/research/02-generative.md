# RA2 — Generative Splats: Text / Image → 3D

*Verified 2026-06-12. Targets the M3 generative path: Generation Spec → conditioning images → L2 coarse gaussians → L3 refinement.*

## 1. Foundation model choice: the TRELLIS family (MIT) wins on license + quality

**[TRELLIS](https://github.com/microsoft/TRELLIS)** (Microsoft, CVPR'25 Spotlight) — structured
latents (SLAT) + rectified flow; decodes to **3D gaussians, radiance fields, AND meshes**.
- Models: TRELLIS-image-large (1.2B) and TRELLIS-text-base/large/xlarge (342M/1.1B/2.0B).
- **License: MIT for models and most code** — flag: submodules `diffoctreerast` (radiance-field
  renderer) and modified FlexiCubes carry separate licenses → verify both in session 2; we
  primarily need the **gaussian head**, so the RF renderer may be avoidable entirely.
- VRAM ≥16 GB → runs on the 4090 box; not on the 3080s.
- Last major update March 2025 (training code + text models) — stable, not abandoned.

**[TRELLIS.2](https://github.com/microsoft/TRELLIS.2)** (May 2026, 4B flow-matching transformer,
**MIT code + weights**, ≥24 GB VRAM) — the new SOTA open image-to-3D
([model card](https://huggingface.co/microsoft/TRELLIS.2-4B), [project page](https://microsoft.github.io/TRELLIS.2/)).
**Critical caveat: mesh-only output** ("O-Voxel" field-free sparse voxels → GLB with full PBR
incl. transparency; no gaussian head; image-to-3D only). Dependencies nvdiffrast/nvdiffrec have
separate (NVIDIA source-available, non-commercial for nvdiffrec?) licenses — **verify session 2**.

**How both fit AURIGA without violating "splats only" (§1):** internal scaffolding is
explicitly allowed. Proposed architecture:

```
L2 (coarse, fast):   TRELLIS-image-large gaussian head  →  instant previewable splats
L3 (refined):        gsplat 2DGS optimization initialized from L2, supervised by
                     (a) multi-view renders of a TRELLIS.2 O-Voxel internal prior
                         (geometry + PBR ground truth for the generated object), and
                     (b) multi-view diffusion guidance for regions the prior lacks.
                     The TRELLIS.2 mesh is never exported — it is an internal proxy
                     bound to the asset (feeds L5 SDF too).
```

This "generate prior → distill into surface splats" recipe gives us TRELLIS.2-grade geometry
in splat form — likely *better* geometric accuracy than any native splat generator today, and
the PBR output of TRELLIS.2 directly seeds the L4 appearance decomposition.

## 2. Disqualified / runner-up generators

| Model | Verdict | Why |
|---|---|---|
| [Hunyuan3D-2/2.1/2.5](https://github.com/Tencent-Hunyuan/Hunyuan3D-2/blob/main/LICENSE) (Tencent) | **Disqualified** | Tencent Community License: territory **excludes EU, UK, South Korea** (outputs included!), >100M MAU requires separate grant ([HN discussion](https://news.ycombinator.com/item?id=43420870), [open issue](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/issues/94)). Unusable for a global SaaS. Mesh-output anyway. |
| LGM (MIT, 2024) | Runner-up for L2 | Multi-view images → feed-forward gaussians in seconds; older, lower fidelity than TRELLIS; tiny VRAM. Keep as 3080-class fallback / speed tier. |
| GRM, TriplaneGaussian | Reference only | Superseded by TRELLIS-class quality; license review not worth the time now. |
| DreamGaussian / GaussianDreamer (SDS) | Reference only | Per-asset optimization, slow, "melted" era. Their SDS-guidance idea survives inside our L3 refinement, not as a pipeline. |
| Scene-level feed-forward (AnySplat, YoNoSplat, Long-LRM++, VolSplat — [survey](https://arxiv.org/pdf/2507.14501)) | → RA3 | These are capture/pose-free reconstructors, evaluated in 03-capture-video.md. |

## 3. Conditioning-image manufacturing (the Meshy lesson)

Text quality is controlled by manufacturing ideal conditioning images, not by text-native 3D
models (meshy-analysis §7.5). TRELLIS-text-* exists but the image path is stronger; route:
**LLM Generation Spec → canonicalized prompt → open T2I model → background removal →
TRELLIS image pipeline.**

- T2I candidates (verify licenses session 2): FLUX.1-schnell (Apache-2.0), Qwen-Image
  (Apache-2.0), SD3.5 (Stability Community License — revenue cap, flag). Default plan:
  FLUX.1-schnell self-hosted on the 4090 box.
- **[MV-Adapter](https://github.com/huanngzh/MV-Adapter)** (ICCV 2025): plug-and-play adapter
  turning SDXL/T2I models into multi-view generators (768²), also does geometry-conditioned
  **texturing** — candidate for both L3 guidance views and L4 texture refinement. License not
  yet confirmed — **verify session 2**.

## 4. Open questions → session 2

1. `diffoctreerast` + FlexiCubes + nvdiffrast/nvdiffrec license terms (could constrain which
   TRELLIS heads/components we ship; gaussian-head-only path likely dodges all of them).
2. MV-Adapter license; current best open multi-view diffusion (Era3D? newer 2026 entrants?).
3. TRELLIS.2 prior-distillation feasibility test on the 4090 box (session 3 smoke check):
   image → O-Voxel mesh render set → 2DGS fit → Chamfer vs the prior. This experiment
   de-risks the whole M3 architecture.
4. Whether TRELLIS (v1) text models are good enough to skip T2I manufacturing for simple
   prompts (cheaper path) — A/B during M3.

## Sources

- https://github.com/microsoft/TRELLIS · https://trellis3d.github.io/
- https://github.com/microsoft/TRELLIS.2 · https://microsoft.github.io/TRELLIS.2/ · https://huggingface.co/microsoft/TRELLIS.2-4B
- https://wilsonwu.me/en/blog/2026/llm-microsoft-trellis-3d/ · https://www.pixelsham.com/2026/05/25/microsoft-trellis-2-open-source-high-resolution-2d-to-3d-generative-modeling/
- https://github.com/Tencent-Hunyuan/Hunyuan3D-2/blob/main/LICENSE · https://news.ycombinator.com/item?id=43420870
- https://github.com/huanngzh/MV-Adapter · https://arxiv.org/abs/2412.03632
- https://arxiv.org/pdf/2507.14501 (feed-forward survey)
