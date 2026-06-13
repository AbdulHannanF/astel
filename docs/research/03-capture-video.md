# RA3 — Capture: Photos / Video → Geometry (L0/L1, L7)

*Verified 2026-06-12. Targets M2 (reality-first capture path) and M6 (dynamics).*

## 1. Pose estimation / reconstruction front-end

### The license minefield, resolved

| Model | License | Verdict |
|---|---|---|
| [DUSt3R](https://github.com/naver/dust3r/blob/main/LICENSE) (Naver) | **CC BY-NC-SA 4.0** | Disqualified (NC) |
| [MASt3R](https://github.com/naver/mast3r) (Naver) | **CC BY-NC-SA 4.0** | Disqualified (NC) |
| [VGGT](https://github.com/facebookresearch/vggt) (Meta, CVPR'25 Best Paper) | Code relicensed **commercial-OK July 29 2025** (military excluded); [HF checkpoint](https://huggingface.co/facebook/VGGT-1B) historically CC-BY-NC — **verify current checkpoint license session 2** | Conditional |
| **[MapAnything](https://github.com/facebookresearch/map-anything)** (Meta) | Code **Apache-2.0**; weights: `facebook/map-anything` (CC-BY-NC) **and `facebook/map-anything-apache` (Apache-2.0, built for commercial use)** | **PRIMARY** |
| [COLMAP](https://colmap.github.io/) / GLOMAP | BSD | Classical fallback + refinement |

### Chosen front-end (draft)

**MapAnything (`map-anything-apache`)** is the keystone: universal feed-forward **metric**
3D reconstruction; flexible inputs (images alone, or + calibration, + poses, + depth — it
consumes whatever exists); 12+ tasks incl. SfM, MVS, monocular depth, registration, depth
completion; actively maintained (v1.1.2, May 30 2026); integrates external estimators
(MoGe, VGGT, …). One model covers: pose-free phone video, multi-photo, and single-photo
depth priors — and it outputs *metric* scale natively, which is AURIGA's L1 grounding
requirement.

**Refinement chain:** MapAnything poses/points → **GLOMAP/COLMAP bundle adjustment** when
view count and overlap permit (classical BA still wins on final pose accuracy for dense
orbits) → L1 dense cloud. COLMAP/GLOMAP are BSD, CPU-heavy → run on the 3080 box's
Threadripper-class CPUs, not GPU nodes (spec §6).

## 2. Metric scale (L1's "metrically-scaled" promise)

- **[MoGe](https://github.com/microsoft/MoGe)-2** (Microsoft, CVPR'25 Oral): **MIT** — metric
  monocular geometry; clean choice for single-image scale priors and for cross-checking
  MapAnything's metric output.
- **[Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3)**: DA3METRIC-LARGE
  is **Apache-2.0** (small/base Apache; large nested variants CC-BY-NC — [license discussion](https://huggingface.co/depth-anything/DA3-LARGE/discussions/2)). Use the metric-large head.
- Disqualified (NC): UniDepthV2 (CC-BY-NC), [Metric3D](https://github.com/YvanYin/Metric3D/blob/main/LICENSE) (BSD-2 *non-commercial* variant).
- **Scale consensus design**: video/photos → MapAnything metric + MoGe-2 per-frame + (if EXIF
  focal available) SfM scale → weighted consensus with a **reported confidence interval**;
  text/image → VLM size estimator (LLM layer) with explicit CI the user can override (spec L1).
  Disagreement above threshold = surfaced in the Truth Meter, never hidden.

## 3. Video specifics

- **Frame selection/deblur**: sharpness scoring (variance of Laplacian) + optical-flow-based
  redundancy pruning — classical, no license risk; learned deblur deferred (nice-to-have).
- **Static path**: selected frames → §1 chain (video is just "more photos").
- **Dynamic path (L7, M6)**: no turnkey permissive 4DGS exists today:
  [4DGS original](https://github.com/hustvl/4DGaussians) and [4D-Scaffold-GS](https://github.com/raikuma/4D-Scaffold-GS)
  (AAAI'26) inherit the Inria 3DGS NC license; [Faster-GS](https://github.com/nerficg-project/faster-gaussian-splatting)
  (CVPR'26) is **Apache-2.0** and research-friendly — candidate base. Plan: deformation-field
  4DGS (HexPlane/MLP deformation on top of gsplat) is mostly *our own model code* over the
  Apache rasterizer — moderate lift, scheduled M6, no blocker now.

## 4. Generative completion of unseen regions (the sacred confidence channel)

Capture pipelines produce partial coverage. Spec §4: generative completion ONLY for unseen
regions, flagged in the confidence channel. Design note: maintain a per-gaussian provenance
scalar (measured ↔ generated, continuous) from L0 onward; completion uses the RA2 generative
stack conditioned on observed views; the Truth Meter's hallucination heatmap renders this
channel directly. This is a data-model requirement for M1's asset schema — record in
DECISIONS.md so the manifest format reserves it from day one.

## 5. Open questions → session 2

1. VGGT-1B checkpoint license today; whether MapAnything-apache matches NC-checkpoint quality
   (Meta says identical functionality, different training data — benchmark on our corpus).
2. MapAnything object-centric performance (it's scene-oriented; orbit-around-object is our
   M2 case) — smoke test session 3 on the 4090 box.
3. GLOMAP vs COLMAP speed/robustness on 200–500 frame orbits.
4. Faster-GS as base framework vs plain gsplat (it claims optimization improvements, CVPR'26).

## Sources

- https://github.com/facebookresearch/map-anything · https://github.com/facebookresearch/vggt · https://huggingface.co/facebook/VGGT-1B
- https://github.com/naver/dust3r/blob/main/LICENSE · https://github.com/naver/mast3r
- https://github.com/microsoft/MoGe · https://github.com/ByteDance-Seed/Depth-Anything-3 · https://huggingface.co/depth-anything/DA3-LARGE/discussions/2
- https://github.com/lpiccinelli-eth/unidepth · https://github.com/YvanYin/Metric3D/blob/main/LICENSE
- https://github.com/hustvl/4DGaussians (via https://guanjunwu.github.io/4dgs/) · https://github.com/raikuma/4D-Scaffold-GS · https://github.com/nerficg-project/faster-gaussian-splatting
- https://arxiv.org/pdf/2507.14798 (DUSt3R/MASt3R/VGGT evaluation) · https://arxiv.org/pdf/2507.14501 (feed-forward survey)
