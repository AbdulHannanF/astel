# RA1 — Core 3DGS Rasterization & Surface-Accurate Variants

*Verified 2026-06-12 via web. Targets layers L2/L3 (coarse → refined surface gaussians).*

## 1. The license fault line (most important finding)

The original Inria reference implementation of 3DGS and its `diff-gaussian-rasterization`
CUDA engine are **non-commercial** — explicit Inria consent required for any commercial
exploitation ([LICENSE](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/LICENSE.md),
[rasterizer](https://github.com/graphdeco-inria/diff-gaussian-rasterization)). Nearly the entire
first wave of surface-accurate research code is a fork of that engine and inherits the
restriction: **SuGaR, Mip-Splatting, GOF, RaDe-GS, PGSR, 2DGS (original repo)** — all
presumed NC until individually proven otherwise (verify each LICENSE file in session 2;
treat as NC for planning).

**Consequence (binding for AURIGA):** we never ship Inria-derived code. The *algorithms*
(published math: regularization losses, depth/normal rendering formulas) are reimplemented
on a permissive backbone. Two permissive backbones exist and we use both:

| Backbone | License | What it gives us |
|---|---|---|
| [gsplat](https://github.com/nerfstudio-project/gsplat) (nerfstudio) | **Apache-2.0** ([docs](https://docs.gsplat.studio/)) | Differentiable rasterization, 3DGS + **2DGS mode built in**, anti-aliased mode (Mip-Splatting equivalent), absgrad densification (AbsGS), **MCMC densification**, bilateral-grid + PPISP appearance compensation, depth/normal rendering, **3DGUT integration** |
| [3dgrut](https://github.com/nv-tlabs/3dgrut) (NVIDIA) | **Apache-2.0** | Ray tracing of gaussian particles (3DGRT), unscented-transform rasterization (3DGUT), hybrid raster+RT, distorted/rolling-shutter cameras, secondary rays (reflections/shadows) — the relighting-grade render path for L4 |

## 2. gsplat status (verified June 2026)

Active and accelerating: v1.5.3+ line; May 2026 added an experimental **HiGS inference-only
path** (macro-tile fused rasterization, fp16 packing — relevant to our viewer/LOD streaming
later) and native CUDA MCMC noise injection; Jan 2026 added PPISP appearance compensation
([releases](https://github.com/nerfstudio-project/gsplat/releases),
[radiancefields.com](https://radiancefields.com/gsplat-1-5-3-released-by-nerfstudio)).
2DGS is integrated with ~4 GB VRAM overhead and improving viewer compatibility. NVIDIA itself
upstreamed 3DGUT into gsplat ([NVIDIA blog](https://developer.nvidia.com/blog/revolutionizing-neural-reconstruction-and-rendering-in-gsplat-with-3dgut/)).

**Decision-grade:** gsplat is the rasterization backbone. It is Apache, maintained by the
nerfstudio org, has the largest community of any splat library, and already contains
permissive reimplementations of four papers we'd otherwise have to reimplement ourselves
(2DGS, Mip-Splatting AA, AbsGS, MCMC).

## 3. Surface-accuracy technique choice for L3

Literature consensus (2024–2026 comparisons, e.g. [GS-SR survey repo](https://github.com/yanxian-ll/GS-SR),
[PGSR paper](https://arxiv.org/html/2406.06521v2)):

- **2DGS** (surfel disks, true normals): fastest, view-consistent geometry, but over-smooths
  volumetric/fuzzy regions ([2DGS-derived analyses](https://arxiv.org/html/2605.00569v1)).
- **PGSR** (planar gaussians + single-view & multi-view photometric-geometric constraints):
  **best surface accuracy** in published comparisons; slower; original code NC.
- **RaDe-GS / GOF**: rasterized depth/normal and opacity-field isosurfacing — good quality,
  NC code lineage.
- **SuGaR**: surface-alignment regularizer + Poisson extraction — historically important,
  superseded in accuracy by PGSR-class; NC.
- **DN-Splatter** ([repo](https://github.com/maturk/dn-splatter), WACV 2025): depth+normal
  *prior supervision* on top of gsplat — directly compatible with our backbone; license to
  confirm in session 2, but its losses are trivially reimplementable. Pairs naturally with
  monocular normal/depth predictors for the capture path.

**Chosen recipe for L3 (draft, to finalize in DECISIONS.md):**
gsplat **2DGS mode** as the representation (real per-splat normals — needed by L4 BRDF and
L5 SDF) + **MCMC or absgrad densification** + **anti-aliased mode** + reimplemented
**PGSR-style multi-view geometric consistency losses** + DN-Splatter-style monocular
depth/normal priors when capture data is available. Runner-up representation: plain 3DGS +
GOF-style opacity-field extraction (kept as an A/B during M2 since 2DGS's over-smoothing on
fuzzy matter conflicts with splats' core strength — evaluate on the fixed asset corpus).

For relighting-grade rendering and any secondary-ray need (L4 validation, Relight Studio
ground truth): **3DGRT/3DGUT** via 3dgrut or gsplat's 3DGUT mode.

## 4. VRAM/runtime notes (for our hardware)

- gsplat trains 1M-splat scenes comfortably within 24 GB (4090); 2DGS mode adds ~4 GB.
  Fits spec §6 "patient mode" on the user's 2×4090 box; 3080s (10–12 GB) are preview-tier
  (L0–L2, ≤300k splats) only.
- MCMC densification gives *fixed splat-count budgets* — directly implements the spec's
  "configurable splat budget" knob (100k/1M/5M tiers) more controllably than classic
  clone-and-split. This is a product feature falling out of a research choice.

## 5. Open questions → session 2

1. Confirm individual LICENSE files: DN-Splatter, RaDe-GS, GOF, PGSR, original 2DGS repo.
2. Quantify 2DGS-vs-3DGS+GOF quality gap on fuzzy/volumetric content (hair, foliage).
3. gsplat HiGS inference path maturity — candidate for the viewer's server-side LOD renderer?
4. How much of PGSR's multi-view constraint set is needed once monocular priors are present
   (ablation plan for M2).

## Sources

- https://github.com/nerfstudio-project/gsplat · https://docs.gsplat.studio/ · https://github.com/nerfstudio-project/gsplat/releases
- https://radiancefields.com/gsplat-1-5-3-released-by-nerfstudio
- https://github.com/graphdeco-inria/gaussian-splatting/blob/main/LICENSE.md · https://github.com/graphdeco-inria/diff-gaussian-rasterization
- https://github.com/nv-tlabs/3dgrut · https://research.nvidia.com/labs/toronto-ai/3DGUT/ · https://developer.nvidia.com/blog/revolutionizing-neural-reconstruction-and-rendering-in-gsplat-with-3dgut/
- https://arxiv.org/html/2406.06521v2 (PGSR) · https://github.com/yanxian-ll/GS-SR · https://github.com/maturk/dn-splatter · https://arxiv.org/abs/2403.17822
- https://arxiv.org/html/2605.00569v1 (2D-SuGaR, 2026)
