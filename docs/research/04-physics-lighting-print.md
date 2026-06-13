# RA4 — Physics (L6), Appearance/Lighting (L4), Solidity & Print (L5)

*Verified 2026-06-12. Targets M4 (world-awareness) — AURIGA's differentiation milestone.*

## 1. Physics layer (L6) + Physics Sandbox

**[PhysGaussian](https://xpandora.github.io/PhysGaussian/)** (CVPR'24) is the spiritual core:
custom MPM treating gaussian kernels directly as material points (kinematic deformation +
stress attributes on the kernels; no mesh ever). Ecosystem since:

- [gs-mpm](https://github.com/ranrandy/gs-mpm): PhysGaussian reimplemented in **Taichi** + elasticity reconstruction.
- [GASP](https://github.com/waczjoan/GASP) (2025): flat (2D) gaussians + black-box physics engines, built on **Genesis** + Taichi Elements — directly relevant since our L3 uses 2DGS surfels.
- [i-PhysGaussian](https://arxiv.org/html/2602.17117v1) (Feb 2026): implicit MPM integrator — larger stable timesteps, better for interactive sandbox.

**Implementation strategy (license-clean by construction):** MPM-on-gaussians is a thin
coupling layer over a permissive simulation engine. Candidates, all to license-verify in
session 2 but historically permissive: **NVIDIA Warp** (Apache-2.0) — first choice (CUDA-native,
differentiable, maintained by NVIDIA, MPM examples exist); **Taichi** (Apache-2.0) — the
canonical MPM tool (and the Meshy founder's own creation — the irony is strategic);
**Genesis** (Apache-2.0) — batteries-included sim platform, heavier dependency.
PhysGaussian's official repo license: verify; we likely reimplement its (published) kernel
treatment on Warp regardless. Sandbox architecture: server-side sim on GPU workers, streamed
state deltas to the browser viewer (spec §8.2).

**L6 semantic reasoning** (per-region material class, density, friction): LLM/VLM pass over
canonical renders + part segmentation — this is our own code on the Anthropic adapter; no
external model licensing beyond the API. Part segmentation candidate: SAM-class models on
renders back-projected to splats (SAM/SAM2 are Apache-2.0 — verify SAM2 still; fallback SAM1).

## 2. Appearance / lighting layer (L4)

Research landscape (all NC-or-unverified code; methods reimplementable):

- [Relightable 3D Gaussian](https://arxiv.org/abs/2311.16043)-class: per-gaussian PBR (normal,
  BRDF) + per-splat ray tracing for visibility.
- GS-IR, [GUS-IR](https://arxiv.org/html/2411.07478) (forward-vs-deferred shading analysis,
  unified solution), GaussianShader (reflective surfaces).
- [RTR-GS](https://arxiv.org/html/2507.07733v1) (ACM MM'25): current quality bar — radiance
  transfer + reflection separation, physically-based deferred rendering, handles secondary
  lighting.

**Chosen direction (draft):** per-gaussian {albedo, roughness, metallic, specular, emissive}
+ jointly-optimized environment map, via **deferred shading on gsplat** (2DGS normals from L3
make this tractable — deferred PBR needs reliable normals, which is exactly what the L3
representation choice provides). **3DGRT (Apache)** supplies ray-traced visibility/secondary
rays for the high-quality tier and Relight Studio ground-truth comparisons. Generative path:
TRELLIS.2's native PBR output (MIT) seeds L4 directly — we inherit de-lighted materials
rather than retrofitting de-lighting (Meshy lesson #7). Capture path: intrinsic decomposition
during L3→L4 optimization, env-light separated, never baked-only (spec L4).

**Export reality:** engines consuming plain colored splats get a "baked preview" SH bake
*generated from* L4 (one-way derivation is allowed; baked-only is not). The AURIGA manifest
carries the decomposed channels.

## 3. Solidity & print path (L5)

Pipeline: **L3 surfels → TSDF fusion of rendered depth (2DGS gives clean depth/normals) →
sparse voxel SDF → (a) marching-cubes isosurface → watertight repair → .3mf/.stl; (b) convex
decomposition for engine collision; (c) mass properties** (volume, center of mass, inertia
tensor from the SDF + L6 densities).

Tooling (permissive, version-verify in session 2): Open3D (MIT — TSDF fusion, marching cubes),
trimesh (MIT — watertight checks, repair, mass properties, 3MF/STL IO), **CoACD** (MIT —
SOTA approximate convex decomposition) with V-HACD (BSD-3) fallback, lib3mf (BSD-2) if trimesh
3MF support proves insufficient. Printability checks (wall thickness via SDF erosion, overhang
analysis from normals, hollowing with drain holes): our own code on the SDF — straightforward
and fully ours. GOF/RaDe-GS-style direct isosurfacing from opacity fields is the research-grade
alternative if TSDF fusion shows artifacts (reimplement on gsplat if needed).

Spec compliance note: the isosurface exists *only* inside L5/print (§1.1) — exporters refuse
to emit it except as .3mf/.stl print files.

## 4. Open questions → session 2

1. License files: PhysGaussian repo, Warp, Taichi, Genesis, SAM2, Open3D/trimesh/CoACD current.
2. Warp MPM example maturity vs Taichi Elements — pick one (criteria: CUDA graph capture for
   the sandbox's interactive loop; differentiability is a bonus, not required for L6 preview).
3. Deferred-vs-forward shading on 2DGS surfels: GUS-IR's analysis says hybrid — prototype both
   on one asset in M4 spike week.
4. Does TRELLIS.2 PBR quality survive distillation into per-gaussian materials? (Couples to
   RA2 Q3 experiment.)

## Sources

- https://xpandora.github.io/PhysGaussian/ · https://openaccess.thecvf.com/content/CVPR2024/papers/Xie_PhysGaussian_Physics-Integrated_3D_Gaussians_for_Generative_Dynamics_CVPR_2024_paper.pdf
- https://github.com/ranrandy/gs-mpm · https://github.com/waczjoan/GASP · https://arxiv.org/html/2409.05819v1 · https://arxiv.org/html/2602.17117v1
- https://arxiv.org/html/2411.07478 (GUS-IR) · https://arxiv.org/html/2507.07733v1 (RTR-GS) · https://dl.acm.org/doi/10.1145/3746027.3755197
- https://arxiv.org/pdf/2404.01223 (Feature Splatting — language-driven physics editing, related to L6)
