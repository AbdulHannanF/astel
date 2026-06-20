# Layered asset model

Every Astel asset is a **Layer Stack** persisted as one `.astel` package (a zip of
standard files plus a JSON manifest). The layers are *bound to one another*, never
collapsed into a single blob — inspectability is the product's flagship
differentiator. Layers, in pipeline order:

| Layer | Name | What it carries |
|---|---|---|
| **L0** | Seed / sparse point cloud | SfM points (capture) or generative latent samples (text/image), with per-point confidence. The cheap preview tier. |
| **L1** | Dense cloud | Densified, metrically-scaled point cloud with normals and per-point semantic logits. Scale is grounded from SfM or a VLM size estimate with an explicit confidence interval. |
| **L2** | Coarse gaussians | Fast feed-forward gaussians (TripoSplat-class) initialised from L1 — good enough to judge shape/identity. Cheap preview. |
| **L3** | Refined surface gaussians | The **hero layer**: 2DGS surface-aligned refinement with anti-aliasing, per-splat normals, configurable splat budget. Geometric error vs. L1 is reported in the quality report. |
| **L4** | Appearance / lighting | Per-gaussian decomposition into albedo + estimated environment illumination so the asset relights instead of shipping lighting baked into colour. |
| **L5** | Collision & solidity | Derived from L3: a sparse-voxel SDF → convex-decomposition collision proxy, a watertight isosurface used **only** for the print path / physics volume, plus centre of mass, inertia tensor, and volume. Data bound to the splat asset — never a visible mesh deliverable. |
| **L6** | Physics-material & semantic | Per-region material classification (rigid/soft/cloth, density, friction/restitution) plus articulation hints, produced by an LLM/VLM reasoning pass. Enables correct mass in engines and a physics preview. |
| **L7** | Dynamics *(optional)* | For dynamic captures: a deformation field / 4DGS keyframes, exportable as animated splats or baked motion. |

> **Splats are the only product.** The internal scaffolding (point clouds, SDFs,
> convex hulls, watertight surfaces) is required and stored, but it lives as layers
> bound to the splat asset. No mesh is ever offered as the deliverable; the print
> path *derives* a surface internally because physical printing requires one.

## Honesty contract

Each layer records provenance and measured error rather than a plausible-looking
number. Unmeasured fields are explicit `null` with a reason (see the
[Truth Meter](truth-meter.md)); the confidence channel separates *measured* regions
from *generated* ones, and generated geometry is never silently passed off as
captured reality.

## Inspecting layers

The web app's **Layer Inspector** lets you scrub the stack — point cloud → dense →
coarse → refined → relit → collision ghost → physics preview — and export any layer
individually.
