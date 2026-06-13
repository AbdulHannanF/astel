# RA5 — Formats, Compression, Standards, Engines, Viewers

*Verified 2026-06-12. Targets M1 (viewer), M5 (plugins/exports), spec §1.5 open-standards mandate.*

## 1. Standards (ride them, don't invent — Meshy lesson #12)

- **KHR_gaussian_splatting (glTF)**: Khronos **release candidate announced Feb 3 2026**;
  formal ratification targeted **Q2 2026 — i.e. imminent** ([Khronos press](https://www.khronos.org/news/press/gltf-gaussian-splatting-press-release),
  [radiancefields](https://radiancefields.com/the-khronos-group-introduces-a-gltf-baseline-for-gaussian-splatting)).
  Defines splat attributes (position, orientation, scale, SH color, opacity) inside glTF 2.0
  mesh primitives. Contributors: Autodesk, Bentley, Huawei, Niantic Spatial, NVIDIA.
  A **second extension for SPZ-compressed streaming** is planned. Action: track ratification;
  implement exporter in M5 against the RC now, validate against the ratified spec.
- **SPZ** (Niantic): ~8–12× compression, on the standards track via the planned glTF
  extension; reference implementation open source (MIT — verify file in session 2).
- **SOG** (PlayCanvas): self-organizing-gaussian-derived compressed format, well supported by
  PlayCanvas/SuperSplat tooling and Spark.
- **.ply** stays the archival/interchange baseline; KSPLAT/SPLAT legacy-supported via Spark.

AURIGA exports (spec §1.5): .ply (archival) · .spz + .sog (delivery) · glTF+KHR_gaussian_splatting
(engines) · USD/USDZ splat payload (VFX — investigate current USD splat schema practice in
session 2) · `.auriga` manifest (zip: standard splat file + JSON manifest + binary layer buffers).

## 2. Web viewer (M1 decision)

| Candidate | License | Notes |
|---|---|---|
| **[Spark](https://github.com/sparkjsdev/spark)** (World Labs) | **MIT** | **PRIMARY.** Three.js; loads PLY/compressed-PLY/SPZ/SPLAT/KSPLAT/SOG; fast on mobile (WebGL2 98%+ targets); real-time per-splat editing, color/displacement, skeletal animation (→ L7 playback!); v2.1.0 May 18 2026, 520 commits, very active. Irony noted: built by a competitor (World Labs) — MIT is irrevocable; fork-resilience acceptable. |
| [GaussianSplats3D](https://github.com/mkkellogg/GaussianSplats3D) | MIT | Mature, WASM sorting; less feature-rich than Spark now. Fallback. |
| [PlayCanvas engine](https://github.com/playcanvas/engine) + [SuperSplat](https://www.thefuture3d.com/software/supersplat/) | MIT | Engine-native splats + the best open splat **editor** (incl. SplatTransform CLI — useful for our export pipeline tests). Runner-up if we outgrow Three.js. |

Layer Inspector implications: Spark's per-splat transform/edit API supports confidence-channel
heatmaps (Truth Meter) and layer scrubbing without forking the renderer. Server-side rendering
for heavy scenes: gsplat's new HiGS inference path (RA1) as a later streamed-pixels tier.

## 3. Engine plugins (M5)

- **Unity**: [aras-p/UnityGaussianSplatting](https://github.com/aras-p/UnityGaussianSplatting)
  (MIT, the reference); [gsplat-unity](https://github.com/wuyize25/gsplat-unity) (newer,
  render-queue-integrated like transparent meshes). Our plugin: import .spz/glTF + AURIGA
  manifest → auto-configure colliders (L5 convex set), mass (L6), materials (L4 bake).
- **UE5**: [NanoGS](https://www.cgchannel.com/2026/03/free-plugin-nanogs-puts-nanite-style-gaussian-splatting-in-unreal-engine/)
  (Mar 2026, "Nanite-style" screen-space-error LOD, GPU radix sort, UE5.6/5.7 — license verify);
  [XScene-UEPlugin](https://radiancefields.com/3d-gaussian-splatting-engine-support) (Niagara-based);
  [MLSLabs renderer](https://github.com/mlslabs/MLSLabsGaussianSplattingRenderer-UE) (4DGS volumetric video!).
  Strategy: build AURIGA importer/physics-setup **on top of** the best existing renderer
  rather than writing a UE splat renderer from scratch — re-evaluate at M5.
- **Blender**: ecosystem exists ([overview](https://radiancefields.com/3d-gaussian-splatting-engine-support));
  evaluate at M5.

## 4. Coordinate conventions (the documented-rotations mandate, spec §2)

The pitfall matrix to be fully specified (with exact quaternion/SH-rotation handling) in M5
docs; the headline conversions:

| System | Handedness | Up | Notes |
|---|---|---|---|
| COLMAP/OpenCV (training data) | RH | −Y (Y down, Z forward) | gsplat native |
| Three.js / glTF / OpenGL | RH | +Y | Z toward viewer |
| Unity | **LH** | +Y | Z forward; mirror + SH coefficient resign |
| Unreal | **LH** | +Z | X forward, **centimeters** |
| USD | RH | +Y or +Z (stage metadata) | meters default, per-stage `metersPerUnit` |

Rotating SH coefficients (band ≥1) under basis change is the classic silent-corruption bug —
unit-test with golden files loaded back in headless engines (spec §10.5).

## 5. Open questions → session 2

1. SPZ + SOG license files; KHR RC spec text (read in full before M1 schema design).
2. USD gaussian-splat payload: current community schema (NVIDIA Omniverse practice?).
3. NanoGS license + source availability; XScene license.
4. Spark SH-band support and max splat counts vs our 5M cinematic tier (may need LOD/streaming
   layer of our own — [SplatBus](https://arxiv.org/html/2601.15431v1) GPU-IPC viewer framework
   as architectural reference).

## Sources

- https://www.khronos.org/news/press/gltf-gaussian-splatting-press-release · https://radiancefields.com/the-khronos-group-introduces-a-gltf-baseline-for-gaussian-splatting · https://www.thefuture3d.com/blog-0/2026/4/4/state-of-gaussian-splatting-2026
- https://github.com/sparkjsdev/spark (v2.1.0) · https://news.ycombinator.com/item?id=44249565 · https://github.com/mkkellogg/GaussianSplats3D
- https://github.com/aras-p/UnityGaussianSplatting · https://github.com/wuyize25/gsplat-unity
- https://www.cgchannel.com/2026/03/free-plugin-nanogs-puts-nanite-style-gaussian-splatting-in-unreal-engine/ · https://github.com/mlslabs/MLSLabsGaussianSplattingRenderer-UE · https://radiancefields.com/3d-gaussian-splatting-engine-support
- https://www.thefuture3d.com/software/supersplat/ · https://arxiv.org/html/2601.15431v1
