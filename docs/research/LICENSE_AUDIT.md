# LICENSE_AUDIT.md — v1 (2026-06-12)

*Each row verified against the repo/model-card on the date above. ✅ ship-safe · ❌ blocked for
commercial · 🔍 boundary-check needed at clone time. This file becomes a CI gate in M1.*

## Verified ship-safe

| Dependency | License | Verified at |
|---|---|---|
| gsplat | Apache-2.0 | github.com/nerfstudio-project/gsplat |
| 3dgrut (3DGRT/3DGUT) | Apache-2.0 | github.com/nv-tlabs/3dgrut |
| TRELLIS v1 models + core code | MIT | github.com/microsoft/TRELLIS |
| TRELLIS.2 code + 4B weights *(but see ❌ deps)* | MIT | github.com/microsoft/TRELLIS.2 |
| MapAnything code + `map-anything-apache` ckpt | Apache-2.0 | github.com/facebookresearch/map-anything |
| MoGe / MoGe-2 | MIT | github.com/microsoft/MoGe |
| Depth Anything 3 (DA3METRIC-LARGE, small/base) | Apache-2.0 | HF depth-anything |
| FLUX.1-schnell weights | Apache-2.0 ("personal, scientific, and commercial") | HF black-forest-labs/FLUX.1-schnell |
| MV-Adapter | Apache-2.0 | github.com/huanngzh/MV-Adapter |
| SAM 2 (code + checkpoints + training code) | Apache-2.0 | github.com/facebookresearch/sam2 |
| NVIDIA Warp | Apache-2.0 | github.com/NVIDIA/warp |
| SPZ reference impl | MIT | github.com/nianticlabs/spz |
| Spark viewer | MIT | github.com/sparkjsdev/spark |
| COLMAP / GLOMAP | BSD | colmap.github.io |
| Open3D, trimesh, CoACD, V-HACD | MIT/MIT/MIT/BSD-3 | (stable, re-pin versions in M1) |

## Blocked for commercial use

| Dependency | License | Where it bites | Mitigation |
|---|---|---|---|
| **nvdiffrast** | NVIDIA Source Code License (research/eval; commercial via NVIDIA Research Licensing form) | **TRELLIS.2 dependency** (mesh rasterization in its texture stage) | See decision below |
| **nvdiffrec** | NVIDIA Source Code License (same family) | TRELLIS.2 dependency | Same |
| Inria 3DGS lineage, DUSt3R/MASt3R, UniDepth, Metric3D, Hunyuan3D | NC variants (RA1/RA3) | Already routed around | — (design references only) |

## Decision: TRELLIS.2 usage boundary (updates DECISIONS.md D#2)

Running TRELLIS.2 server-side in a paid/commercial product = commercial use of its
dependencies. Therefore the prior-distillation recipe is amended:

1. ✅ **Geometry prior stays TRELLIS.2.** Import-graph audit (2026-06-13, CPU-only,
   [12-trellis-import-audit.md](12-trellis-import-audit.md)): O-Voxel geometry decode
   (`FlexiDualGridVaeDecoder` / `flexible_dual_grid_to_mesh`) is tainted by nvdiffrast ONLY
   via one eager `from . import postprocess` line in `o_voxel/__init__.py` (texture/GLB-baking
   stage, as predicted) — the decode math itself has zero NC deps. Fix: fork `o_voxel` (MIT),
   make that one import lazy (matches the pattern already used everywhere else in this
   codebase). After the patch, geometry decode is clean.
2. **Appearance guidance moves to MV-Adapter (Apache)** + FLUX.1-schnell renders; we do NOT
   need TRELLIS.2's texture stage — L4 is our own decomposition anyway, and prior-view
   rendering for distillation uses our own renderer (pyrender/moderngl — inference-only,
   no differentiability needed) instead of nvdiffrast.
3. ✅ **Fallback confirmed viable too:** TRELLIS v1 end-to-end (MIT gaussian head) —
   `TrellisImageTo3DPipeline.run()` → `outputs['gaussian']` has zero nvdiffrast/diffoctreerast
   in its import chain, *as long as* we avoid `trellis.utils.render_utils` /
   `postprocessing_utils` (demo helpers we don't need; write our own gsplat preview renderer).
   FlexiCubes (vendored submodule) + Kaolin, which `import trellis` does pull in via
   `representations.mesh`, are both Apache-2.0 — not a blocker.
4. **Parallel track (USER, optional, $0 to ask):** submit NVIDIA Research Licensing inquiry
   for nvdiffrast commercial terms — removes the constraint entirely if granted on
   acceptable terms. (No longer blocking — kept open as a nice-to-have.)

## Closed this session (2026-06-13, CPU-only import audit)

- ✅ **TRELLIS v1 gaussian-path import graph** — clean as-is (see
  [12-trellis-import-audit.md](12-trellis-import-audit.md) §1).
- ✅ **TRELLIS.2 geometry-decode import graph** — clean after 1-line MIT→MIT fork patch to
  `o_voxel/__init__.py` (§2 of same doc).
- ✅ **SOG reference implementation license** — `playcanvas/splat-transform`, MIT.
- ✅ **Spark SH-band limits** — supports SH0–SH3 (up to 56 bytes/splat), no documented 1M+
  splat-count ceiling; sufficient for our L3/L4 output.
- ✅ **VGGT-1B checkpoint license** — base `VGGT-1B` remains NC; new
  `facebook/VGGT-1B-Commercial` (vggt-aup-license, since ~Jul 2025) is commercial-use-ok for
  our use case. Doesn't change our chosen MapAnything-apache capture front-end; unblocks VGGT
  as a future option if revisited.

🔍 **Remaining (not yet closed):** gsplat/3dgrut version pin matrix; NanoGS;
Taichi/Genesis (moot unless Warp disappoints).
