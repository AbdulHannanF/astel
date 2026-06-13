# 12 — TRELLIS / TRELLIS.2 Import-Graph Audit (2026-06-13, CPU-only session)

*Method: shallow-cloned `microsoft/TRELLIS` and `microsoft/TRELLIS.2` (current `main`, confirmed
live on GitHub today) into `%TEMP%\astel-audit\`, statically traced every `import`/`from` for
`nvdiffrast`, `nvdiffrec`, `kaolin`, `flexicubes`, `diso`, `diffoctreerast`, `cumesh`, `flex_gemm`
across both repos and their submodules, including transitive package-`__init__.py` execution
order (Python always runs a package's `__init__.py` before any submodule import, even for
`from pkg.sub import x`). Clones deleted after analysis — nothing persisted outside `%TEMP%`.*

## 0. Repo identity check (2026-06-13)

- `github.com/microsoft/TRELLIS` — still the canonical TRELLIS v1 repo (MIT). Live, default
  branch `main`.
- `github.com/microsoft/TRELLIS.2` — live, MIT-licensed, actively maintained (issues opened as
  recently as late May 2026). Ships `trellis2/` (model+pipeline code) and `o-voxel/` (the
  O-Voxel/Flexible-Dual-Grid voxel representation + I/O + postprocessing package, published
  separately as the `o_voxel` pip package via `o-voxel/pyproject.toml`).
- Weights: `microsoft/TRELLIS.2-4B` on Hugging Face — MIT per repo card.

## 1. TRELLIS v1 — gaussian-head (L2) import graph

**Top-level `LICENSE`**: MIT (Microsoft Corporation). ✅

`trellis/__init__.py` eagerly does:
```python
from . import models, modules, pipelines, renderers, representations, utils
```
- `renderers/__init__.py` and `representations/__init__.py` (and TRELLIS.2's `pipelines`,
  `models`, `renderers`, `representations` `__init__.py` files) all use a **lazy
  `importlib`-based `__getattr__`** pattern — submodules are only imported the moment a specific
  name is *accessed*, not at package-import time. This is the load-bearing mechanism that keeps
  `import trellis` itself clean.
- `representations/__init__.py` *does* eagerly do `from .mesh import MeshExtractResult` →
  `cube2mesh.py` → `from .flexicubes.flexicubes import FlexiCubes` (vendored git submodule,
  `MaxtirError/FlexiCubes`, a fork of `nv-tlabs/FlexiCubes`, **Apache-2.0** ✅) →
  `flexicubes.py` imports `from kaolin.utils.testing import check_tensor`. **Kaolin
  (`NVIDIAGameWorks/kaolin`) is Apache-2.0** ✅ and `kaolin.utils.testing` does not pull
  nvdiffrast as a hard import (nvdiffrast integration lives in the separate, optional
  `kaolin.render.mesh.nvdiffrast_context` submodule, not imported by `check_tensor`). →
  **`import trellis` is clean of NC code.**

### Tainted modules (import nvdiffrast/diffoctreerast at module top-level)
| Module | Taint | Trigger |
|---|---|---|
| `trellis/renderers/mesh_renderer.py` | `import nvdiffrast.torch as dr` (line 2) | accessed via `renderers.MeshRenderer` |
| `trellis/renderers/octree_renderer.py` | function-level `from diffoctreerast import ...` (lazy, inside method) + module-level use | accessed via `renderers.OctreeRenderer` |
| `trellis/utils/postprocessing_utils.py` | `import nvdiffrast.torch as dr` (line 5) | only via `from trellis.utils import postprocessing_utils` |
| `trellis/utils/render_utils.py` | not itself, but line 7 `from ..renderers import OctreeRenderer, GaussianRenderer, MeshRenderer` — this is an eager `from X import a,b,c`, which calls `__getattr__` for **all three names immediately**, so importing `render_utils` transitively imports `octree_renderer.py` and `mesh_renderer.py` → **tainted** | only via `from trellis.utils import render_utils` |

### Clean entry point for L2 (gaussian-head inference)
```python
from trellis.pipelines import TrellisImageTo3DPipeline   # clean
pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
outputs = pipeline.run(image)   # outputs['gaussian'] — no nvdiffrast/diffoctreerast in chain
```
`trellis/pipelines/trellis_image_to_3d.py` imports only `torch`, `torchvision`, `rembg`,
`.base`, `.samplers`, `..modules.sparse` — **none of these touch renderers/utils**.
`trellis/renderers/gaussian_render.py` (the actual rasterizer used by `GaussianRenderer`,
accessed lazily) imports only `torch`, `numpy`, `easydict`, `..representations.gaussian`,
`.sh_utils` — **clean of nvdiffrast**. (Note: `gaussian_render.py` is the Inria
diff-gaussian-rasterization-derived code, already correctly flagged elsewhere in
LICENSE_AUDIT.md as NC-licensed 3DGS lineage — separate issue from nvdiffrast, tracked
independently; gsplat replaces it in our stack regardless.)

**Verdict (1a): TRELLIS v1 gaussian-head inference (`TrellisImageTo3DPipeline.run()` →
`outputs['gaussian']`) runs with ZERO nvdiffrast/nvdiffrec/diffoctreerast in the import
chain**, provided we do NOT import `trellis.utils.render_utils` or
`trellis.utils.postprocessing_utils` (both are demo/export helpers for mesh-extraction paths
we don't need — write our own gsplat-based preview renderer instead, as already planned).

## 2. TRELLIS.2 — O-Voxel geometry decode (L3 supervision) import graph

**Top-level `LICENSE`**: MIT (Microsoft Corporation). ✅

`trellis2/__init__.py` eagerly imports `models, modules, pipelines, renderers, representations,
utils` — **all five of these `__init__.py` files use the same lazy importlib pattern** (verified
by reading each), so `import trellis2` alone is clean.

### The taint: `o_voxel` package's eager `__init__.py`
```python
# o-voxel/o_voxel/__init__.py
from . import (convert, io, postprocess, rasterize, serialize)
```
This is **eager**, unlike every `__init__.py` inside `trellis2/`. `postprocess.py` line 10:
`import nvdiffrast.torch as dr` (used for GLB texture-baking in
`trellis2_texturing.py`/`o_voxel.postprocess.to_glb`). Because Python always executes a
package's `__init__.py` before any of its submodules, **any `import o_voxel` or
`from o_voxel.X import Y` — including `from o_voxel.convert import flexible_dual_grid_to_mesh`
— first runs `o_voxel/__init__.py`, which imports `postprocess`, which imports nvdiffrast.**

### Path from O-Voxel geometry decode to the taint
`trellis2/models/__init__.py` (lazy) → `FlexiDualGridVaeDecoder` → loads
`trellis2/models/sc_vaes/fdg_vae.py` → line 20:
`from o_voxel.convert import flexible_dual_grid_to_mesh` → **executes `o_voxel/__init__.py`
→ `postprocess.py` → `import nvdiffrast.torch as dr`.**

**`FlexiDualGridVaeDecoder` (the O-Voxel geometry decoder — exactly the module we'd want for L3
geometry-prior distillation) is therefore tainted, but only incidentally**: the decode math in
`fdg_vae.py` calls `flexible_dual_grid_to_mesh` (in `o_voxel/convert/flexible_dual_grid.py`,
which itself imports only `torch`/`cumesh`/`flex_gemm` — no nvdiffrast), never anything in
`postprocess.py`. The taint is a single bad eager-import line in `o_voxel/__init__.py`, not an
architectural dependency.

`o_voxel`'s own `o-voxel/pyproject.toml` does **not declare `nvdiffrast` as a dependency at
all** — confirming it's an incidental/undeclared import, not a real runtime requirement of the
package as published.

### CuMesh / FlexGEMM (o_voxel's actual declared deps)
- `cumesh` (`JeffreyXiang/CuMesh`) — **MIT**. Depends on `cubvh`, `xatlas`, `pamo` — no
  nvdiffrast/kaolin references found.
- `flex_gemm` (`JeffreyXiang/FlexGEMM`) — **MIT**. Deps: PyTorch ≥2.4, Triton ≥3.2 — no
  nvdiffrast/kaolin.

### Verdict (2): TRELLIS.2 O-Voxel geometry/FlexiDualGrid decode
**Tainted as published**, but via exactly one line (`from . import postprocess` in
`o-voxel/o_voxel/__init__.py`). **Remediation (required before we vendor):**
1. Fork `o_voxel` (MIT permits this); change `o_voxel/__init__.py` to the same lazy
   `importlib.__getattr__` pattern used everywhere else in this codebase (or simply drop
   `postprocess` from the eager import list and make it lazy-accessible).
2. After that one-line patch, `FlexiDualGridVaeDecoder` / `flexible_dual_grid_to_mesh` import
   with zero nvdiffrast/nvdiffrec in the chain — confirmed clean (only `torch`, `cumesh`,
   `flex_gemm`, `numpy`, `plyfile`, `trimesh`, `tqdm`, `zstandard`, `easydict`, all
   MIT/Apache/BSD).
3. `trellis2_texturing.py` (the PBR/GLB texturing pipeline) and `o_voxel.postprocess` /
   `pbr_mesh_renderer.py` (lines importing `nvdiffrec_render.light.EnvironmentLight` too — a
   second NC dependency, `nvdiffrec`) remain genuinely entangled with nvdiffrast/nvdiffrec —
   **we don't need this stage** (DECISIONS.md: appearance guidance via MV-Adapter + our own
   renderer for distillation views). Do not vendor `trellis2_texturing.py`,
   `pbr_mesh_renderer.py`, or unpatched `o_voxel`.

## 3. Combined verdict for our two use cases

| Use case | Status | Action |
|---|---|---|
| **(a) TRELLIS v1 gaussian-head inference (L2)** | ✅ **Clean as-is** — no fork needed. Avoid importing `trellis.utils.render_utils` / `postprocessing_utils` (use our own gsplat preview renderer, per existing plan). FlexiCubes/Kaolin in the default `import trellis` path are Apache-2.0, not a concern. | Use `microsoft/TRELLIS` upstream directly for L2. |
| **(b) TRELLIS.2 O-Voxel geometry decode (L3 supervision)** | 🟡 **Clean after a 1-line vendored patch** to `o_voxel/__init__.py` (make `postprocess` import lazy). Geometry math itself (`fdg_vae.py`, `flexible_dual_grid_to_mesh`, `cumesh`, `flex_gemm`) has zero NC deps. | Fork `o_voxel` (MIT), patch `__init__.py`, vendor the patched fork + `trellis2.models.sc_vaes.fdg_vae` only. Do NOT vendor `trellis2_texturing.py` / `pbr_mesh_renderer.py` / unpatched `o_voxel`. |

Net result: **both planned use cases are commercially clean**, (b) conditional on a trivial,
license-compatible (MIT→MIT) one-line fork patch we control and can document/upstream.

## 4. Other items closed this session (web research)

- **SOG reference implementation**: `playcanvas/splat-transform` — **MIT** (PlayCanvas Ltd,
  2011-2026), provides the reference SOG writer; PlayCanvas Engine provides the reference
  loader. ✅ ship-safe.
- **Spark viewer SH-degree support**: Spark (`sparkjsdev/spark`, **MIT**, World Labs
  Technologies copyright) supports **SH0/SH1/SH2/SH3** in `PackedSplats` (8/16/16 bytes resp.,
  up to 56 bytes/splat total with full SH3). No documented hard splat-count ceiling for 1M+
  assets — encoding ranges (`splatEncoding`) are tunable. ✅ Spark supports our full L3/L4 SH
  output; no band-limit blocker.
- **VGGT-1B checkpoint license status**: the *original* `facebook/VGGT-1B` checkpoint remains
  **non-commercial**. A new **`facebook/VGGT-1B-Commercial`** checkpoint exists (since ~Jul
  2025) under Meta's "vggt-aup-license" (research-materials license + Acceptable Use Policy;
  commercial use permitted, AUP prohibits weapons/military/ITAR, fraud, illegal activity, etc.).
  ✅ **`VGGT-1B-Commercial` is ship-safe for our stack** (our use case — 3D reconstruction — is
  not on the AUP prohibited list), but it's a *separate, gated* checkpoint from the original
  `VGGT-1B` — must explicitly request/use the `-Commercial` weights, never the base `VGGT-1B`.
  This doesn't change our chosen capture front-end (MapAnything `-apache`, ✅ already), but
  unblocks VGGT as a future runner-up option if ever revisited.

## 5. Cleanup

Both clones (`%TEMP%\astel-audit\trellis-v1`, `trellis-v2`) and the ad-hoc FlexiCubes check
clone deleted at end of session. No artifacts left outside `%TEMP%`. No git commands run inside
the AURIGA repo.
