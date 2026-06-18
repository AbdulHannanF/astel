# astel-solid — L5 solidification & print path

Derives the **L5 collision/solidity layer** (CLAUDE.md §3) from an oriented splat
cloud: signed distance field → watertight isosurface → mass properties (volume,
centre of mass, inertia tensor) → print exports (`.stl`).

**This surface is internal scaffolding, never a product deliverable.** Per the
binding constraint (§1.2) the user-facing asset is always Gaussian splats; the
watertight mesh exists only to (a) drive the 3D-print path, (b) define the physics
volume (L6), and (c) seed engine collision proxies. No mesh is offered as the
asset.

Pure CPU / torch-free, so it runs in CI and validates against analytic
ground truth (a sampled sphere recovers `4/3·π·r³`, an isotropic inertia tensor,
and a centred COM within tolerance).

## Pipeline

```
positions + outward normals
   → oriented-point IMLS SDF on a voxel grid   (sdf.py, scipy KDTree)
   → marching cubes at the zero level set       (isosurface.py, skimage)
   → watertight triangle mesh (outward-wound)
   → mass properties via the divergence theorem (mass.py, pure numpy)
   → binary .stl                                (stl.py)
   → OPC 3MF archive                            (print3mf.py)
   → convex hull set (CoACD / scipy fallback)   (convex.py)
   → printability report                        (printability.py)
```

`solidify.solidify(...)` ties it together; `surfel_normals(...)` derives per-splat
outward normals from 2DGS quats + log-scales for the producer integration.

## Print exports

| Format | Module | Notes |
|--------|--------|-------|
| `.stl` | `stl.py` | Binary STL, hand-rolled, no deps |
| `.3mf` | `print3mf.py` | OPC zip, 3MF Core spec, hand-rolled stdlib `zipfile` |

## Collision proxies

`convex.py` runs **CoACD** (PyPI `coacd`, MIT) when available; falls back to a
single scipy convex hull. `write_convex_glb` writes `.glb` (trimesh optional) or
`.npz` (numpy-only fallback). `ConvexSet.method` records which path was taken.

## Printability analysis

`printability.analyze_printability(SolidResult)` returns a `PrintabilityReport`
with: `min_wall_model_units`/`min_wall_mm` (from interior SDF), `overhang_fraction`
(area-weighted, configurable threshold), `hollow_volume_fraction` (material savings
estimate), and honesty caveats for discretization bias and scale grounding.
Pass `meters_per_unit` to enable metric (mm) wall-thickness conversion.
