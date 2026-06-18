# Splats 101 for studios

A practical guide to Gaussian splatting for artists, TDs, and engineers who work in VFX, games, or real-time rendering.

## What is a Gaussian splat?

A 3D Gaussian splat is a scene representation where the geometry and appearance are encoded as a cloud of oriented ellipsoids ("Gaussians"). Each Gaussian has:

- **Position** (xyz) — where it lives in space
- **Scale** (3 axes) — how big it is in each direction
- **Rotation** (quaternion) — how it is oriented
- **Colour** (SH coefficients) — view-dependent colour via Spherical Harmonics
- **Opacity** — how transparent it is

When rendered, each Gaussian is projected onto screen space as a 2D splat, and all visible splats are alpha-composited in depth order. The result is a real-time-renderable scene captured from photos, video, or generated from text.

## Why splats, not meshes?

| Property | Mesh | Gaussian splat |
|---|---|---|
| Captures fuzzy/organic surfaces | Poor | Excellent |
| Real-time render in browser | Yes (via rasterisation) | Yes (via splat rasteriser) |
| Photo-realistic from photos | Requires texturing + baking | Direct from photos |
| Watertight for physics/print | Yes | Via SDF extraction (L5) |
| LOD / streaming | Via mesh simplification | Via hierarchical splats |

Splats are not a replacement for meshes in all contexts — they excel at *appearance* and *real-world capture*. Astel's L5 layer extracts a watertight surface internally for physics and printing, but the deliverable stays splats.

## The `.astel` format

A generation produces a self-contained `.astel` **package** plus a set of **sibling delivery artifacts** (served individually by the API):

```
package.astel/                       # the zip package (mimetype + manifest + layers)
├── manifest.json                    # the manifest (layer registry, scale, provenance, quality report)
└── layers/
    ├── l0_seed/points.ply           # sparse seed point cloud
    ├── l3_refined/splats.ply        # refined surface splats (INRIA PLY master)
    ├── l4_appearance/…              # albedo PLY + estimated SH environment
    ├── l5_collision/                # watertight surface (.stl/.3mf), convex set (.glb/.npz), l5-mass.json
    └── l6_physics/l6.json           # per-region material + articulation

sibling delivery artifacts (download alongside the package):
├── l3.spz / l3.sog                  # compressed splats (Niantic SPZ / PlayCanvas SOG)
├── l3.glb                           # KHR_gaussian_splatting glTF (RC) — broad interop
├── engine.json                      # flat physics-setup descriptor for the Unity/UE5 plugins
└── quality-report.json              # Truth Meter — geometric error, confidence, origin
```

## Coordinate conventions

See [Coordinate conventions](coordinate-conventions.md) for the exact transforms to apply when importing into Unity, Unreal Engine 5, or any other engine.

Short version:

| Engine | Up | Handedness | Units | Key transform |
|---|---|---|---|---|
| 3DGS world (Astel) | +Y | Right | metres | — |
| glTF / Web | +Y | Right | metres | Quat reorder only |
| Unity | +Y | Left | metres | Negate X |
| Unreal | +Z | Left | cm | Remap axes × 100 |

## Rendering in the browser

Astel's viewer uses [Spark](https://github.com/huggingface/gsplat.js) (Three.js based). Any Three.js or PlayCanvas scene can load `.ply` or `.spz` directly with a splat renderer. The `KHR_gaussian_splatting` glTF extension (RC Feb 2026) enables engine-native import.

## Physically based relighting (L4)

Astel decomposes each splat's baked colour into:

- **Albedo** — intrinsic surface colour
- **SH environment** — low-frequency estimated illumination

With these separate, you can relight the asset by supplying a different HDRI environment. The Relight Studio in the viewer does this live. For engine use, export the albedo PLY and treat the environment map as an emissive contribution.

## Physics (L5 + L6)

- **L5** provides a convex hull set (CoACD multi-hull, or scipy single-hull fallback for thin objects), mass, centre of mass, and inertia tensor.
- **L6** provides per-region material properties (density, friction, restitution) and articulation hints.

The Unity and UE5 plugins auto-configure Rigidbody/UBodySetup from the flat `engine.json` sidecar (which denormalises the L5/L6 data). For custom engines, either read `engine.json` (simplest) or parse `manifest.json` and follow its `l5`/`l6` file references.
