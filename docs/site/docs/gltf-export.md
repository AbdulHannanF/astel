# glTF export (KHR_gaussian_splatting)

Astel exports `.glb` files with the `KHR_gaussian_splatting` extension (RC schema, Khronos, Feb 2026).

## Quick start

```python
from astel_splat_io import write_gltf, read_ply

cloud = read_ply("l3.ply")
write_gltf(cloud, "asset.glb")
```

## What's in the file

Each Gaussian is stored as a vertex in a POINTS mesh primitive:

| Attribute | Type | Description |
|---|---|---|
| `POSITION` | VEC3 float | Gaussian centre (world-space metres) |
| `_ROTATION` | VEC4 float | Unit quaternion `(x,y,z,w)` |
| `_SCALE` | VEC3 float | World-space σ per axis (exp of log_scale) |
| `COLOR_0` | VEC4 float | `(r,g,b,α)` in [0,1] from SH band-0 + sigmoid opacity |

The extension is declared:
```json
{
  "extensionsUsed": ["KHR_gaussian_splatting"],
  "meshes": [{ "primitives": [{ "extensions": { "KHR_gaussian_splatting": { "sh_degree": 0 } } }] }]
}
```

## Coordinate convention

3DGS world and glTF 2.0 are both right-handed, +Y up — so **no position transform is applied**. The only change is quaternion reorder: `(w,x,y,z) → (x,y,z,w)`.

Round-trip is lossless to float32 precision.

## Status

The `KHR_gaussian_splatting` extension was a **release candidate** as of Feb 2026. Ratification was expected Q2 2026. Re-verify the schema before shipping assets to external parties:

```
https://www.khronos.org/blog/gaussian-splatting-in-gltf
```

The exporter flags exports as RC via `"generator": "astel-splat-io"` in the asset metadata. Update the generator string after ratification to indicate conformance.

## API

```python
from astel_splat_io import write_gltf, read_gltf
from astel_splat_io import SplatCloud

# Write
n_bytes = write_gltf(cloud, "out.glb")

# Read back
cloud = read_gltf("out.glb")  # raises ValueError if extension missing
```
