# astel-scene — scene seeds / scene composition

Composes multiple single-object Gaussian-splat clouds into a small
**multi-object scene** with ground-contact placement and no-overlap resolution
(CLAUDE.md §8 "Scene seeds").

**Torch-free, CPU-only.** Operates on raw numpy arrays — no SplatCloud import
needed, keeping this library dependency-light and CI-fast.

## Asset convention

Each object is represented by five raw arrays (`ObjectSplats`):

| Field | Shape | Description |
|-------|-------|-------------|
| `positions` | (N,3) float32 | Gaussian means |
| `quats` | (N,4) float32 | Rotation quaternions **(w,x,y,z)** |
| `log_scales` | (N,3) float32 | log of per-axis scale |
| `opacity` | (N,) float32 | Opacity values |
| `colors_dc` | (N,3) float32 | SH band-0 colour |

## Pipeline

```
SceneLayout (JSON)
   ↓
apply_placement(obj, placement)   — yaw + uniform scale + translate (transform.py)
   ↓
ground_drop(obj, ground_y)        — drop onto ground plane (contacts.py)
   ↓
resolve_no_overlap(objects)       — XZ-plane AABB push-apart (contacts.py)
   ↓
compose_scene(...)                — concatenate + return (start,end) index ranges
```

## Layout format

`SceneLayout` serialises to JSON with schema key `"astel.scene-layout/v0"`.
`write_json` / `read_json` round-trip losslessly.

## Honesty

If two objects cannot be placed without overlap under the constraints given
(e.g. `padding` forces an impossible arrangement), `resolve_no_overlap` does
**not** silently interpenetrate. It pushes objects apart greedily in input
order and the caller receives the actual (possibly still-tight-fitting) result
with no hidden silencing.
