# Coordinate-convention reference

> Canonical doc for all Astel engine integrations. Every plugin, SDK, and
> export function must use the transforms defined here. Round-trip tests live
> in `libs/astel_splat_io/tests/test_conventions.py`.

## 1. 3DGS world space (Astel canonical)

The training pipeline and all internal representations use:

| Property | Value |
|---|---|
| Handedness | Right-handed |
| Up axis | +Y |
| Forward axis | +Z (toward viewer / out of screen) |
| Units | metres |
| Quaternion order | `(w, x, y, z)` |

This matches gsplat's default and object-centric COLMAP reconstructions.
Captured scenes (real-world video/photo) are normalised to this frame at L0.

---

## 2. glTF 2.0 / KHR_gaussian_splatting

| Property | Value |
|---|---|
| Handedness | Right-handed |
| Up axis | +Y |
| Forward axis | −Z |
| Units | metres |
| Quaternion order | `(x, y, z, w)` |

**Position transform:** identity — same orientation as 3DGS world.

**Quaternion transform:** reorder only — `(w,x,y,z) → (x,y,z,w)`.

```python
from astel_splat_io.conventions import gltf_positions, gltf_quats
```

---

## 3. Unity (URP / Built-in RP)

| Property | Value |
|---|---|
| Handedness | Left-handed |
| Up axis | +Y |
| Forward axis | +Z |
| Units | metres |
| Quaternion order (Unity API) | `(x, y, z, w)` |

**Position transform:** negate X.

```
pos_unity = (−x,  y,  z)
```

**Quaternion transform:** reorder + negate x and w.

```
q_unity = (−qx,  qy,  qz,  −qw)
```

**Why `−qx` and `−qw`?**  Reflecting the X axis flips the handedness of the
coordinate frame.  The equivalent quaternion for the same physical rotation in a
left-handed frame has `qx` and `qw` negated (this is the standard formula for
conjugating a reflection through the YZ plane into quaternion space).

```python
from astel_splat_io.conventions import unity_positions, unity_quats
```

**Unity import note:** aras-p UnityGaussianSplatting imports `.ply` / `.spz`
natively; the Astel Unity package calls the importer then applies these
transforms and auto-configures physics from the `.astel` manifest.

---

## 4. Unreal Engine 5

| Property | Value |
|---|---|
| Handedness | Left-handed |
| Up axis | +Z |
| Forward axis | +X |
| Units | centimetres |
| Quaternion order (FQuat) | `(x, y, z, w)` |

**Position transform:**

```
pos_ue = (−z × 100,   x × 100,   y × 100)   [cm]
```

Axis mapping: 3DGS `(x, y, z)` → UE5 `(−z, x, y)`, then ×100 for cm.

**Quaternion transform:**

```
q_ue = (−qz,  qx,  qy,  −qw)
```

This encodes the same axis remap applied to the rotation's frame.

**Scale transform:**

```
scale_ue = scale_world × 100   [cm]
```

```python
from astel_splat_io.conventions import unreal_positions, unreal_quats, unreal_scales
```

---

## 5. Summary table

| Target | pos_x | pos_y | pos_z | units | quat order | quat transform |
|---|---|---|---|---|---|---|
| 3DGS world | x | y | z | m | w,x,y,z | — |
| glTF | x | y | z | m | x,y,z,w | reorder |
| Unity | −x | y | z | m | x,y,z,w | −qx, qy, qz, −qw |
| Unreal | −z×100 | x×100 | y×100 | cm | x,y,z,w | −qz, qx, qy, −qw |

---

## 6. Scale convention

3DGS stores `log_scales` (log of world-space σ in each axis).  Engine plugins
and the glTF exporter convert to linear world-space scale:

```python
scales_world = np.exp(log_scales)      # metres, per-axis σ
scales_ue    = scales_world * 100      # cm for Unreal
```

---

## 7. Checklist for new engine targets

When adding a new engine, verify:

1. Handedness of the target coordinate system.
2. Up / forward axis mapping.
3. Unit system (metres / cm / arbitrary).
4. How the engine represents quaternions (order, convention).
5. Write a round-trip fixture test in `test_conventions.py`.
6. Document the transform here and in the plugin's own README.
