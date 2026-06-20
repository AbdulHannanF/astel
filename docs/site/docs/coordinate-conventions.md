# Coordinate conventions

This page summarises the engine-interop transforms defined in
`docs/architecture/coordinate-conventions.md` and implemented in
`libs/astel_splat_io/src/astel_splat_io/conventions.py`.

## Quick-reference

| Target | pos_x | pos_y | pos_z | units | quat (out) |
|---|---|---|---|---|---|
| 3DGS world | x | y | z | m | w,x,y,z |
| glTF 2.0 | x | y | z | m | x,y,z,w (reorder) |
| Unity | −x | y | z | m | −qx, qy, qz, −qw |
| Unreal Engine 5 | −z×100 | x×100 | y×100 | cm | −qz, qx, qy, −qw |

For the full derivation and round-trip tests see the repo reference doc at
`docs/architecture/coordinate-conventions.md`.

## Python helpers

```python
from astel_splat_io.conventions import (
    gltf_positions, gltf_quats,
    unity_positions, unity_quats,
    unreal_positions, unreal_quats, unreal_scales,
)

# Convert a SplatCloud's positions to Unity space
pos_unity = unity_positions(cloud.positions)
q_unity   = unity_quats(cloud.quats)   # returns (x,y,z,w) Unity order

# UE5
pos_ue = unreal_positions(cloud.positions)  # cm
q_ue   = unreal_quats(cloud.quats)
s_ue   = unreal_scales(np.exp(cloud.log_scales))
```
