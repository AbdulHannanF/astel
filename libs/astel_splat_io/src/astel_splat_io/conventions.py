"""Coordinate-convention transforms for engine interop.

3DGS world space (our canonical training frame)
------------------------------------------------
Right-handed, +Y up, +Z toward viewer (out of screen).  This matches gsplat's
training convention for object-centric scenes.  COLMAP reconstructions may
vary per dataset; the GPU pipeline normalises to this frame at L0.

glTF 2.0
--------
Right-handed, +Y up, -Z forward.  Identical to the 3DGS frame — no position
transform is needed.  Quaternion axis order: (x,y,z,w) vs our (w,x,y,z).

Unity (Universal Render Pipeline / Built-in)
--------------------------------------------
Left-handed, +Y up, +Z forward.  To convert from 3DGS right-handed:
  • Negate the X-axis of positions: pos_unity = (−x,  y,  z).
  • Flip the handedness of rotations: negate qx and qw.
  • Scale: 1 world-unit = 1 metre (same as glTF).

Unreal Engine 5
---------------
Left-handed, +Z up, +X forward, centimetres.  Full transform from 3DGS:
  • Axes: pos_ue = (−z * 100,  x * 100,  y * 100)  [cm].
  • Rotation: see ``gltf_to_unreal_quat``.
  • Scale: multiply by 100 (metres → centimetres).

See also
--------
``docs/architecture/coordinate-conventions.md`` — canonical reference with
round-trip fixture tests.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Position transforms
# ---------------------------------------------------------------------------


def gltf_positions(pos: NDArray[np.float32]) -> NDArray[np.float32]:
    """Return positions unchanged — 3DGS world == glTF frame."""
    return np.ascontiguousarray(pos, dtype=np.float32)


def unity_positions(pos: NDArray[np.float32]) -> NDArray[np.float32]:
    """Convert 3DGS world positions to Unity left-handed (+Y up, +Z forward).

    Negate the X axis to flip handedness.  Y and Z are unchanged.
    """
    out = pos.copy().astype(np.float32)
    out[:, 0] = -pos[:, 0]
    return out


def unreal_positions(pos: NDArray[np.float32]) -> NDArray[np.float32]:
    """Convert 3DGS world positions to UE5 left-handed (+Z up, +X forward, cm).

    Mapping:   (x, y, z) → (−z*100,  x*100,  y*100)
    """
    out = np.empty_like(pos)
    out[:, 0] = -pos[:, 2] * 100.0
    out[:, 1] = pos[:, 0] * 100.0
    out[:, 2] = pos[:, 1] * 100.0
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Quaternion transforms
# Input convention: (w, x, y, z) normalised, right-handed 3DGS world.
# ---------------------------------------------------------------------------


def gltf_quats(quats: NDArray[np.float32]) -> NDArray[np.float32]:
    """Reorder (w,x,y,z) → (x,y,z,w) for glTF.  No axis flip."""
    w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    return np.column_stack([x, y, z, w]).astype(np.float32)


def unity_quats(quats: NDArray[np.float32]) -> NDArray[np.float32]:
    """Convert (w,x,y,z) 3DGS quat to Unity left-handed (x,y,z,w).

    Flip from right-handed to left-handed: negate x and w (equivalent to
    reflecting through the YZ plane, consistent with negating positions.x).
    """
    w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    # Unity stores (x,y,z,w); negate x and w for handedness flip.
    return np.column_stack([-x, y, z, -w]).astype(np.float32)


def unreal_quats(quats: NDArray[np.float32]) -> NDArray[np.float32]:
    """Convert (w,x,y,z) 3DGS quat to UE5 left-handed (x,y,z,w).

    UE5 coordinate mapping: world axes (X→−Z, Y→X, Z→Y) rotated.
    The quaternion permutation for the axis reassignment (x,y,z)→(−z,x,y) is:
      q_ue = q_perm where q_perm encodes the same frame rotation.

    Applied as: rotate by the frame-change rotation, then flip handedness.
    """
    w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    # Axis remap (x,y,z) → (−z, x, y):  new_x=−z, new_y=x, new_z=y
    # Corresponding quaternion component remap + handedness negate:
    return np.column_stack([-z, x, y, -w]).astype(np.float32)


# ---------------------------------------------------------------------------
# Scale transforms
# Input: world-space σ (exp of log_scale), isotropic or anisotropic vec3.
# ---------------------------------------------------------------------------


def unreal_scales(scales: NDArray[np.float32]) -> NDArray[np.float32]:
    """Convert world-space scales to UE5 centimetres (multiply by 100)."""
    return (scales * 100.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Round-trip helpers
# ---------------------------------------------------------------------------


def round_trip_positions(
    pos: NDArray[np.float32],
    target: str,
) -> NDArray[np.float32]:
    """Apply the forward transform for ``target`` ('gltf', 'unity', 'unreal').

    For testing: the inverse of each forward transform restores the input.
    """
    if target == "gltf":
        return gltf_positions(pos)
    if target == "unity":
        return unity_positions(pos)
    if target == "unreal":
        return unreal_positions(pos)
    raise ValueError(f"unknown target {target!r}")
