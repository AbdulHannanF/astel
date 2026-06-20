"""Rigid placement of a single object's Gaussian splats.

Quaternion convention throughout: **(w, x, y, z)** — the Astel internal order.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from .layout import Placement
from .splats import ObjectSplats


def quat_from_yaw(yaw_deg: float) -> NDArray[np.float32]:
    """Unit quaternion for a rotation of *yaw_deg* degrees about the +Y axis.

    The +Y axis is Astel's "up" axis.  A positive yaw rotates a point on the
    +X axis towards the **−Z** axis (right-hand rule about +Y).

    Returns
    -------
    (4,) float32 in **(w, x, y, z)** order.
    """
    half = math.radians(yaw_deg) * 0.5
    w = math.cos(half)
    y = math.sin(half)
    return np.array([w, 0.0, y, 0.0], dtype=np.float32)


def quat_multiply(
    a: NDArray[np.float32],
    b: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Hamilton product of two unit quaternions in **(w, x, y, z)** order.

    Supports broadcasting: *a* may be (4,) or (N, 4) and *b* may be (4,) or
    (N, 4).  The result has the shape produced by numpy broadcasting of the
    leading dimensions.

    Parameters
    ----------
    a, b:
        Quaternion(s) in (w, x, y, z) order.

    Returns
    -------
    Product quaternion(s) in (w, x, y, z) order, float32.
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]

    rw = aw * bw - ax * bx - ay * by - az * bz
    rx = aw * bx + ax * bw + ay * bz - az * by
    ry = aw * by - ax * bz + ay * bw + az * bx
    rz = aw * bz + ax * by - ay * bx + az * bw

    return np.stack([rw, rx, ry, rz], axis=-1).astype(np.float32)


def _rotate_positions_by_yaw(
    positions: NDArray[np.float32],
    yaw_deg: float,
) -> NDArray[np.float32]:
    """Rotate (N, 3) positions about the +Y axis by *yaw_deg* degrees."""
    theta = math.radians(yaw_deg)
    c = math.cos(theta)
    s = math.sin(theta)
    # Rotation matrix about +Y (right-hand rule):
    #   [c,  0, s]   [x]
    #   [0,  1, 0] × [y]
    #   [-s, 0, c]   [z]
    rot = np.array(
        [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
        dtype=np.float32,
    )
    return (positions @ rot.T).astype(np.float32)


def apply_placement(obj: ObjectSplats, placement: Placement) -> ObjectSplats:
    """Apply a rigid :class:`~astel_scene.layout.Placement` to *obj*.

    The transform is applied in this order:

    1. **Yaw** — rotate positions and quaternions about the +Y axis by
       ``placement.yaw_deg``.
    2. **Scale** — multiply positions by ``placement.uniform_scale``; add
       ``log(uniform_scale)`` to every element of ``log_scales`` (uniform
       scaling shifts all log-scales identically).
    3. **Translate** — add ``placement.translation`` to positions.

    Colours and opacity are **unchanged** by a rigid placement.

    The gaussian orientation quaternions are updated by **left-multiplying**
    the yaw quaternion: ``q_new = q_yaw ⊗ q_old``.  Left-multiplication
    applies the yaw in *world space* (the body then rotates within the already-
    rotated frame), which is the correct semantic for an artist placing an
    object in a scene.

    Parameters
    ----------
    obj:
        Source splats (not mutated).
    placement:
        Rigid placement parameters.

    Returns
    -------
    A new :class:`ObjectSplats` with updated positions, quats, and log_scales.
    """
    # --- 1. Yaw rotation ---
    pos = _rotate_positions_by_yaw(obj.positions, placement.yaw_deg)
    q_yaw = quat_from_yaw(placement.yaw_deg)  # (4,)
    # Left-multiply every per-splat quat: q_new = q_yaw ⊗ q_old
    quats = quat_multiply(q_yaw, obj.quats)  # (N, 4)

    # --- 2. Uniform scale ---
    s = float(placement.uniform_scale)
    pos = (pos * s).astype(np.float32)
    log_s_delta = math.log(s) if s > 0.0 else 0.0
    log_scales = (obj.log_scales + log_s_delta).astype(np.float32)

    # --- 3. Translate ---
    t = np.array(placement.translation, dtype=np.float32)
    pos = (pos + t).astype(np.float32)

    return ObjectSplats(
        positions=pos,
        quats=quats,
        log_scales=log_scales,
        opacity=obj.opacity,
        colors_dc=obj.colors_dc,
    )
