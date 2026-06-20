"""Ground-contact placement and XZ no-overlap resolution.

Design notes
------------
**1st-percentile ground drop**
    ``ground_drop`` uses the 1st percentile of Y-coordinates rather than the
    strict minimum.  Reconstructed splat clouds often contain a handful of
    stray low-weight gaussians well below the object's true base (noise from
    the reconstruction or background leakage).  Using the strict minimum would
    sink the object into the ground by the depth of those outliers.  The 1st
    percentile is a robust low-floor that ignores the bottom ~1 % of points
    while keeping the object visually sitting on the ground plane.  Users who
    need an exact floor should pass pre-cleaned splats.

**XZ no-overlap (AABB push-apart)**
    ``resolve_no_overlap`` operates on the XZ ground-plane projection (Y is
    up, so XZ is the floor plane).  It processes objects in *input order*
    (deterministic, stable under repeated calls).  For each pair of
    consecutive already-placed object and new candidate:

    1. Compute the XZ AABB (+ padding) of every already-placed object.
    2. If the candidate's XZ AABB overlaps any of them, push it along the
       +X axis by the minimum distance needed to clear the overlap.

    This greedy left-to-right push is O(N²) in the number of objects, which
    is fine for "scene seeds" (≤ ~10 objects per scene).  It is **not** a
    global optimiser — the result is honest: objects are not silently
    interpenetrated; if the push makes the arrangement look sparse, that is
    reported faithfully in the positions.

    HONESTY (CLAUDE.md §10.4): the algorithm makes no attempt to fill the
    resulting gaps or rotate objects to pack them more tightly.  It pushes
    greedily and returns the result.  Callers who need tighter packing should
    run a layout-LLM pass (WP2-wire) upstream.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .splats import ObjectSplats


def aabb(obj: ObjectSplats) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Axis-aligned bounding box of *obj*.

    Returns
    -------
    (min_corner, max_corner)
        Each is a ``(3,)`` float32 array ``[x_min/max, y_min/max, z_min/max]``.
    """
    lo: NDArray[np.float32] = obj.positions.min(axis=0).astype(np.float32)
    hi: NDArray[np.float32] = obj.positions.max(axis=0).astype(np.float32)
    return lo, hi


def ground_drop(obj: ObjectSplats, ground_y: float = 0.0) -> ObjectSplats:
    """Translate *obj* along +Y so its base sits at *ground_y*.

    The "base" is defined as the **1st percentile** of Y-coordinates across
    all splats.  This robust low-floor ignores the bottom ~1 % of stray splats
    that often sit below the object's true base in reconstructed clouds.  See
    module docstring for the full rationale.

    Parameters
    ----------
    obj:
        Source splats (not mutated).
    ground_y:
        Target Y-coordinate for the object's base.

    Returns
    -------
    A new :class:`ObjectSplats` translated so that ``percentile(y, 1) == ground_y``.
    """
    y_low = float(np.percentile(obj.positions[:, 1], 1.0))
    dy = ground_y - y_low
    new_pos = obj.positions.copy()
    new_pos[:, 1] += dy
    return ObjectSplats(
        positions=new_pos.astype(np.float32),
        quats=obj.quats,
        log_scales=obj.log_scales,
        opacity=obj.opacity,
        colors_dc=obj.colors_dc,
    )


def _xz_aabb(
    obj: ObjectSplats,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """XZ footprint of *obj* — (min_xz, max_xz) each (2,) float32."""
    xz = obj.positions[:, [0, 2]]
    return xz.min(axis=0).astype(np.float32), xz.max(axis=0).astype(np.float32)


def _xz_overlap_x_push(
    lo_a: NDArray[np.float32],
    hi_a: NDArray[np.float32],
    lo_b: NDArray[np.float32],
    hi_b: NDArray[np.float32],
    padding: float,
) -> float:
    """Return the +X translation for B to clear A's padded XZ AABB.

    Returns 0.0 if there is no overlap.
    """
    # Padded A extents
    lo_a_p = lo_a - padding
    hi_a_p = hi_a + padding

    # Check for XZ overlap (AABB ∩ AABB)
    if lo_b[0] > hi_a_p[0] or hi_b[0] < lo_a_p[0]:
        return 0.0  # no overlap in X
    if lo_b[1] > hi_a_p[1] or hi_b[1] < lo_a_p[1]:
        return 0.0  # no overlap in Z

    # There is overlap; compute push distance in +X
    push = float(hi_a_p[0] - lo_b[0])
    return max(push, 0.0)


def resolve_no_overlap(
    objects: list[ObjectSplats],
    *,
    padding: float = 0.0,
) -> list[ObjectSplats]:
    """Greedily push objects apart in the XZ ground plane.

    Processes objects in *input order*.  The first object is anchored; each
    subsequent object is pushed in the **+X direction** by the minimum amount
    needed to clear all previously placed objects' padded XZ AABBs.

    This is a greedy O(N²) algorithm suitable for small scenes (≤ ~10
    objects).  See module docstring for design rationale and honesty notes.

    Parameters
    ----------
    objects:
        List of splat objects; each element must be an :class:`ObjectSplats`.
    padding:
        Extra clearance (in world units) added on all sides of each object's
        XZ footprint before the overlap check.

    Returns
    -------
    A new list of :class:`ObjectSplats`, repositioned so that their XZ AABBs
    (expanded by *padding*) do not overlap.  The input list is not mutated.
    """
    if not objects:
        return []

    placed: list[ObjectSplats] = [objects[0]]
    placed_xz: list[tuple[NDArray[np.float32], NDArray[np.float32]]] = [
        _xz_aabb(objects[0])
    ]

    for obj in objects[1:]:
        lo_b, hi_b = _xz_aabb(obj)
        total_push = 0.0
        for lo_a, hi_a in placed_xz:
            push = _xz_overlap_x_push(
                lo_a, hi_a, lo_b + total_push, hi_b + total_push, padding
            )
            total_push += push

        if total_push != 0.0:
            new_pos = obj.positions.copy()
            new_pos[:, 0] += total_push
            new_obj = ObjectSplats(
                positions=new_pos.astype(np.float32),
                quats=obj.quats,
                log_scales=obj.log_scales,
                opacity=obj.opacity,
                colors_dc=obj.colors_dc,
            )
        else:
            new_obj = obj

        placed.append(new_obj)
        placed_xz.append(_xz_aabb(new_obj))

    return placed
