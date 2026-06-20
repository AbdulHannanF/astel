"""Scene composition — assemble a multi-object splat scene.

``compose_scene`` is the top-level entry point.  It:

1. Validates that ``objects`` and ``layout.objects`` have the same length.
2. For each object (matched by position in the list):
   a. Applies the :class:`~astel_scene.layout.Placement` rigid transform.
   b. Drops the object onto the ground plane (if ``ground_contact`` is set
      on the placement **and** ``apply_ground_contact`` is True).
3. Optionally calls :func:`~astel_scene.contacts.resolve_no_overlap` across
   all placed objects.
4. Concatenates all arrays into a single combined
   :class:`~astel_scene.splats.ObjectSplats` and returns the per-object
   ``(start, end)`` index ranges.
"""

from __future__ import annotations

import numpy as np

from .contacts import ground_drop, resolve_no_overlap
from .layout import SceneLayout
from .splats import ObjectSplats
from .transform import apply_placement


def compose_scene(
    objects: list[ObjectSplats],
    layout: SceneLayout,
    *,
    apply_ground_contact: bool = True,
    resolve_overlap: bool = True,
) -> tuple[ObjectSplats, list[tuple[int, int]]]:
    """Compose multiple objects into one combined splat scene.

    Parameters
    ----------
    objects:
        List of per-object :class:`~astel_scene.splats.ObjectSplats`, one
        entry per :class:`~astel_scene.layout.SceneObject` in *layout*.  The
        lists are matched **by index** (parallel lists).
    layout:
        :class:`~astel_scene.layout.SceneLayout` describing placements,
        coordinate convention, and ground-plane Y.
    apply_ground_contact:
        When *True* (default), objects whose
        :attr:`~astel_scene.layout.Placement.ground_contact` flag is set are
        dropped onto ``layout.ground_y`` after the rigid transform.
    resolve_overlap:
        When *True* (default), :func:`~astel_scene.contacts.resolve_no_overlap`
        is called across all placed objects to separate overlapping XZ AABBs.

    Returns
    -------
    (combined, ranges)
        *combined* — a single :class:`~astel_scene.splats.ObjectSplats`
        whose arrays are the concatenation of all placed objects (in input
        order).

        *ranges* — ``list[tuple[int, int]]`` of ``(start, end)`` index pairs,
        one per input object, such that
        ``combined.positions[start:end]`` recovers that object's splats.
        Ranges are contiguous and partition ``[0, total_splat_count)``.

    Raises
    ------
    ValueError
        If ``len(objects) != len(layout.objects)``.
    """
    if len(objects) != len(layout.objects):
        raise ValueError(
            f"objects list length ({len(objects)}) does not match "
            f"layout.objects length ({len(layout.objects)})."
        )

    # --- Step 1 & 2: apply placement + optional ground drop ---
    placed: list[ObjectSplats] = []
    for obj, scene_obj in zip(objects, layout.objects, strict=True):
        transformed = apply_placement(obj, scene_obj.placement)
        if apply_ground_contact and scene_obj.placement.ground_contact:
            transformed = ground_drop(transformed, layout.ground_y)
        placed.append(transformed)

    # --- Step 3: optional no-overlap resolution ---
    if resolve_overlap:
        placed = resolve_no_overlap(placed)

    # --- Step 4: concatenate ---
    ranges: list[tuple[int, int]] = []
    cursor = 0
    for p in placed:
        n = p.count
        ranges.append((cursor, cursor + n))
        cursor += n

    combined = ObjectSplats(
        positions=np.concatenate([p.positions for p in placed], axis=0).astype(
            np.float32
        ),
        quats=np.concatenate([p.quats for p in placed], axis=0).astype(np.float32),
        log_scales=np.concatenate([p.log_scales for p in placed], axis=0).astype(
            np.float32
        ),
        opacity=np.concatenate([p.opacity for p in placed], axis=0).astype(np.float32),
        colors_dc=np.concatenate([p.colors_dc for p in placed], axis=0).astype(
            np.float32
        ),
    )

    return combined, ranges
