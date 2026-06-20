"""Contact tests: ground_drop and resolve_no_overlap."""

from __future__ import annotations

import numpy as np
from _objects import box_object

from astel_scene.contacts import aabb, ground_drop, resolve_no_overlap
from astel_scene.splats import ObjectSplats

# ---------------------------------------------------------------------------
# ground_drop
# ---------------------------------------------------------------------------


def test_ground_drop_1pct_y_at_ground_y() -> None:
    """After ground_drop, the 1st-percentile Y should equal ground_y."""
    obj = box_object(1000, center=(0.0, 5.0, 0.0), size=(2.0, 4.0, 2.0), seed=0)
    dropped = ground_drop(obj, ground_y=0.0)
    y_low = float(np.percentile(dropped.positions[:, 1], 1.0))
    assert abs(y_low - 0.0) < 1e-4, f"1st-pct y = {y_low}, expected 0.0"


def test_ground_drop_custom_ground_y() -> None:
    obj = box_object(500, center=(0.0, 0.0, 0.0), size=(1.0, 1.0, 1.0), seed=1)
    target = -3.0
    dropped = ground_drop(obj, ground_y=target)
    y_low = float(np.percentile(dropped.positions[:, 1], 1.0))
    assert abs(y_low - target) < 1e-4


def test_ground_drop_stray_outlier_does_not_sink_object() -> None:
    """A single stray splat far below the object must not drag the base down."""
    obj = box_object(200, center=(0.0, 1.0, 0.0), size=(1.0, 1.0, 1.0), seed=2)
    # Inject one stray splat 100 units below
    stray_pos = np.vstack([obj.positions, [[0.0, -100.0, 0.0]]])
    stray_quats = np.vstack([obj.quats, [[1.0, 0.0, 0.0, 0.0]]])
    stray_ls = np.vstack([obj.log_scales, [[0.0, 0.0, 0.0]]])
    stray_op = np.append(obj.opacity, 1.0)
    stray_dc = np.vstack([obj.colors_dc, [[0.5, 0.5, 0.5]]])
    obj_with_stray = ObjectSplats(
        positions=stray_pos.astype(np.float32),
        quats=stray_quats.astype(np.float32),
        log_scales=stray_ls.astype(np.float32),
        opacity=stray_op.astype(np.float32),
        colors_dc=stray_dc.astype(np.float32),
    )
    dropped = ground_drop(obj_with_stray, ground_y=0.0)
    # The 5th-percentile Y should be near 0 (not dragged to -100)
    y_5pct = float(np.percentile(dropped.positions[:, 1], 5.0))
    assert y_5pct > -1.0, f"Object was wrongly dragged by outlier: 5th-pct y={y_5pct}"


def test_ground_drop_preserves_xz_and_other_fields() -> None:
    obj = box_object(100, seed=3)
    dropped = ground_drop(obj, ground_y=0.0)
    np.testing.assert_allclose(
        dropped.positions[:, [0, 2]], obj.positions[:, [0, 2]], atol=1e-6
    )
    np.testing.assert_array_equal(dropped.quats, obj.quats)
    np.testing.assert_array_equal(dropped.log_scales, obj.log_scales)
    np.testing.assert_array_equal(dropped.opacity, obj.opacity)
    np.testing.assert_array_equal(dropped.colors_dc, obj.colors_dc)


# ---------------------------------------------------------------------------
# aabb
# ---------------------------------------------------------------------------


def test_aabb_matches_known_box() -> None:
    """aabb of a box centred at origin with size 2 should be (-1,-1,-1)/(1,1,1)."""
    obj = box_object(2000, center=(0.0, 0.0, 0.0), size=(2.0, 2.0, 2.0), seed=10)
    lo, hi = aabb(obj)
    # With many random samples the AABB approaches ±1
    assert lo.min() > -1.1
    assert hi.max() < 1.1
    assert lo.min() < -0.9
    assert hi.max() > 0.9


# ---------------------------------------------------------------------------
# resolve_no_overlap
# ---------------------------------------------------------------------------


def test_resolve_no_overlap_single_object_unchanged() -> None:
    obj = box_object(100, center=(0.0, 0.0, 0.0), seed=20)
    result = resolve_no_overlap([obj])
    assert len(result) == 1
    np.testing.assert_array_equal(result[0].positions, obj.positions)


def test_resolve_no_overlap_two_non_overlapping_untouched() -> None:
    obj_a = box_object(100, center=(0.0, 0.0, 0.0), size=(1.0, 1.0, 1.0), seed=30)
    obj_b = box_object(100, center=(5.0, 0.0, 0.0), size=(1.0, 1.0, 1.0), seed=31)
    result = resolve_no_overlap([obj_a, obj_b])
    np.testing.assert_array_equal(result[0].positions, obj_a.positions)
    np.testing.assert_array_equal(result[1].positions, obj_b.positions)


def test_resolve_no_overlap_two_overlapping_boxes_separate() -> None:
    """Two overlapping 1x1 boxes must have non-overlapping XZ AABBs after resolution."""
    obj_a = box_object(500, center=(0.0, 0.0, 0.0), size=(1.0, 1.0, 1.0), seed=40)
    obj_b = box_object(500, center=(0.3, 0.0, 0.0), size=(1.0, 1.0, 1.0), seed=41)

    result = resolve_no_overlap([obj_a, obj_b])

    assert len(result) == 2

    pos_a_xz = result[0].positions[:, [0, 2]]
    pos_b_xz = result[1].positions[:, [0, 2]]
    lo_a, hi_a = pos_a_xz.min(0), pos_a_xz.max(0)
    lo_b, hi_b = pos_b_xz.min(0), pos_b_xz.max(0)

    # No overlap in X or Z
    x_overlap = hi_a[0] > lo_b[0] and hi_b[0] > lo_a[0]
    z_overlap = hi_a[1] > lo_b[1] and hi_b[1] > lo_a[1]
    assert not (x_overlap and z_overlap), (
        f"Objects still overlap: A=[{lo_a},{hi_a}] B=[{lo_b},{hi_b}]"
    )


def test_resolve_no_overlap_preserves_non_position_fields() -> None:
    """resolve_no_overlap must not alter quats/log_scales/opacity/colors_dc."""
    obj_a = box_object(100, center=(0.0, 0.0, 0.0), seed=50)
    obj_b = box_object(100, center=(0.1, 0.0, 0.0), seed=51)
    result = resolve_no_overlap([obj_a, obj_b])
    # First object is always anchored (no change)
    np.testing.assert_array_equal(result[0].positions, obj_a.positions)
    # Second object: other fields unchanged, positions may differ in X
    np.testing.assert_array_equal(result[1].quats, obj_b.quats)
    np.testing.assert_array_equal(result[1].log_scales, obj_b.log_scales)
    np.testing.assert_array_equal(result[1].opacity, obj_b.opacity)
    np.testing.assert_array_equal(result[1].colors_dc, obj_b.colors_dc)


def test_resolve_no_overlap_with_padding() -> None:
    """With padding=0.5, objects that touch must be pushed further apart."""
    obj_a = box_object(500, center=(0.0, 0.0, 0.0), size=(1.0, 1.0, 1.0), seed=60)
    obj_b = box_object(500, center=(1.0, 0.0, 0.0), size=(1.0, 1.0, 1.0), seed=61)

    result = resolve_no_overlap([obj_a, obj_b], padding=0.5)

    pos_a_xz = result[0].positions[:, [0, 2]]
    pos_b_xz = result[1].positions[:, [0, 2]]
    hi_a = pos_a_xz.max(0)
    lo_b = pos_b_xz.min(0)

    # With padding=0.5, hi_a[0] + 0.5 <= lo_b[0] (or no X overlap at all)
    assert hi_a[0] + 0.5 <= lo_b[0] + 1e-4, (
        f"Objects with padding still overlap: hi_a_x={hi_a[0]:.3f} lo_b_x={lo_b[0]:.3f}"
    )
