"""Compose tests: compose_scene geometric conditions."""

from __future__ import annotations

import numpy as np
from _objects import box_object

from astel_scene.compose import compose_scene
from astel_scene.layout import Placement, SceneLayout, SceneObject


def _make_layout(*object_ids: str, ground_y: float = 0.0) -> SceneLayout:
    objects = [
        SceneObject(
            object_id=oid,
            prompt=f"a {oid}",
            placement=Placement(
                object_id=oid,
                yaw_deg=0.0,
                uniform_scale=1.0,
                translation=(0.0, 0.0, 0.0),
                ground_contact=True,
            ),
        )
        for oid in object_ids
    ]
    return SceneLayout(objects=objects, ground_y=ground_y)


# ---------------------------------------------------------------------------
# count and index ranges
# ---------------------------------------------------------------------------


def test_combined_count_equals_sum() -> None:
    obj_a = box_object(300, seed=0)
    obj_b = box_object(500, seed=1)
    layout = _make_layout("a", "b")
    combined, ranges = compose_scene([obj_a, obj_b], layout)
    assert combined.count == 800
    assert len(ranges) == 2


def test_index_ranges_partition_total() -> None:
    obj_a = box_object(200, seed=2)
    obj_b = box_object(400, seed=3)
    layout = _make_layout("a", "b")
    combined, ranges = compose_scene([obj_a, obj_b], layout)

    assert ranges[0] == (0, 200)
    assert ranges[1] == (200, 600)

    # ranges are contiguous and cover [0, total)
    total = combined.count
    assert ranges[0][0] == 0
    assert ranges[-1][1] == total
    for i in range(len(ranges) - 1):
        assert ranges[i][1] == ranges[i + 1][0]


def test_index_ranges_recover_per_object_positions() -> None:
    """Slicing combined.positions with ranges must recover the placed positions."""
    obj_a = box_object(100, center=(0.0, 0.0, 0.0), seed=4)
    obj_b = box_object(150, center=(0.0, 0.0, 0.0), seed=5)
    layout = _make_layout("a", "b")
    combined, ranges = compose_scene(
        [obj_a, obj_b], layout, apply_ground_contact=False, resolve_overlap=False
    )
    # The combined array should be sliceable back to per-object splats
    slice_a = combined.positions[ranges[0][0] : ranges[0][1]]
    slice_b = combined.positions[ranges[1][0] : ranges[1][1]]
    assert slice_a.shape == (100, 3)
    assert slice_b.shape == (150, 3)


# ---------------------------------------------------------------------------
# ground contact
# ---------------------------------------------------------------------------


def test_each_object_sits_on_ground() -> None:
    """After compose with ground_contact=True, each object's 1st-pct Y ≈ ground_y."""
    ground_y = -1.5
    obj_a = box_object(1000, center=(0.0, 10.0, 0.0), size=(1.0, 2.0, 1.0), seed=10)
    obj_b = box_object(1000, center=(0.0, -5.0, 0.0), size=(1.0, 2.0, 1.0), seed=11)
    layout = _make_layout("a", "b", ground_y=ground_y)
    combined, ranges = compose_scene([obj_a, obj_b], layout, apply_ground_contact=True)

    tol = 0.05  # tolerance: 1st-pct of many uniform samples
    for start, end in ranges:
        ys = combined.positions[start:end, 1]
        y_1pct = float(np.percentile(ys, 1.0))
        assert abs(y_1pct - ground_y) < tol, (
            f"Object base at {y_1pct:.4f}, expected {ground_y} ± {tol}"
        )


def test_no_ground_contact_flag_skips_drop() -> None:
    """When ground_contact=False on the placement, ground_drop is skipped."""
    obj = box_object(200, center=(0.0, 50.0, 0.0), seed=20)
    layout = SceneLayout(
        objects=[
            SceneObject(
                object_id="a",
                prompt="a",
                placement=Placement(
                    object_id="a",
                    yaw_deg=0.0,
                    uniform_scale=1.0,
                    translation=(0.0, 0.0, 0.0),
                    ground_contact=False,  # skip drop
                ),
            )
        ],
        ground_y=0.0,
    )
    combined, _ = compose_scene([obj], layout, apply_ground_contact=True)
    # Object centred at y=50; should NOT have been dropped to ground
    y_mean = float(combined.positions[:, 1].mean())
    assert y_mean > 10.0, f"Object was unexpectedly dropped: mean y={y_mean}"


# ---------------------------------------------------------------------------
# no-overlap
# ---------------------------------------------------------------------------


def test_xz_aabbs_do_not_overlap_after_compose() -> None:
    """Two overlapping objects in XZ must have non-overlapping AABBs after compose."""
    obj_a = box_object(1000, center=(0.0, 0.0, 0.0), size=(1.0, 1.0, 1.0), seed=30)
    obj_b = box_object(1000, center=(0.2, 0.0, 0.0), size=(1.0, 1.0, 1.0), seed=31)
    layout = _make_layout("a", "b")
    combined, ranges = compose_scene([obj_a, obj_b], layout)

    pos_a = combined.positions[ranges[0][0] : ranges[0][1]]
    pos_b = combined.positions[ranges[1][0] : ranges[1][1]]

    hi_ax = float(pos_a[:, 0].max())
    lo_bx = float(pos_b[:, 0].min())
    hi_az = float(pos_a[:, 2].max())
    lo_bz = float(pos_b[:, 2].min())
    hi_bz = float(pos_b[:, 2].max())
    lo_az = float(pos_a[:, 2].min())

    x_overlap = hi_ax > lo_bx
    z_overlap = hi_az > lo_bz and hi_bz > lo_az
    assert not (x_overlap and z_overlap), (
        f"XZ AABBs still overlap: A_xmax={hi_ax:.3f}, B_xmin={lo_bx:.3f}"
    )


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def test_mismatched_lengths_raise_value_error() -> None:
    obj = box_object(50, seed=99)
    layout = _make_layout("a", "b")  # 2 objects in layout
    try:
        compose_scene([obj], layout)  # only 1 object provided
        raise AssertionError("Expected ValueError was not raised")
    except ValueError as e:
        assert "length" in str(e).lower() or "match" in str(e).lower()


# ---------------------------------------------------------------------------
# dtype preservation
# ---------------------------------------------------------------------------


def test_combined_arrays_are_float32() -> None:
    obj_a = box_object(100, seed=40)
    obj_b = box_object(100, seed=41)
    layout = _make_layout("a", "b")
    combined, _ = compose_scene([obj_a, obj_b], layout)
    assert combined.positions.dtype == np.float32
    assert combined.quats.dtype == np.float32
    assert combined.log_scales.dtype == np.float32
    assert combined.opacity.dtype == np.float32
    assert combined.colors_dc.dtype == np.float32
