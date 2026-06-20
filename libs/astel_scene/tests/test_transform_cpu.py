"""Transform tests: apply_placement with yaw, scale, translate."""

from __future__ import annotations

import math

import numpy as np
import pytest
from _objects import box_object

from astel_scene.layout import Placement
from astel_scene.transform import apply_placement, quat_from_yaw, quat_multiply

# ---------------------------------------------------------------------------
# quat_from_yaw basics
# ---------------------------------------------------------------------------


def test_quat_from_yaw_zero_is_identity() -> None:
    q = quat_from_yaw(0.0)
    np.testing.assert_allclose(q, [1.0, 0.0, 0.0, 0.0], atol=1e-6)


def test_quat_from_yaw_180_is_half_turn() -> None:
    q = quat_from_yaw(180.0)
    # cos(π/2)=0, sin(π/2)=1 → (0, 0, 1, 0)
    np.testing.assert_allclose(q, [0.0, 0.0, 1.0, 0.0], atol=1e-6)


# ---------------------------------------------------------------------------
# identity placement: zero yaw, scale=1, zero translation
# ---------------------------------------------------------------------------


def test_identity_placement_does_not_move_positions() -> None:
    obj = box_object(200, center=(1.0, 2.0, 3.0), seed=1)
    pl = Placement(
        object_id="a",
        yaw_deg=0.0,
        uniform_scale=1.0,
        translation=(0.0, 0.0, 0.0),
    )
    out = apply_placement(obj, pl)
    np.testing.assert_allclose(out.positions, obj.positions, atol=1e-5)


def test_identity_placement_leaves_quats_unchanged() -> None:
    obj = box_object(100, seed=2)
    pl = Placement(
        object_id="a",
        yaw_deg=0.0,
        uniform_scale=1.0,
        translation=(0.0, 0.0, 0.0),
    )
    out = apply_placement(obj, pl)
    np.testing.assert_allclose(out.quats, obj.quats, atol=1e-5)


# ---------------------------------------------------------------------------
# pure translation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("t", [(1.0, 2.0, 3.0), (-5.0, 0.0, 0.5)])
def test_translation_shifts_positions_exactly(
    t: tuple[float, float, float],
) -> None:
    obj = box_object(150, center=(0.0, 0.0, 0.0), seed=3)
    pl = Placement(
        object_id="a",
        yaw_deg=0.0,
        uniform_scale=1.0,
        translation=t,
    )
    out = apply_placement(obj, pl)
    expected = obj.positions + np.array(t, dtype=np.float32)
    np.testing.assert_allclose(out.positions, expected, atol=1e-5)
    # quats unchanged when yaw=0
    np.testing.assert_allclose(out.quats, obj.quats, atol=1e-5)


# ---------------------------------------------------------------------------
# yaw=90 deg: +X → −Z
# ---------------------------------------------------------------------------


def test_yaw_90_rotates_x_axis_to_neg_z() -> None:
    """A point on the +X axis should land on the −Z axis after 90° yaw.

    Right-hand rule about +Y: +X rotates towards −Z.
    We place a single splat at (1, 0, 0) and verify it lands near (0, 0, -1).
    """
    obj = box_object(1, center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0), seed=42)
    # Override positions to exactly (1, 0, 0)
    import dataclasses

    obj = dataclasses.replace(
        obj,
        positions=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
    )
    pl = Placement(
        object_id="a",
        yaw_deg=90.0,
        uniform_scale=1.0,
        translation=(0.0, 0.0, 0.0),
    )
    out = apply_placement(obj, pl)
    # cos(90°)=0, sin(90°)=1 → [c*x + s*z, y, -s*x + c*z] = [0,0,-1]
    np.testing.assert_allclose(out.positions[0], [0.0, 0.0, -1.0], atol=1e-4)


def test_yaw_90_updates_identity_quat_correctly() -> None:
    """After 90° yaw, the identity quat should become the yaw quat."""
    obj = box_object(10, seed=7)
    pl = Placement(
        object_id="a",
        yaw_deg=90.0,
        uniform_scale=1.0,
        translation=(0.0, 0.0, 0.0),
    )
    out = apply_placement(obj, pl)
    q_yaw = quat_from_yaw(90.0)
    # identity input quats: each output quat == q_yaw ⊗ (1,0,0,0) == q_yaw
    np.testing.assert_allclose(out.quats, np.tile(q_yaw, (10, 1)), atol=1e-5)


# ---------------------------------------------------------------------------
# uniform scale
# ---------------------------------------------------------------------------


def test_uniform_scale_doubles_position_extent() -> None:
    """Scale=2 should double the span of positions about the origin."""
    obj = box_object(300, center=(0.0, 0.0, 0.0), size=(1.0, 1.0, 1.0), seed=5)
    pl = Placement(
        object_id="a",
        yaw_deg=0.0,
        uniform_scale=2.0,
        translation=(0.0, 0.0, 0.0),
    )
    out = apply_placement(obj, pl)
    np.testing.assert_allclose(out.positions, obj.positions * 2.0, atol=1e-5)


def test_uniform_scale_adds_log_scale() -> None:
    """Scale=2 adds log(2) to every element of log_scales."""
    obj = box_object(50, seed=6)
    pl = Placement(
        object_id="a",
        yaw_deg=0.0,
        uniform_scale=2.0,
        translation=(0.0, 0.0, 0.0),
    )
    out = apply_placement(obj, pl)
    expected = obj.log_scales + math.log(2.0)
    np.testing.assert_allclose(out.log_scales, expected, atol=1e-5)


def test_colors_and_opacity_unchanged_by_placement() -> None:
    """apply_placement must not change colors_dc or opacity."""
    obj = box_object(80, seed=8)
    pl = Placement(
        object_id="a",
        yaw_deg=45.0,
        uniform_scale=3.0,
        translation=(1.0, -2.0, 0.5),
    )
    out = apply_placement(obj, pl)
    np.testing.assert_array_equal(out.colors_dc, obj.colors_dc)
    np.testing.assert_array_equal(out.opacity, obj.opacity)


# ---------------------------------------------------------------------------
# quat_multiply
# ---------------------------------------------------------------------------


def test_quat_multiply_identity() -> None:
    """q ⊗ identity == q."""
    q = np.array([0.707, 0.707, 0.0, 0.0], dtype=np.float32)
    ident = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    result = quat_multiply(q, ident)
    np.testing.assert_allclose(result, q, atol=1e-5)


def test_quat_multiply_180_twice_is_identity() -> None:
    """Two 180° yaw rotations == identity (full spin)."""
    q180 = quat_from_yaw(180.0)
    result = quat_multiply(q180, q180)
    # Should be ±identity
    np.testing.assert_allclose(np.abs(result), [1.0, 0.0, 0.0, 0.0], atol=1e-5)
