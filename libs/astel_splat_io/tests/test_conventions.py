"""Tests for coordinate-convention transforms."""

from __future__ import annotations

import numpy as np
import pytest

from astel_splat_io.conventions import (
    gltf_positions,
    gltf_quats,
    unity_positions,
    unity_quats,
    unreal_positions,
    unreal_quats,
    unreal_scales,
)


def _make_pos(n: int = 8, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-2.0, 2.0, (n, 3)).astype(np.float32)


def _make_quats(n: int = 8, seed: int = 13) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(n, 4)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return q


# ---------------------------------------------------------------------------
# glTF: identity position, quaternion reorder only
# ---------------------------------------------------------------------------


def test_gltf_positions_unchanged() -> None:
    pos = _make_pos()
    np.testing.assert_array_equal(gltf_positions(pos), pos)


def test_gltf_quats_reorder() -> None:
    """(w,x,y,z) → (x,y,z,w) and norms preserved."""
    q = _make_quats()
    out = gltf_quats(q)
    # x,y,z,w of output = x,y,z,w of input
    np.testing.assert_array_equal(out[:, 0], q[:, 1])  # x
    np.testing.assert_array_equal(out[:, 1], q[:, 2])  # y
    np.testing.assert_array_equal(out[:, 2], q[:, 3])  # z
    np.testing.assert_array_equal(out[:, 3], q[:, 0])  # w


# ---------------------------------------------------------------------------
# Unity: negate X, quat handedness flip
# ---------------------------------------------------------------------------


def test_unity_positions_negate_x() -> None:
    pos = _make_pos()
    out = unity_positions(pos)
    np.testing.assert_array_equal(out[:, 0], -pos[:, 0])
    np.testing.assert_array_equal(out[:, 1], pos[:, 1])
    np.testing.assert_array_equal(out[:, 2], pos[:, 2])


def test_unity_positions_double_negate_is_identity() -> None:
    pos = _make_pos()
    np.testing.assert_array_equal(unity_positions(unity_positions(pos)), pos)


def test_unity_quats_negate_x_and_w() -> None:
    q = _make_quats()
    out = unity_quats(q)  # returns (x,y,z,w) with -x, -w
    np.testing.assert_array_equal(out[:, 0], -q[:, 1])  # -x
    np.testing.assert_array_equal(out[:, 1], q[:, 2])   # y
    np.testing.assert_array_equal(out[:, 2], q[:, 3])   # z
    np.testing.assert_array_equal(out[:, 3], -q[:, 0])  # -w


def test_unity_quats_unit_norm_preserved() -> None:
    q = _make_quats()
    out = unity_quats(q)
    norms = np.linalg.norm(out, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Unreal: (x,y,z) → (−z*100, x*100, y*100)
# ---------------------------------------------------------------------------


def test_unreal_positions_axis_remap() -> None:
    pos = _make_pos()
    out = unreal_positions(pos)
    np.testing.assert_allclose(out[:, 0], -pos[:, 2] * 100, atol=1e-5)
    np.testing.assert_allclose(out[:, 1], pos[:, 0] * 100, atol=1e-5)
    np.testing.assert_allclose(out[:, 2], pos[:, 1] * 100, atol=1e-5)


def test_unreal_positions_scale_cm() -> None:
    pos = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    out = unreal_positions(pos)
    # (1,0,0) → (0*100, 1*100, 0*100) = (0, 100, 0) in UE cm
    np.testing.assert_allclose(out[0], [0.0, 100.0, 0.0], atol=1e-5)


def test_unreal_quats_unit_norm_preserved() -> None:
    q = _make_quats()
    out = unreal_quats(q)
    norms = np.linalg.norm(out, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Scale: centimetre conversion
# ---------------------------------------------------------------------------


def test_unreal_scales_multiply_100() -> None:
    scales = np.array([[0.01, 0.02, 0.05]], dtype=np.float32)
    out = unreal_scales(scales)
    np.testing.assert_allclose(out, scales * 100, atol=1e-5)


# ---------------------------------------------------------------------------
# Rotational correctness (the property the norm/reorder tests cannot catch)
#
# A pure coordinate change by an orthogonal matrix M maps a source-frame
# rotation R to R' = M R Mᵀ in the target frame. The engine quaternion the
# convention helper emits MUST reproduce R' (up to the q ~ -q double cover,
# which the rotation-matrix comparison absorbs). This is the real test that a
# handedness flip is correct, not merely norm-preserving.
# ---------------------------------------------------------------------------


def _rotmat_wxyz(q: np.ndarray) -> np.ndarray:
    """Rotation matrix from a (w,x,y,z) Hamilton quaternion (matches gltf.py)."""
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


# M (rotation/reflection part only) for each target, applied to source-frame
# positions: pos_target = M @ pos_source.
_TARGET_M = {
    # Unity: negate X (reflection through the YZ plane).
    "unity": np.diag([-1.0, 1.0, 1.0]),
    # Unreal: axis remap (x,y,z) -> (-z, x, y) (the cm scale is irrelevant to R).
    "unreal": np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
}
_TARGET_FN = {"unity": unity_quats, "unreal": unreal_quats}


@pytest.mark.parametrize("target", ["unity", "unreal"])
def test_engine_quat_reproduces_mirrored_rotation(target: str) -> None:
    """rotmat(engine_quat) == M @ rotmat(source_quat) @ Mᵀ for each engine."""
    m = _TARGET_M[target]
    quats = _make_quats(n=16, seed=99)
    out_xyzw = _TARGET_FN[target](quats)
    for src, dst in zip(quats, out_xyzw, strict=True):
        r_src = _rotmat_wxyz(src.astype(np.float64))
        r_expected = m @ r_src @ m.T
        r_actual = _rotmat_wxyz(_xyzw_to_wxyz(dst))
        np.testing.assert_allclose(r_actual, r_expected, atol=1e-5)
