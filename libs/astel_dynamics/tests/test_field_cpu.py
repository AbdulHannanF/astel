"""DeformationField construction and apply() correctness tests."""

from __future__ import annotations

import numpy as np
import pytest
from _motion import static_cloud

from astel_dynamics.field import DeformationField


def _identity_field(n: int, k: int, f: int) -> DeformationField:
    """A field where every node transform is identity (R=I, t=0)."""
    rng = np.random.default_rng(7)
    node_positions = rng.random((k, 3)).astype(np.float32)

    # Uniform weights
    weights = np.ones((n, k), dtype=np.float32) / k

    # Identity transforms: R=I, t=0 → [I | 0] is (3, 4)
    node_transforms = np.zeros((f, k, 3, 4), dtype=np.float32)
    for fi in range(f):
        for ki in range(k):
            node_transforms[fi, ki, :3, :3] = np.eye(3, dtype=np.float32)

    return DeformationField(
        node_positions=node_positions,
        weights=weights,
        node_transforms=node_transforms,
    )


def test_identity_transform_returns_base() -> None:
    """apply() with identity transforms must return base_positions exactly."""
    n, k, f = 100, 4, 8
    base = static_cloud(n, seed=1)
    field = _identity_field(n, k, f)

    for frame in range(f):
        out = field.apply(base, frame=frame)
        np.testing.assert_allclose(out, base, atol=1e-5)


def test_single_node_known_rigid_transform() -> None:
    """K=1 with a known rigid rotation+translation must match analytic result."""
    n = 50
    base = static_cloud(n, seed=2)

    # 90-degree rotation around Z axis
    angle = np.pi / 2
    R = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )
    t = np.array([0.5, -0.3, 0.1], dtype=np.float32)

    # Single node, weight=1 for all gaussians
    node_positions = np.mean(base, axis=0, keepdims=True)  # (1, 3)
    weights = np.ones((n, 1), dtype=np.float32)

    # Transform = [R | t] shape (3, 4)
    tf = np.concatenate([R, t[:, np.newaxis]], axis=1)  # (3, 4)
    node_transforms = tf[np.newaxis, np.newaxis, :, :]  # (1, 1, 3, 4)

    field = DeformationField(
        node_positions=node_positions,
        weights=weights,
        node_transforms=node_transforms,
    )

    deformed = field.apply(base, frame=0)  # (N, 3)

    # Analytic: R @ base[n] + t for each n
    expected = (base @ R.T) + t  # (N, 3)

    np.testing.assert_allclose(deformed, expected, atol=1e-4)


def test_properties() -> None:
    field = _identity_field(n=80, k=5, f=12)
    assert field.n_gaussians == 80
    assert field.n_nodes == 5
    assert field.n_frames == 12


def test_wrong_base_shape_raises() -> None:
    field = _identity_field(n=10, k=2, f=3)
    bad_base = np.zeros((5, 3), dtype=np.float32)  # wrong N
    with pytest.raises(ValueError, match="base_positions"):
        field.apply(bad_base, frame=0)


def test_out_of_range_frame_raises() -> None:
    field = _identity_field(n=10, k=2, f=3)
    base = static_cloud(10, seed=3)
    with pytest.raises(ValueError, match="frame"):
        field.apply(base, frame=3)  # valid range is [0, 3)


def test_negative_frame_raises() -> None:
    field = _identity_field(n=10, k=2, f=3)
    base = static_cloud(10, seed=4)
    with pytest.raises(ValueError, match="frame"):
        field.apply(base, frame=-1)


def test_wrong_node_positions_shape_raises() -> None:
    with pytest.raises(ValueError, match="node_positions"):
        DeformationField(
            node_positions=np.zeros((3, 4), dtype=np.float32),  # bad: (K,4) not (K,3)
            weights=np.ones((10, 3), dtype=np.float32) / 3,
            node_transforms=np.zeros((2, 3, 3, 4), dtype=np.float32),
        )


def test_wrong_weights_shape_raises() -> None:
    with pytest.raises(ValueError, match="weights"):
        DeformationField(
            node_positions=np.zeros((4, 3), dtype=np.float32),
            weights=np.ones((10, 3), dtype=np.float32) / 3,  # bad: K mismatch
            node_transforms=np.zeros((2, 4, 3, 4), dtype=np.float32),
        )


def test_output_dtype_is_float32() -> None:
    n, k, f = 30, 2, 4
    base = static_cloud(n, seed=5)
    field = _identity_field(n, k, f)
    out = field.apply(base, frame=0)
    assert out.dtype == np.float32
