"""CPU-only tests for :mod:`astel_gpu.metrics`.

No gsplat import here -- runs on any machine via plain ``uv run pytest``.
"""

from __future__ import annotations

import torch

from astel_gpu.metrics import chamfer_distance, meters_to_millimeters


def test_chamfer_distance_identical_sets_is_zero() -> None:
    a = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    result = chamfer_distance(a, a.clone())

    assert result["a_to_b"] == 0.0
    assert result["b_to_a"] == 0.0
    assert result["symmetric"] == 0.0


def test_chamfer_distance_known_offset() -> None:
    # a has a single point at the origin; b has a single point at (3, 4, 0),
    # i.e. distance 5 (3-4-5 triangle). Both directions are exactly 5.
    a = torch.tensor([[0.0, 0.0, 0.0]])
    b = torch.tensor([[3.0, 4.0, 0.0]])

    result = chamfer_distance(a, b)

    assert result["a_to_b"] == 5.0
    assert result["b_to_a"] == 5.0
    assert result["symmetric"] == 5.0


def test_chamfer_distance_asymmetric_nearest_neighbours() -> None:
    # a = {(0,0,0), (10,0,0)}, b = {(0,0,0)}.
    # a->b: nearest neighbour of each a point in b is (0,0,0) -> distances
    # 0 and 10, mean = 5.
    # b->a: nearest neighbour of (0,0,0) in a is (0,0,0) -> distance 0.
    a = torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    b = torch.tensor([[0.0, 0.0, 0.0]])

    result = chamfer_distance(a, b)

    assert result["a_to_b"] == 5.0
    assert result["b_to_a"] == 0.0
    assert result["symmetric"] == 2.5


def test_meters_to_millimeters() -> None:
    distances = {"a_to_b": 0.001, "b_to_a": 0.002, "symmetric": 0.0015}
    mm = meters_to_millimeters(distances)

    assert mm["a_to_b"] == 1.0
    assert mm["b_to_a"] == 2.0
    assert mm["symmetric"] == 1.5


def test_chamfer_distance_rejects_wrong_shape() -> None:
    import pytest

    a = torch.zeros(5, 2)
    b = torch.zeros(5, 3)

    with pytest.raises(ValueError, match="shape"):
        chamfer_distance(a, b)
