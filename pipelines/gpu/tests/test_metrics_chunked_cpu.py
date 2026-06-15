"""CPU tests: chunked Chamfer must match the exact cdist version (any chunk size)."""

from __future__ import annotations

import pytest
import torch

from astel_gpu.metrics import (
    chamfer_distance,
    chamfer_distance_chunked,
    nn_distances,
)


@pytest.mark.parametrize("chunk", [1, 3, 7, 64, 10_000])
def test_chunked_matches_exact(chunk: int) -> None:
    gen = torch.Generator().manual_seed(0)
    a = torch.rand(37, 3, generator=gen)
    b = torch.rand(53, 3, generator=gen)

    exact = chamfer_distance(a, b)
    chunked = chamfer_distance_chunked(a, b, chunk_size=chunk)
    for key in ("a_to_b", "b_to_a", "symmetric"):
        assert chunked[key] == pytest.approx(exact[key], rel=1e-5)


def test_chunked_asymmetric_sizes() -> None:
    gen = torch.Generator().manual_seed(1)
    a = torch.rand(200, 3, generator=gen)
    b = torch.rand(5, 3, generator=gen)
    exact = chamfer_distance(a, b)
    chunked = chamfer_distance_chunked(a, b, chunk_size=8)
    assert chunked["symmetric"] == pytest.approx(exact["symmetric"], rel=1e-5)


def test_chunked_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError):
        chamfer_distance_chunked(torch.rand(4, 2), torch.rand(4, 3))
    with pytest.raises(ValueError):
        chamfer_distance_chunked(torch.rand(0, 3), torch.rand(4, 3))


def test_nn_distances_matches_bruteforce() -> None:
    gen = torch.Generator().manual_seed(2)
    q = torch.rand(40, 3, generator=gen)
    r = torch.rand(25, 3, generator=gen)
    d = nn_distances(q, r, chunk_size=7)
    brute = torch.cdist(q, r).min(dim=1).values
    assert d.shape == (40,)
    assert torch.allclose(d, brute, atol=1e-5)
