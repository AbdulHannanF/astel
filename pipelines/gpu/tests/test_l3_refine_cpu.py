"""CPU tests for the pure 2DGS surface-regularization seam (no gsplat/CUDA)."""

from __future__ import annotations

import math

import torch

from astel_gpu.l3_refine import surface_reg_loss


def _normals(vec: tuple[float, float, float], shape: tuple[int, ...]) -> torch.Tensor:
    """A (*shape, 3) field of a single (normalized) direction ``vec``."""
    v = torch.tensor(vec, dtype=torch.float32)
    v = v / v.norm()
    field: torch.Tensor = v.expand(*shape, 3).contiguous()
    return field


def test_aligned_normals_zero_normal_term() -> None:
    shape = (2, 4, 4)  # V, H, W
    n = _normals((0.0, 0.0, 1.0), shape)
    distort = torch.zeros(*shape, 1)
    # Perfectly aligned render/depth normals -> 1 - <n,n> = 0.
    loss = surface_reg_loss(n, n, distort, lambda_normal=0.05, lambda_dist=0.0)
    assert math.isclose(float(loss), 0.0, abs_tol=1e-6)


def test_opposite_normals_max_normal_term() -> None:
    shape = (1, 3, 3)
    n = _normals((0.0, 0.0, 1.0), shape)
    loss = surface_reg_loss(
        n, -n, torch.zeros(*shape, 1), lambda_normal=0.05, lambda_dist=0.0
    )
    # 1 - <n,-n> = 1 - (-1) = 2; weighted by lambda_normal.
    assert math.isclose(float(loss), 0.05 * 2.0, rel_tol=1e-5)


def test_distortion_term_weighted_and_additive() -> None:
    shape = (1, 2, 2)
    n = _normals((1.0, 0.0, 0.0), shape)
    distort = torch.full((*shape, 1), 0.3)
    # Aligned normals -> only the distortion term survives, scaled by lambda_dist.
    loss = surface_reg_loss(n, n, distort, lambda_normal=0.05, lambda_dist=2.0)
    assert math.isclose(float(loss), 2.0 * 0.3, rel_tol=1e-5)


def test_loss_is_differentiable() -> None:
    shape = (1, 2, 2)
    n = _normals((0.0, 1.0, 0.0), shape).requires_grad_(True)
    sn = _normals((0.0, 0.0, 1.0), shape)
    loss = surface_reg_loss(
        n, sn, torch.zeros(*shape, 1), lambda_normal=0.05, lambda_dist=0.0
    )
    loss.backward()  # type: ignore[no-untyped-call]
    assert n.grad is not None
    assert torch.isfinite(n.grad).all()
