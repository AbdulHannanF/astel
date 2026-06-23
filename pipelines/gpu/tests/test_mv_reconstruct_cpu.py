"""CPU tests for the pure multi-view reconstruction seams (no gsplat / rembg)."""

from __future__ import annotations

import torch

from astel_gpu.mv_reconstruct import (
    ortho_cameras,
    reconstruction_loss,
    sphere_init,
)


def _cam_centres(viewmats: torch.Tensor) -> torch.Tensor:
    r = viewmats[:, :3, :3]
    t = viewmats[:, :3, 3]
    return -torch.bmm(r.transpose(1, 2), t[:, :, None]).squeeze(-1)


def test_ortho_cameras_shapes_and_intrinsics() -> None:
    vm, ks = ortho_cameras((0, 90, 180, 270), 0.0, 768, frustum=1.1, distance=3.0)
    assert vm.shape == (4, 4, 4)
    assert ks.shape == (4, 3, 3)
    # ortho scale fx = resolution / frustum
    assert abs(float(ks[0, 0, 0]) - 768 / 1.1) < 1e-3
    assert abs(float(ks[0, 0, 2]) - 384.0) < 1e-3


def test_ortho_cameras_az0_is_minus_y() -> None:
    vm, _ = ortho_cameras((0, 90, 180), 0.0, 256, distance=3.0)
    centres = _cam_centres(vm)
    # azimuth 0 viewed from -Y; 90 from +X; 180 from +Y (the MV-Adapter convention).
    assert torch.allclose(centres[0], torch.tensor([0.0, -3.0, 0.0]), atol=1e-4)
    assert torch.allclose(centres[1], torch.tensor([3.0, 0.0, 0.0]), atol=1e-4)
    assert torch.allclose(centres[2], torch.tensor([0.0, 3.0, 0.0]), atol=1e-4)


def test_ortho_cameras_elevation_raises_in_z() -> None:
    vm, _ = ortho_cameras((0,), 30.0, 256, distance=2.0)
    centre = _cam_centres(vm)[0]
    assert float(centre[2]) > 0.5  # elevated above the equator (+Z up)


def test_sphere_init_within_radius() -> None:
    cloud = sphere_init(2000, radius=0.5, device="cpu")
    assert cloud.count == 2000
    assert float(cloud.means.norm(dim=-1).max()) <= 0.5 + 1e-5
    assert cloud.colors.shape == (2000, 3)
    assert cloud.quats.shape == (2000, 4)


def test_reconstruction_loss_zero_for_perfect_match() -> None:
    rgb = torch.rand(2, 8, 8, 3)
    alpha = torch.rand(2, 8, 8, 1)
    loss = reconstruction_loss(rgb, alpha, rgb.clone(), alpha.clone())
    assert float(loss) < 1e-6


def test_reconstruction_loss_positive_for_mismatch() -> None:
    rgb = torch.zeros(1, 8, 8, 3)
    target = torch.ones(1, 8, 8, 3)
    alpha = torch.zeros(1, 8, 8, 1)
    mask = torch.ones(1, 8, 8, 1)
    assert float(reconstruction_loss(rgb, alpha, target, mask)) > 0.0
