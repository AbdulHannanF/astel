"""CPU tests for l2_triposplat helpers (no weights/CUDA/gsplat needed)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from astel_gpu.l2_triposplat import (
    TRIPOSPLAT_DEFAULT_TRANSFORM,
    gaussian_to_splat_cloud,
    splat_cloud_from_fields,
)


def test_splat_cloud_from_fields_clamps_opacity() -> None:
    n = 3
    xyz = np.zeros((n, 3), dtype=np.float32)
    f_dc = np.zeros((n, 3), dtype=np.float32)
    log_scales = np.zeros((n, 3), dtype=np.float32)
    quats = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (n, 1))
    # Exactly 1.0 and 0.0 are the fp16-saturation cases that produce inf
    # logits via upstream's _inverse_opacity_activation.
    opacity_activated = np.array([1.0, 0.0, 0.5], dtype=np.float32)

    cloud = splat_cloud_from_fields(
        xyz=xyz,
        f_dc=f_dc,
        opacity_activated=opacity_activated,
        log_scales=log_scales,
        quats=quats,
    )

    assert cloud.positions.shape == (n, 3)
    assert cloud.colors_dc.shape == (n, 3)
    assert cloud.opacity.shape == (n,)
    assert cloud.log_scales.shape == (n, 3)
    assert cloud.quats.shape == (n, 4)

    assert np.isfinite(cloud.opacity).all()
    # alpha=0.5 -> logit(0.5) == 0.0
    assert np.isclose(cloud.opacity[2], 0.0, atol=1e-5)


class _FakeGaussian:
    """Minimal stand-in for ``triposplat.Gaussian`` exercising the adapter."""

    def __init__(self) -> None:
        self.n = 4
        self._xyz = np.zeros((self.n, 3), dtype=np.float32)
        self._normals = np.zeros((self.n, 3), dtype=np.float32)
        self._f_dc = np.zeros((self.n, 3), dtype=np.float32)
        # Upstream's inf-prone logit field: includes inf entries (from
        # opacity == 1.0) that the adapter must NOT use.
        self._unsafe_opacity_logit = np.array(
            [np.inf, -np.inf, 0.0, 1.0], dtype=np.float32
        ).reshape(self.n, 1)
        self._scale_log = np.zeros((self.n, 3), dtype=np.float32)
        self._rotation = np.tile(
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (self.n, 1)
        )
        # Activated opacity in [0, 1], including the fp16-saturation 1.0 case.
        self._opacity_activated = torch.tensor(
            [1.0, 0.0, 0.5, 0.9], dtype=torch.float32
        )

    def _get_ply_data(self, transform: list[list[float]]) -> tuple[Any, ...]:
        return (
            self._xyz,
            self._normals,
            self._f_dc,
            self._unsafe_opacity_logit,
            self._scale_log,
            self._rotation,
        )

    @property
    def get_opacity(self) -> torch.Tensor:
        return self._opacity_activated


def test_gaussian_to_splat_cloud_ignores_unsafe_logit() -> None:
    fake = _FakeGaussian()

    cloud = gaussian_to_splat_cloud(fake, transform=TRIPOSPLAT_DEFAULT_TRANSFORM)

    assert cloud.positions.shape == (fake.n, 3)
    assert cloud.opacity.shape == (fake.n,)
    assert np.isfinite(cloud.opacity).all()
