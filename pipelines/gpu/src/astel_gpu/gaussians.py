"""Random / target Gaussian-splat clouds for the smoke-test pipeline.

Parameters here are stored in "training" form (raw tensors that
``gsplat.rasterization`` consumes directly): linear scales (not log), raw SH
DC color (not yet converted to the INRIA ``f_dc`` convention), and sigmoid
opacity (not logit). Conversion to the INRIA PLY convention happens at export
time in :mod:`astel_gpu.export`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class GaussianParams:
    """Trainable (or fixed) Gaussian-splat parameters, training-space units.

    - ``means``: ``(N, 3)`` world-space xyz.
    - ``scales``: ``(N, 3)`` *linear* world-space sigma (gsplat applies
      ``exp`` internally if you pass log-scales; here we pass linear scales
      directly and store log-scales separately for export).
    - ``quats``: ``(N, 4)`` unnormalized quaternion, (w, x, y, z); gsplat
      normalizes internally.
    - ``opacities``: ``(N,)`` in ``[0, 1]`` (gsplat applies sigmoid if you pass
      logits — here we pass already-activated opacities directly).
    - ``colors``: ``(N, 3)`` linear RGB in roughly ``[0, 1]`` (SH band-0 only).
    """

    means: torch.Tensor
    scales: torch.Tensor
    quats: torch.Tensor
    opacities: torch.Tensor
    colors: torch.Tensor

    @property
    def count(self) -> int:
        return int(self.means.shape[0])


def build_target_cloud(
    n: int, seed: int, device: torch.device
) -> GaussianParams:
    """Build a known target cloud: a torus-knot ribbon of gaussians.

    Mirrors the shape used in ``pipelines/stub/make_sample_splat.py`` (a
    (p=2, q=3) torus knot) at reduced count, as a recognizable, non-trivial
    target for the render-then-refit smoke test.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)

    p, q = 2, 3
    tube_radius = 0.18
    knot_scale = 1.0

    t = torch.rand(n, generator=gen) * (2.0 * torch.pi)
    cos_qt = torch.cos(q * t)
    centre = knot_scale * torch.stack(
        [
            (2.0 + cos_qt) * torch.cos(p * t),
            (2.0 + cos_qt) * torch.sin(p * t),
            torch.sin(q * t),
        ],
        dim=1,
    )

    angle = torch.rand(n, generator=gen) * (2.0 * torch.pi)
    radial = torch.sqrt(torch.rand(n, generator=gen)) * tube_radius
    offset = torch.stack(
        [
            torch.cos(angle) * radial,
            torch.sin(angle) * radial,
            (torch.rand(n, generator=gen) - 0.5) * tube_radius,
        ],
        dim=1,
    )
    means = (centre + offset).to(device=device, dtype=torch.float32)

    # Small, fairly isotropic splats sized to the tube radius.
    base = 0.04 * tube_radius / 0.18
    scales = (
        base * (0.6 + 0.8 * torch.rand(n, 3, generator=gen))
    ).to(device=device, dtype=torch.float32)

    quats = torch.zeros(n, 4, dtype=torch.float32, device=device)
    quats[:, 0] = 1.0  # identity orientation; gsplat normalizes

    opacities = torch.full((n,), 0.95, dtype=torch.float32, device=device)

    # Brass -> teal gradient along the knot parameter, matching the stub.
    brass = torch.tensor([0.82, 0.55, 0.16])
    teal = torch.tensor([0.16, 0.62, 0.61])
    grad = (0.5 + 0.5 * torch.sin(q * t)).unsqueeze(1)
    rgb = brass[None, :] * (1.0 - grad) + teal[None, :] * grad
    colors = rgb.to(device=device, dtype=torch.float32).clamp(0.0, 1.0)

    return GaussianParams(
        means=means,
        scales=scales,
        quats=quats,
        opacities=opacities,
        colors=colors,
    )


def build_random_init_cloud(
    n: int, seed: int, device: torch.device, spread: float = 1.5
) -> GaussianParams:
    """Build a fresh random gaussian cloud as the optimization start point."""
    gen = torch.Generator(device="cpu").manual_seed(seed)

    means = (
        (torch.rand(n, 3, generator=gen) - 0.5) * 2.0 * spread
    ).to(device=device, dtype=torch.float32)
    scales = (
        0.05 * torch.ones(n, 3)
        * (0.5 + torch.rand(n, 3, generator=gen))
    ).to(device=device, dtype=torch.float32)
    quats = torch.zeros(n, 4, dtype=torch.float32, device=device)
    quats[:, 0] = 1.0
    quats += 0.01 * torch.randn(n, 4, generator=gen).to(device=device)
    opacities = torch.full((n,), 0.5, dtype=torch.float32, device=device)
    colors = torch.rand(n, 3, generator=gen).to(device=device, dtype=torch.float32)

    return GaussianParams(
        means=means,
        scales=scales,
        quats=quats,
        opacities=opacities,
        colors=colors,
    )
