"""CPU tests for adaptive density control (pure torch; no gsplat/CUDA)."""

from __future__ import annotations

import pytest
import torch

from astel_gpu.densify import (
    DensifyConfig,
    DensityController,
    clone_mask,
    densify_and_prune,
    prune_mask,
    reset_opacity,
    split_mask,
)
from astel_gpu.gaussians import GaussianParams


def _cloud(
    *,
    scales: list[float],
    opacities: list[float],
) -> GaussianParams:
    n = len(scales)
    return GaussianParams(
        means=torch.arange(n * 3, dtype=torch.float32).reshape(n, 3),
        scales=torch.tensor([[s, s, s] for s in scales], dtype=torch.float32),
        quats=torch.tile(torch.tensor([1.0, 0.0, 0.0, 0.0]), (n, 1)),
        opacities=torch.tensor(opacities, dtype=torch.float32),
        colors=torch.rand(n, 3),
    )


_CFG = DensifyConfig()  # unit-extent defaults


def test_clone_split_masks_are_disjoint_and_scale_gated() -> None:
    cloud = _cloud(scales=[0.005, 0.1], opacities=[0.9, 0.9])
    grads = torch.tensor([1e-3, 1e-3])  # both above grad_threshold

    cl = clone_mask(grads, cloud.scales, _CFG)
    sp = split_mask(grads, cloud.scales, _CFG)
    assert cl.tolist() == [True, False]  # small -> clone
    assert sp.tolist() == [False, True]  # large -> split
    assert not bool((cl & sp).any())


def test_low_gradient_is_never_densified() -> None:
    cloud = _cloud(scales=[0.005, 0.1], opacities=[0.9, 0.9])
    grads = torch.tensor([0.0, 0.0])
    assert not bool(clone_mask(grads, cloud.scales, _CFG).any())
    assert not bool(split_mask(grads, cloud.scales, _CFG).any())


def test_prune_mask_flags_faint_and_oversized() -> None:
    cloud = _cloud(scales=[0.005, 0.6, 0.005], opacities=[0.9, 0.9, 0.01])
    pm = prune_mask(cloud.opacities, cloud.scales, _CFG)
    assert pm.tolist() == [False, True, True]  # big blob + faint splat


def test_densify_and_prune_full_step_counts() -> None:
    # idx0 small+high-grad -> clone; idx1 large+high-grad -> split(2);
    # idx2 untouched; idx3 faint -> pruned.
    cloud = _cloud(
        scales=[0.005, 0.1, 0.005, 0.005], opacities=[0.9, 0.9, 0.9, 0.01]
    )
    grads = torch.tensor([1e-3, 1e-3, 0.0, 0.0])
    gen = torch.Generator().manual_seed(0)

    out, stats = densify_and_prune(cloud, grads, _CFG, generator=gen)

    assert stats["cloned"] == 1
    assert stats["split"] == 1
    assert stats["split_children"] == 2
    assert stats["pruned"] == 1  # the faint splat
    # survivors(3: idx0,2,3) + clone(1) + children(2) = 6, minus 1 pruned = 5
    assert stats["output"] == 5
    assert out.count == 5


def test_split_children_are_shrunk() -> None:
    cloud = _cloud(scales=[0.1], opacities=[0.9])
    grads = torch.tensor([1e-3])
    gen = torch.Generator().manual_seed(0)
    out, stats = densify_and_prune(cloud, grads, _CFG, generator=gen)

    assert stats["split"] == 1
    # parent removed; both children carry scale = 0.1 / 1.6
    expected = 0.1 / _CFG.split_scale_factor
    assert torch.allclose(out.scales, torch.full_like(out.scales, expected))


def test_densify_is_deterministic_under_seed() -> None:
    cloud = _cloud(scales=[0.1], opacities=[0.9])
    grads = torch.tensor([1e-3])
    a, _ = densify_and_prune(
        cloud, grads, _CFG, generator=torch.Generator().manual_seed(7)
    )
    b, _ = densify_and_prune(
        cloud, grads, _CFG, generator=torch.Generator().manual_seed(7)
    )
    assert torch.allclose(a.means, b.means)


def test_max_gaussians_cap_blocks_growth_but_allows_prune() -> None:
    cloud = _cloud(scales=[0.005, 0.005], opacities=[0.9, 0.01])
    grads = torch.tensor([1e-3, 1e-3])
    capped = DensifyConfig(max_gaussians=2)  # already at cap
    out, stats = densify_and_prune(cloud, grads, capped)

    assert stats["at_cap"] is True
    assert stats["cloned"] == 0 and stats["split"] == 0
    assert stats["pruned"] == 1  # the faint one is still removed
    assert out.count == 1


def test_prune_never_empties_the_cloud() -> None:
    # All faint: a naive prune would delete everything; the guard keeps it.
    cloud = _cloud(scales=[0.005, 0.005], opacities=[0.001, 0.001])
    grads = torch.tensor([0.0, 0.0])
    out, _ = densify_and_prune(cloud, grads, _CFG)
    assert out.count == 2


def test_reset_opacity_clamps_down_only() -> None:
    cloud = _cloud(scales=[0.01, 0.01], opacities=[0.9, 0.02])
    out = reset_opacity(cloud, value=0.05)
    assert float(out.opacities[0]) == pytest.approx(0.05)  # high clamped down
    assert float(out.opacities[1]) == pytest.approx(0.02)  # already low, unchanged
    assert torch.allclose(out.means, cloud.means)  # geometry untouched


def test_controller_records_and_schedules() -> None:
    ctrl = DensityController(n=2, warmup=100, interval=100, stop=1000)
    ctrl.record(torch.tensor([[3.0, 4.0], [0.0, 0.0]]).reshape(2, 2))  # norms 5, 0
    ctrl.record(torch.tensor([[3.0, 4.0], [0.0, 0.0]]).reshape(2, 2))
    avg = ctrl.avg_grad()
    assert torch.allclose(avg, torch.tensor([5.0, 0.0]))

    assert ctrl.should_densify(100)
    assert not ctrl.should_densify(150)
    assert not ctrl.should_densify(50)  # before warmup
    assert ctrl.should_reset_opacity(300)


def test_controller_step_resets_stats_to_new_count() -> None:
    cloud = _cloud(scales=[0.1], opacities=[0.9])
    ctrl = DensityController(n=1, generator=torch.Generator().manual_seed(0))
    ctrl.record(torch.tensor([[1.0, 0.0, 0.0]]))  # high grad on the one splat
    new, stats = ctrl.step(cloud)

    assert new.count == stats["output"]
    # accumulators were resized to the new count and zeroed.
    assert ctrl.grad_accum.shape[0] == new.count
    assert float(ctrl.denom.sum()) == 0.0
