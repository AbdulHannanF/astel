"""CPU tests for splat cleanup (pure torch + scipy KD-tree; no gsplat/CUDA)."""

from __future__ import annotations

from dataclasses import replace

import torch

from astel_gpu.gaussians import GaussianParams
from astel_gpu.splat_clean import (
    CleanConfig,
    clean_gaussians,
    elongation_mask,
    opacity_mask,
    oversize_mask,
    statistical_outlier_mask,
)


def _cloud(
    means: torch.Tensor,
    *,
    scales: torch.Tensor | None = None,
    opacities: torch.Tensor | None = None,
) -> GaussianParams:
    n = means.shape[0]
    return GaussianParams(
        means=means,
        scales=torch.full((n, 3), 0.02) if scales is None else scales,
        quats=torch.tile(torch.tensor([1.0, 0.0, 0.0, 0.0]), (n, 1)),
        opacities=torch.full((n,), 0.9) if opacities is None else opacities,
        colors=torch.rand(n, 3),
    )


def _good_cluster(n: int = 200, seed: int = 0) -> GaussianParams:
    """A dense, uniform, opaque ball near the origin — stands in for a real object.

    Uniform volume density (radius ``∝ U**(1/3)``) so SOR has no sparse tail to
    trim — a real object surface is dense and uniform, unlike a Gaussian blob.
    """
    gen = torch.Generator().manual_seed(seed)
    directions = torch.randn(n, 3, generator=gen)
    directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    radii = 0.3 * torch.rand(n, generator=gen).pow(1.0 / 3.0).unsqueeze(-1)
    return _cloud(directions * radii)


def test_opacity_mask_drops_faint_splats() -> None:
    opac = torch.tensor([0.9, 0.01, 0.5, 0.04])
    mask = opacity_mask(opac, 0.06)
    assert mask.tolist() == [True, False, True, False]


def test_elongation_mask_drops_needles_keeps_flat_surfels() -> None:
    scales = torch.tensor(
        [
            [0.02, 0.02, 0.02],  # sphere -> keep
            [0.0001, 0.02, 0.02],  # flat surfel disc -> keep (s_mid ~= s_max)
            [0.02, 0.02, 1.0],  # needle -> drop (s_max/s_mid = 50)
        ]
    )
    mask = elongation_mask(scales, max_elongation=12.0)
    assert mask.tolist() == [True, True, False]


def test_oversize_mask_drops_giant_blobs() -> None:
    scales = torch.cat(
        [torch.full((20, 3), 0.02), torch.tensor([[1.0, 1.0, 1.0]])], dim=0
    )
    mask = oversize_mask(scales, max_scale_factor=10.0)
    assert bool(mask[:20].all())  # all the median-sized splats survive
    assert not bool(mask[20])  # the 50x blob is dropped


def test_statistical_outlier_mask_drops_disconnected_floater() -> None:
    good = _good_cluster(n=120).means
    floater = torch.tensor([[8.0, 8.0, 8.0]])  # far from the cluster
    means = torch.cat([good, floater], dim=0)
    mask = statistical_outlier_mask(means, nb_neighbors=8, std_ratio=2.0)
    assert bool(mask[:120].all())  # cluster kept
    assert not bool(mask[-1])  # floater removed


def test_clean_gaussians_removes_junk_keeps_object() -> None:
    good = _good_cluster(n=300)
    # Inject one of each junk type.
    junk_means = torch.tensor(
        [[6.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.05, 0.0, 0.0]]
    )
    junk_scales = torch.tensor(
        [[0.02, 0.02, 0.02], [0.02, 0.02, 0.02], [0.02, 0.02, 2.0]]
    )
    junk_opac = torch.tensor([0.9, 0.01, 0.9])  # 2nd is faint
    junk = _cloud(junk_means, scales=junk_scales, opacities=junk_opac)

    cloud = GaussianParams(
        means=torch.cat([good.means, junk.means]),
        scales=torch.cat([good.scales, junk.scales]),
        quats=torch.cat([good.quats, junk.quats]),
        opacities=torch.cat([good.opacities, junk.opacities]),
        colors=torch.cat([good.colors, junk.colors]),
    )

    cleaned, stats = clean_gaussians(
        cloud, CleanConfig(sor_neighbors=8, sor_iters=2)
    )

    assert stats["input"] == 303
    # Each junk category is detected by at least one filter.
    assert stats["opacity_removed"] >= 1
    assert stats["elongation_removed"] >= 1
    assert stats["spatial_outliers_removed"] >= 1
    # The bulk of the real object survives.
    assert cleaned.count >= 290
    assert stats["kept"] == cleaned.count
    assert stats["removed_total"] == 303 - cleaned.count
    # All surviving splats are near the origin (no far floaters left).
    assert float(cleaned.means.norm(dim=-1).max()) < 1.0


def test_disabled_config_passes_through() -> None:
    cloud = _good_cluster(n=50)
    cleaned, stats = clean_gaussians(cloud, CleanConfig(enabled=False))
    assert cleaned.count == 50
    assert stats["removed_total"] == 0
    assert stats["enabled"] is False


def test_apply_skips_filter_that_would_empty_cloud() -> None:
    # opacity_min above every splat's opacity would remove all -> filter skipped.
    cloud = _good_cluster(n=40)  # all opacity 0.9
    cleaned, stats = clean_gaussians(
        cloud, CleanConfig(opacity_min=0.99, sor_iters=0)
    )
    assert stats["opacity_removed"] == -1  # skipped, not applied
    assert cleaned.count == 40  # nothing destroyed


def test_spatial_false_skips_sor() -> None:
    good = _good_cluster(n=120).means
    floater = torch.tensor([[8.0, 8.0, 8.0]])
    means = torch.cat([good, floater], dim=0)
    cloud = _cloud(means)
    cleaned, stats = clean_gaussians(cloud, CleanConfig(), spatial=False)
    assert stats["spatial_outliers_removed"] == 0
    assert cleaned.count == 121  # SOR skipped -> floater survives


def test_from_env_overrides() -> None:
    env = {
        "ASTEL_CLEAN": "1",
        "ASTEL_CLEAN_OPACITY_MIN": "0.2",
        "ASTEL_CLEAN_MAX_ELONGATION": "5",
        "ASTEL_CLEAN_SOR_STD_RATIO": "1.5",
        "ASTEL_CLEAN_SOR_ITERS": "3",
    }
    cfg = CleanConfig.from_env(env)
    assert cfg.enabled is True
    assert cfg.opacity_min == 0.2
    assert cfg.max_elongation == 5.0
    assert cfg.sor_std_ratio == 1.5
    assert cfg.sor_iters == 3


def test_from_env_disable() -> None:
    assert CleanConfig.from_env({"ASTEL_CLEAN": "0"}).enabled is False
    assert CleanConfig.from_env({"ASTEL_CLEAN": "off"}).enabled is False


def test_from_env_defaults_on_bad_values() -> None:
    cfg = CleanConfig.from_env({"ASTEL_CLEAN_OPACITY_MIN": "notanumber"})
    assert cfg.opacity_min == replace(CleanConfig()).opacity_min
