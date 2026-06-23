"""CPU tests for the degenerate-asset critic (pure stats; no gsplat/CUDA)."""

from __future__ import annotations

import torch

from astel_gpu.gaussians import GaussianParams, build_target_cloud
from astel_gpu.geometry_qa import score_cloud


def _healthy_cloud(n: int = 4096) -> GaussianParams:
    return build_target_cloud(n, seed=0, device=torch.device("cpu"))


def _smoke_cloud(n: int = 4096) -> GaussianParams:
    """A mostly-transparent cloud — the 'faint smoke' failure."""
    g = _healthy_cloud(n)
    return GaussianParams(
        means=g.means,
        scales=g.scales,
        quats=g.quats,
        opacities=torch.full((n,), 0.02),
        colors=g.colors,
    )


def _exploded_cloud(n: int = 4096) -> GaussianParams:
    """A compact body plus a far floater halo — long radial tail."""
    g = _healthy_cloud(n)
    means = g.means.clone()
    # Fling the last 5% of splats far from the body.
    k = n // 20
    means[-k:] = means[-k:] + torch.tensor([50.0, 50.0, 50.0])
    return GaussianParams(
        means=means,
        scales=g.scales,
        quats=g.quats,
        opacities=g.opacities,
        colors=g.colors,
    )


def test_healthy_cloud_scores_high_and_accepts() -> None:
    score = score_cloud(
        _healthy_cloud(),
        clean_removed_fraction=0.02,
        selfconsistency_psnr_db=32.0,
    )
    assert score.accept
    assert score.overall > 0.8
    assert score.opacity > 0.9
    assert score.compactness > 0.8
    assert score.flags == []


def test_smoke_cloud_fails_on_opacity() -> None:
    score = score_cloud(_smoke_cloud(), selfconsistency_psnr_db=32.0)
    assert score.opacity < 0.3
    assert any("opacity" in f for f in score.flags)
    assert score.overall < score_cloud(_healthy_cloud()).overall


def test_exploded_cloud_fails_on_compactness() -> None:
    score = score_cloud(_exploded_cloud())
    assert score.compactness < 0.3
    assert any("exploded" in f or "floater" in f for f in score.flags)


def test_high_removed_fraction_lowers_retention() -> None:
    junky = score_cloud(_healthy_cloud(), clean_removed_fraction=0.6)
    clean = score_cloud(_healthy_cloud(), clean_removed_fraction=0.0)
    assert junky.retention is not None and junky.retention < 0.2
    assert clean.retention == 1.0
    assert any("floater" in f for f in junky.flags)


def test_low_psnr_flags_and_lowers_fidelity() -> None:
    score = score_cloud(_healthy_cloud(), selfconsistency_psnr_db=10.0)
    assert score.fidelity == 0.0
    assert any("PSNR" in f for f in score.flags)


def test_optional_inputs_drop_out_of_overall() -> None:
    # With neither retention nor fidelity supplied, overall blends only the two
    # geometry sub-scores and must still be a valid [0,1] number.
    score = score_cloud(_healthy_cloud())
    assert score.retention is None
    assert score.fidelity is None
    assert 0.0 <= score.overall <= 1.0


def test_scorecard_is_json_serialisable() -> None:
    import json

    score = score_cloud(
        _healthy_cloud(), clean_removed_fraction=0.05, selfconsistency_psnr_db=30.0
    )
    json.dumps(score.to_dict())
