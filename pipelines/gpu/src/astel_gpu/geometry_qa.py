"""Degenerate-asset critic — score a generated gaussian cloud for soundness.

The reference-image critic (:mod:`astel_gpu.image_qa`) catches a bad *input*; this
catches a bad *output*. Even from a clean image, TripoSplat samples from random
noise (a fresh draw per task) and occasionally produces a degenerate cloud:
mostly-transparent "smoke", an exploded floater halo, or geometry so collapsed the
2DGS distillation cannot even reproduce it. Those are the asset-side half of "same
prompt, sometimes wrong".

This module scores a cloud from CHEAP statistics only — no rendering, no GPU
kernel — so it can gate a best-of-K asset re-roll (and feed the Truth Meter)
without materially adding cost. Sub-scores (each ``[0, 1]``, higher = better):

* **retention** — fraction of the raw generation that survived floater cleaning.
  A cloud that was mostly junk (high removed-fraction) is a bad draw.
* **opacity** — share of reasonably-solid splats. A cloud dominated by
  near-transparent gaussians renders as faint smoke.
* **compactness** — radial-distance ``p99/p50`` ratio about the centroid. A clean
  object is compact; a floater-haloed or exploded cloud has a long radial tail.
* **fidelity** — the held-out self-consistency PSNR (when supplied). A very low
  value means the surfel distillation could not reproduce the generator at all.

HONESTY: this scores a GENERATED asset's internal soundness, never accuracy versus
a real object (a generated object has no scan to compare against).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch

from .gaussians import GaussianParams


@dataclass(frozen=True)
class GeometryQAConfig:
    """Thresholds + weights for :func:`score_cloud`."""

    #: ``retention`` hits 0 once this fraction (or more) of the raw cloud was
    #: removed by cleaning; 0 removed -> 1.
    retention_zero_removed_fraction: float = 0.5
    #: A splat counts as "solid" at/above this activated opacity.
    solid_opacity_min: float = 0.3
    #: ``opacity`` hits 1 once this share of splats are solid.
    opacity_full_solid_fraction: float = 0.5
    #: ``compactness`` is 1 at/below this radial p99/p50 ratio and 0 at/above
    #: ``compactness_ratio_bad``.
    compactness_ratio_good: float = 4.0
    compactness_ratio_bad: float = 16.0
    #: ``fidelity`` ramps 0->1 across this PSNR (dB) band.
    fidelity_psnr_floor: float = 15.0
    fidelity_psnr_ceil: float = 30.0
    #: Overall acceptance threshold.
    accept_threshold: float = 0.5
    #: Sub-score weights (overall is weight-normalised over the AVAILABLE scores;
    #: retention/fidelity drop out when their inputs are absent).
    w_retention: float = 0.25
    w_opacity: float = 0.25
    w_compactness: float = 0.3
    w_fidelity: float = 0.2


@dataclass(frozen=True)
class GeometryScore:
    """Per-cloud scorecard from :func:`score_cloud`."""

    overall: float
    accept: bool
    retention: float | None
    opacity: float
    compactness: float
    fidelity: float | None
    solid_fraction: float
    radial_p99_over_p50: float
    splats: int
    flags: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _ramp(value: float, lo: float, hi: float) -> float:
    """Linear ramp: 0 at/below ``lo``, 1 at/above ``hi`` (handles lo>=hi)."""
    if hi <= lo:
        return 1.0 if value >= hi else 0.0
    return float(max(0.0, min(1.0, (value - lo) / (hi - lo))))


def _radial_ratio(means: torch.Tensor) -> float:
    """Radial-distance p99/p50 about the centroid (1.0 for a degenerate cloud)."""
    if means.shape[0] < 8:
        return 1.0
    centroid = means.mean(dim=0)
    dist = (means - centroid).norm(dim=-1)
    p50 = torch.quantile(dist, 0.50).clamp_min(1e-8)
    p99 = torch.quantile(dist, 0.99)
    return float(p99 / p50)


def score_cloud(
    params: GaussianParams,
    *,
    clean_removed_fraction: float | None = None,
    selfconsistency_psnr_db: float | None = None,
    config: GeometryQAConfig | None = None,
) -> GeometryScore:
    """Score a generated cloud's internal soundness from cheap statistics.

    Pure (CPU-runnable; no gsplat). ``clean_removed_fraction`` (from
    :func:`astel_gpu.splat_clean.clean_gaussians` stats) and
    ``selfconsistency_psnr_db`` (the held-out distillation PSNR) are optional; the
    corresponding sub-score is omitted from the weighted overall when absent.
    Returns a :class:`GeometryScore` with ``flags`` naming each crossed threshold.
    """
    cfg = config or GeometryQAConfig()
    flags: list[str] = []
    n = params.count

    opacities = params.opacities.detach().float()
    solid_fraction = (
        float((opacities >= cfg.solid_opacity_min).float().mean()) if n else 0.0
    )
    opacity_score = _ramp(solid_fraction, 0.0, cfg.opacity_full_solid_fraction)
    if solid_fraction < 0.5 * cfg.opacity_full_solid_fraction:
        flags.append(f"low solid-opacity fraction ({solid_fraction:.2f})")

    radial_ratio = _radial_ratio(params.means.detach().float())
    compactness = 1.0 - _ramp(
        radial_ratio, cfg.compactness_ratio_good, cfg.compactness_ratio_bad
    )
    if radial_ratio >= cfg.compactness_ratio_bad:
        flags.append(f"floater-haloed / exploded cloud (p99/p50={radial_ratio:.1f})")

    weights: list[float] = [cfg.w_opacity, cfg.w_compactness]
    scores: list[float] = [opacity_score, compactness]

    retention: float | None = None
    if clean_removed_fraction is not None:
        retention = 1.0 - _ramp(
            clean_removed_fraction, 0.0, cfg.retention_zero_removed_fraction
        )
        weights.append(cfg.w_retention)
        scores.append(retention)
        if clean_removed_fraction >= 0.5 * cfg.retention_zero_removed_fraction:
            flags.append(
                f"large fraction removed as floaters ({clean_removed_fraction:.2f})"
            )

    fidelity: float | None = None
    if selfconsistency_psnr_db is not None:
        fidelity = _ramp(
            selfconsistency_psnr_db, cfg.fidelity_psnr_floor, cfg.fidelity_psnr_ceil
        )
        weights.append(cfg.w_fidelity)
        scores.append(fidelity)
        if selfconsistency_psnr_db < cfg.fidelity_psnr_floor:
            flags.append(
                f"low self-consistency PSNR ({selfconsistency_psnr_db:.1f} dB)"
            )

    total_w = sum(weights)
    weighted = sum(w * s for w, s in zip(weights, scores, strict=True))
    overall = float(weighted / total_w) if total_w else 0.0

    return GeometryScore(
        overall=overall,
        accept=overall >= cfg.accept_threshold,
        retention=retention,
        opacity=opacity_score,
        compactness=compactness,
        fidelity=fidelity,
        solid_fraction=solid_fraction,
        radial_p99_over_p50=radial_ratio,
        splats=n,
        flags=flags,
    )
