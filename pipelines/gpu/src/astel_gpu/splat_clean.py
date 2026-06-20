"""Splat cleanup — remove the floaters/needles/blobs raw generators emit.

TripoSplat (L2) — like every feed-forward / SDS gaussian generator — sprays a
minority of junk splats around the real object: faint dark "smoke" (low-opacity
gaussians), oversized translucent blobs forming a halo, elongated "needle"
gaussians that fan out as streaks, and disconnected floater clusters hanging in
empty space. The L3 distillation FREEZES positions and is trained to *reproduce*
the L2 appearance, so it faithfully reproduces those floaters instead of removing
them. The fix is to delete the junk before it is baked in.

This module is a torch-native, scale-invariant cleaner. Every filter is relative
to the cloud's own distribution (median scale, global mean/std of neighbour
distance), so it works whether the cloud is in TripoSplat's native units or a
normalised unit frame, and on both 3D gaussians (L2) and flat 2DGS surfels (L3):

* **opacity floor** — drop near-transparent splats (the dark smoky fuzz).
* **elongation cap** — drop needles. Elongation is ``s_max / s_mid`` (the ratio
  of the two LARGEST extents), so an intentionally-flat surfel disc
  (``s_min ≈ 0``, ``s_mid ≈ s_max``) is NOT penalised — only genuine slivers are.
* **oversize cap** — drop blobs whose largest extent dwarfs the median (the halo).
* **statistical outlier removal** — drop splats whose mean distance to their k
  nearest neighbours is an outlier vs the whole cloud (the disconnected floaters).
  Same definition as Open3D's ``remove_statistical_outlier``; KD-tree via scipy.

HONESTY: cleaning removes GENERATED junk, never measured reality — this only ever
runs on generated L2/L3 clouds. Every stage's removal count is returned so the
producer logs exactly what was deleted; nothing is silently dropped.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from .gaussians import GaussianParams

_EPS = 1e-8


@dataclass(frozen=True)
class CleanConfig:
    """Thresholds for :func:`clean_gaussians`. All scale-invariant / relative.

    Defaults are conservative — tuned to remove obvious floaters without eating
    into legitimate surface detail of an otherwise-good object. Override per-run
    via the environment (see :meth:`from_env`) without touching code.
    """

    #: Keep splats with activated opacity >= this. Kills faint dark "smoke".
    opacity_min: float = 0.06
    #: Keep splats with ``s_max / s_mid`` <= this. Kills "needle" streaks while
    #: leaving flat surfels (small ``s_min``, balanced ``s_mid``/``s_max``) alone.
    max_elongation: float = 12.0
    #: Keep splats whose largest extent <= ``max_scale_factor * median(s_max)``.
    #: Kills the few oversized translucent blobs that form the halo.
    max_scale_factor: float = 10.0
    #: k for statistical outlier removal (neighbours per splat, self excluded).
    sor_neighbors: int = 16
    #: Keep splats whose mean-kNN-distance <= ``mean + std_ratio * std`` over the
    #: whole cloud. Lower == more aggressive. Kills disconnected floaters.
    sor_std_ratio: float = 2.0
    #: Repeat the SOR pass this many times (each pass tightens the distribution).
    sor_iters: int = 2
    #: Master switch. ``False`` disables all cleaning (passes the cloud through).
    enabled: bool = True

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> CleanConfig:
        """Build a config from ``ASTEL_CLEAN*`` env vars, falling back to defaults.

        Lets the founder dial cleanup strength per run without editing code:
        ``ASTEL_CLEAN=0`` disables it; ``ASTEL_CLEAN_OPACITY_MIN``,
        ``ASTEL_CLEAN_MAX_ELONGATION``, ``ASTEL_CLEAN_MAX_SCALE_FACTOR``,
        ``ASTEL_CLEAN_SOR_NEIGHBORS``, ``ASTEL_CLEAN_SOR_STD_RATIO``,
        ``ASTEL_CLEAN_SOR_ITERS`` override individual thresholds.
        """
        env = env if env is not None else dict(os.environ)
        d = cls()

        def _flag(name: str, default: bool) -> bool:
            raw = env.get(name)
            if raw is None:
                return default
            return raw.strip().lower() not in {"0", "false", "no", "off", ""}

        def _f(name: str, default: float) -> float:
            raw = env.get(name)
            try:
                return float(raw) if raw is not None and raw != "" else default
            except ValueError:
                return default

        def _i(name: str, default: int) -> int:
            raw = env.get(name)
            try:
                return int(raw) if raw is not None and raw != "" else default
            except ValueError:
                return default

        return cls(
            opacity_min=_f("ASTEL_CLEAN_OPACITY_MIN", d.opacity_min),
            max_elongation=_f("ASTEL_CLEAN_MAX_ELONGATION", d.max_elongation),
            max_scale_factor=_f("ASTEL_CLEAN_MAX_SCALE_FACTOR", d.max_scale_factor),
            sor_neighbors=_i("ASTEL_CLEAN_SOR_NEIGHBORS", d.sor_neighbors),
            sor_std_ratio=_f("ASTEL_CLEAN_SOR_STD_RATIO", d.sor_std_ratio),
            sor_iters=_i("ASTEL_CLEAN_SOR_ITERS", d.sor_iters),
            enabled=_flag("ASTEL_CLEAN", d.enabled),
        )


def _subset(params: GaussianParams, mask: torch.Tensor) -> GaussianParams:
    """Index every per-splat tensor by a boolean ``mask``."""
    return GaussianParams(
        means=params.means[mask],
        scales=params.scales[mask],
        quats=params.quats[mask],
        opacities=params.opacities[mask],
        colors=params.colors[mask],
    )


def opacity_mask(opacities: torch.Tensor, opacity_min: float) -> torch.Tensor:
    """Keep splats with activated opacity >= ``opacity_min`` (``[N]`` bool)."""
    return opacities >= opacity_min


def elongation_mask(scales: torch.Tensor, max_elongation: float) -> torch.Tensor:
    """Keep splats whose two largest extents are within ``max_elongation``.

    ``elongation = s_max / s_mid`` from the per-splat sorted extents, so a flat
    surfel disc (tiny ``s_min``, comparable ``s_mid``/``s_max``) reads ~1 and is
    kept, while a sliver/needle (``s_mid ≈ 0``) reads huge and is dropped.
    """
    s_sorted, _ = torch.sort(scales.abs(), dim=1)
    s_mid = s_sorted[:, -2].clamp_min(_EPS)
    s_max = s_sorted[:, -1]
    return (s_max / s_mid) <= max_elongation


def oversize_mask(scales: torch.Tensor, max_scale_factor: float) -> torch.Tensor:
    """Keep splats whose largest extent <= ``max_scale_factor * median(s_max)``."""
    s_max = scales.abs().max(dim=1).values
    median = s_max.median().clamp_min(_EPS)
    return s_max <= median * max_scale_factor


def _knn_mean_distance(points: np.ndarray, k: int) -> np.ndarray:
    """Mean distance from each point to its ``k`` nearest neighbours (self excl.).

    Uses a scipy KD-tree (fast for the 10^5–10^6 splat counts we ship). Falls
    back to a dense pairwise computation only for tiny clouds / if scipy is
    unavailable, which is all the CPU tests need.
    """
    n = points.shape[0]
    k_eff = min(k, max(1, n - 1))
    try:
        from scipy.spatial import cKDTree  # noqa: PLC0415 (optional heavy dep)

        tree = cKDTree(points)
        # k+1: the first neighbour is the point itself (distance 0).
        dist, _ = tree.query(points, k=k_eff + 1, workers=-1)
        dist = np.atleast_2d(dist)
        neighbours = dist[:, 1:]
    except ImportError:
        diff = points[:, None, :] - points[None, :, :]
        full = np.sqrt((diff * diff).sum(axis=-1))
        full.sort(axis=1)
        neighbours = full[:, 1 : k_eff + 1]
    if neighbours.shape[1] == 0:
        return np.zeros(n, dtype=np.float64)
    return np.asarray(neighbours.mean(axis=1), dtype=np.float64)


def statistical_outlier_mask(
    means: torch.Tensor, *, nb_neighbors: int, std_ratio: float
) -> torch.Tensor:
    """Keep splats whose mean-kNN-distance is not a global outlier (``[N]`` bool).

    Threshold = ``mean + std_ratio * std`` of the per-splat mean-kNN distances,
    matching Open3D's ``remove_statistical_outlier``. Disconnected floaters sit
    far from everything, so their mean-kNN distance is large and they are dropped.
    """
    n = int(means.shape[0])
    if n <= nb_neighbors:
        return torch.ones(n, dtype=torch.bool, device=means.device)
    pts = means.detach().cpu().to(torch.float64).numpy()
    mean_d = _knn_mean_distance(pts, nb_neighbors)
    threshold = float(mean_d.mean() + std_ratio * mean_d.std())
    keep_np = mean_d <= threshold
    return torch.from_numpy(keep_np).to(means.device)


def _apply(
    params: GaussianParams, mask: torch.Tensor, stats: dict[str, Any], key: str
) -> GaussianParams:
    """Apply ``mask`` unless it would empty the cloud; record removed count.

    A filter that keeps zero splats is a misconfiguration, not a result — we skip
    it (record ``-1``) rather than destroy the asset. Otherwise we subset and
    record how many splats this stage removed.
    """
    keep = int(mask.sum())
    if keep == 0:
        stats[key] = -1  # skipped: would have removed everything
        return params
    stats[key] = int(params.count - keep)
    return _subset(params, mask)


def clean_gaussians(
    params: GaussianParams,
    config: CleanConfig | None = None,
    *,
    spatial: bool = True,
) -> tuple[GaussianParams, dict[str, Any]]:
    """Remove floaters / needles / blobs from a generated gaussian cloud.

    Applies, in order: opacity floor → elongation cap → oversize cap → (optional)
    statistical outlier removal. Cheap per-splat filters run first so the KD-tree
    in SOR operates on the already-thinned cloud. ``spatial=False`` skips SOR —
    used for the cheap final L3 pass, where positions are unchanged from the
    already-SOR'd L2 input so re-running it is wasted work.

    Returns ``(cleaned_params, stats)``. ``stats`` records the input count, the
    splats removed at each stage (``-1`` == stage skipped to avoid emptying the
    cloud), the totals, and the config used — so the producer can log exactly
    what was deleted (CLAUDE.md §10.4: no silent drops).
    """
    config = config or CleanConfig()
    n_in = params.count
    stats: dict[str, Any] = {
        "enabled": config.enabled,
        "input": n_in,
        "opacity_removed": 0,
        "elongation_removed": 0,
        "oversize_removed": 0,
        "spatial_outliers_removed": 0,
        "config": asdict(config),
    }

    if not config.enabled or n_in == 0:
        stats["kept"] = n_in
        stats["removed_total"] = 0
        stats["removed_fraction"] = 0.0
        return params, stats

    params = _apply(
        params,
        opacity_mask(params.opacities, config.opacity_min),
        stats,
        "opacity_removed",
    )
    params = _apply(
        params,
        elongation_mask(params.scales, config.max_elongation),
        stats,
        "elongation_removed",
    )
    params = _apply(
        params,
        oversize_mask(params.scales, config.max_scale_factor),
        stats,
        "oversize_removed",
    )
    if spatial and config.sor_iters > 0 and config.sor_neighbors > 0:
        removed = 0
        for _ in range(config.sor_iters):
            before = params.count
            params = _apply(
                params,
                statistical_outlier_mask(
                    params.means,
                    nb_neighbors=config.sor_neighbors,
                    std_ratio=config.sor_std_ratio,
                ),
                stats,
                "spatial_outliers_removed",
            )
            removed += max(0, before - params.count)
            if params.count == before:
                break  # converged: nothing more to drop
        stats["spatial_outliers_removed"] = removed

    stats["kept"] = params.count
    stats["removed_total"] = n_in - params.count
    stats["removed_fraction"] = (n_in - params.count) / n_in if n_in else 0.0
    return params, stats
