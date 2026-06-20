"""Splat cleanup — remove the floaters/needles/blobs raw generators emit.

TripoSplat (L2) — like every feed-forward / SDS gaussian generator — sprays a
minority of junk splats around the real object: faint dark "smoke" (low-opacity
gaussians), oversized translucent blobs forming a halo, elongated "needle"
gaussians that fan out as streaks, and disconnected floater clusters hanging in
empty space. The L3 distillation FREEZES positions and is trained to *reproduce*
the L2 appearance, so it faithfully reproduces those floaters instead of removing
them. The fix is to delete the junk before it is baked in.

GEOMETRY-PRESERVING BY DESIGN. The cardinal rule (founder, 2026-06-20): a cleaner
that eats real geometry is worse than no cleaner — "we have no use for a broken
splat". So every filter here is conservative and, crucially, **density-agnostic**
where it matters:

* **opacity floor** — drop near-invisible splats (the dark smoky fuzz).
* **elongation cap** — drop needles. Elongation is ``s_max / s_mid`` (the ratio
  of the two LARGEST extents), so a flat surfel disc (``s_min ≈ 0``,
  ``s_mid ≈ s_max``) is NOT penalised — only genuine slivers are.
* **oversize cap** — drop blobs whose largest extent dwarfs the median (the halo).
* **connected-component removal** — keep the large connected cluster(s); drop
  only floaters that are *spatially disconnected* from the body. This is
  density-agnostic: thin/weak surface regions stay because they remain connected
  to the main object, no matter how sparse they are.

What we deliberately do NOT do by default: **statistical outlier removal** (SOR).
SOR thresholds on a GLOBAL mean+std of neighbour distance, so on a real generated
cloud — dense where there is detail, sparse on smooth faces — it flags whole
legitimate low-density regions as outliers and deletes them (measured: ~40% of a
real helmet vanished). It remains available, opt-in, for the rare uniform-density
case; it is OFF by default.

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
    """Thresholds for :func:`clean_gaussians`. Conservative and geometry-safe.

    Defaults remove obvious floaters without eating into legitimate surface
    detail. Override per-run via the environment (see :meth:`from_env`).
    """

    #: Keep splats with activated opacity >= this. Kills faint dark "smoke".
    #: Low (0.04): generated opaque objects rarely have legitimate sub-0.04 splats.
    opacity_min: float = 0.04
    #: Keep splats with ``s_max / s_mid`` <= this. Kills "needle" streaks while
    #: leaving flat surfels (small ``s_min``, balanced ``s_mid``/``s_max``) alone.
    #: High (16): only genuine slivers exceed it; honest surface splats do not.
    max_elongation: float = 16.0
    #: Keep splats whose largest extent <= ``max_scale_factor * median(s_max)``.
    #: Kills the few oversized translucent blobs that form the halo.
    max_scale_factor: float = 12.0

    #: Connected-component floater removal — the default spatial filter. Density-
    #: agnostic: keeps any cluster reachable through the point graph, so thin/weak
    #: *attached* geometry survives and only disconnected floaters are dropped.
    components_enabled: bool = True
    #: Neighbours per splat used to build the connectivity graph.
    cc_neighbors: int = 16
    #: Two splats are "connected" if within ``cc_radius_factor * median_nn_dist``.
    #: Generous (6) so the real body never fragments; a too-large value only
    #: under-cleans (safe), a too-small one would split the body (but large
    #: fragments are still kept, so no geometry is lost either way).
    cc_radius_factor: float = 6.0
    #: Keep components whose size >= ``cc_min_cluster_fraction * largest``.
    cc_min_cluster_fraction: float = 0.01
    #: Absolute floor: never keep a component smaller than this many splats.
    cc_min_cluster_size: int = 64

    #: Statistical outlier removal — OFF by default (``sor_iters = 0``). Uses a
    #: GLOBAL density threshold that deletes legitimate low-density regions on real
    #: generated clouds; opt-in only, for uniform-density inputs.
    sor_neighbors: int = 16
    sor_std_ratio: float = 2.0
    sor_iters: int = 0

    #: Master switch. ``False`` disables all cleaning (passes the cloud through).
    enabled: bool = True

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> CleanConfig:
        """Build a config from ``ASTEL_CLEAN*`` env vars, falling back to defaults.

        Lets the founder dial cleanup strength per run without editing code.
        ``ASTEL_CLEAN=0`` disables it. Per-threshold overrides:
        ``ASTEL_CLEAN_OPACITY_MIN``, ``ASTEL_CLEAN_MAX_ELONGATION``,
        ``ASTEL_CLEAN_MAX_SCALE_FACTOR``, ``ASTEL_CLEAN_COMPONENTS`` (0/1),
        ``ASTEL_CLEAN_CC_RADIUS_FACTOR``, ``ASTEL_CLEAN_CC_MIN_FRACTION``,
        ``ASTEL_CLEAN_CC_MIN_SIZE``, ``ASTEL_CLEAN_SOR_NEIGHBORS``,
        ``ASTEL_CLEAN_SOR_STD_RATIO``, ``ASTEL_CLEAN_SOR_ITERS`` (>0 enables SOR).
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
            components_enabled=_flag("ASTEL_CLEAN_COMPONENTS", d.components_enabled),
            cc_neighbors=_i("ASTEL_CLEAN_CC_NEIGHBORS", d.cc_neighbors),
            cc_radius_factor=_f("ASTEL_CLEAN_CC_RADIUS_FACTOR", d.cc_radius_factor),
            cc_min_cluster_fraction=_f(
                "ASTEL_CLEAN_CC_MIN_FRACTION", d.cc_min_cluster_fraction
            ),
            cc_min_cluster_size=_i("ASTEL_CLEAN_CC_MIN_SIZE", d.cc_min_cluster_size),
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


def connected_component_mask(
    means: torch.Tensor,
    *,
    nb_neighbors: int,
    radius_factor: float,
    min_cluster_fraction: float,
    min_cluster_size: int,
) -> torch.Tensor:
    """Keep splats in large connected clusters; drop disconnected floaters.

    Builds a k-NN graph, links pairs closer than ``radius_factor`` times the
    cloud's median nearest-neighbour distance (so the link radius adapts to point
    spacing), labels connected components, and keeps every component whose size is
    at least ``max(min_cluster_size, min_cluster_fraction * largest_component)``.

    This is DENSITY-AGNOSTIC: a thin or weakly-sampled but *attached* surface
    region stays because it remains graph-connected to the body, unlike global
    statistical outlier removal which would delete it. Floaters sit in their own
    small components and are dropped. Degrades safely — if the body fragments, the
    large fragments are all kept, so no real geometry is lost (only under-cleaned).
    """
    n = int(means.shape[0])
    if n <= nb_neighbors or n < 2:
        return torch.ones(n, dtype=torch.bool, device=means.device)
    try:
        from scipy.sparse import csr_matrix  # noqa: PLC0415
        from scipy.sparse.csgraph import connected_components  # noqa: PLC0415
        from scipy.spatial import cKDTree  # noqa: PLC0415
    except ImportError:
        return torch.ones(n, dtype=torch.bool, device=means.device)

    pts = means.detach().cpu().to(torch.float64).numpy()
    k = min(nb_neighbors, n - 1)
    tree = cKDTree(pts)
    dist, idx = tree.query(pts, k=k + 1, workers=-1)  # self in column 0
    dist = np.atleast_2d(dist)
    idx = np.atleast_2d(idx)

    nn = dist[:, 1]  # nearest non-self neighbour distance per point
    positive = nn[nn > 0]
    if positive.size == 0:
        return torch.ones(n, dtype=torch.bool, device=means.device)
    radius = float(np.median(positive)) * radius_factor

    neigh_dist = dist[:, 1:]
    neigh_idx = idx[:, 1:]
    edge_mask = neigh_dist <= radius
    rows = np.repeat(np.arange(n), k)[edge_mask.ravel()]
    cols = neigh_idx.ravel()[edge_mask.ravel()]
    data = np.ones(rows.shape[0], dtype=np.uint8)
    graph = csr_matrix((data, (rows, cols)), shape=(n, n))

    _, labels = connected_components(graph, directed=False)
    sizes = np.bincount(labels)
    largest = int(sizes.max())
    threshold = max(min_cluster_size, int(min_cluster_fraction * largest))
    keep_labels = np.flatnonzero(sizes >= threshold)
    keep_np = np.isin(labels, keep_labels)
    return torch.from_numpy(keep_np).to(means.device)


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
    matching Open3D's ``remove_statistical_outlier``. WARNING: density-sensitive —
    deletes legitimate low-density surface on non-uniform generated clouds. Used
    only when explicitly enabled (``sor_iters > 0``); prefer
    :func:`connected_component_mask`.
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

    Applies, in order: opacity floor → elongation cap → oversize cap → (spatial)
    connected-component floater removal → (spatial, opt-in) statistical outlier
    removal. Cheap per-splat filters run first so the KD-trees operate on the
    already-thinned cloud. ``spatial=False`` skips both graph passes — used for
    the cheap final L3 pass, where positions are unchanged from the already-
    cleaned L2 input so re-running them is wasted work.

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
        "components_removed": 0,
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
    if spatial and config.components_enabled:
        params = _apply(
            params,
            connected_component_mask(
                params.means,
                nb_neighbors=config.cc_neighbors,
                radius_factor=config.cc_radius_factor,
                min_cluster_fraction=config.cc_min_cluster_fraction,
                min_cluster_size=config.cc_min_cluster_size,
            ),
            stats,
            "components_removed",
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
