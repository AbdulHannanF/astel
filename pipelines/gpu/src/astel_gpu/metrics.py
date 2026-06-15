"""Pure-torch geometry metrics (no gsplat import -> runs on CPU).

Used by the synthetic ground-truth eval harness (:mod:`astel_gpu.synthetic_eval`)
to compute a REAL measured Chamfer distance between a refit gaussian cloud's
means and a known synthetic ground-truth point cloud.
"""

from __future__ import annotations

import torch


def chamfer_distance(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    """Bidirectional Chamfer distance between point sets ``a`` (N,3) and ``b`` (M,3).

    Returns a dict with the mean nearest-neighbour distance in each direction
    (``a_to_b``, ``b_to_a``) and their symmetric mean (``symmetric``), all in
    the same units as the input points (typically metres).

    Uses ``torch.cdist`` for pairwise distances; for the small point counts
    used by the synthetic eval (thousands of points) this is fast and exact.
    """
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError(f"expected a with shape (N, 3), got {tuple(a.shape)}")
    if b.ndim != 2 or b.shape[1] != 3:
        raise ValueError(f"expected b with shape (M, 3), got {tuple(b.shape)}")

    dists = torch.cdist(a, b)  # (N, M)
    a_to_b = dists.min(dim=1).values.mean()
    b_to_a = dists.min(dim=0).values.mean()
    symmetric = 0.5 * (a_to_b + b_to_a)

    return {
        "a_to_b": float(a_to_b),
        "b_to_a": float(b_to_a),
        "symmetric": float(symmetric),
    }


def meters_to_millimeters(distances: dict[str, float]) -> dict[str, float]:
    """Convert a :func:`chamfer_distance` result (metres) to millimetres."""
    return {key: value * 1000.0 for key, value in distances.items()}


def _nn_min_dists(query: torch.Tensor, ref: torch.Tensor, chunk: int) -> torch.Tensor:
    """For each ``query`` point, the distance to its nearest ``ref`` point.

    Chunks over BOTH point sets so peak memory is ``O(chunk**2)`` regardless of
    the (possibly millions) of points -- required for real GT clouds (DTU's STL
    scan is ~2.9M points, far past what a single ``cdist`` can hold).
    """
    out = torch.empty(query.shape[0], device=query.device, dtype=query.dtype)
    for i in range(0, query.shape[0], chunk):
        qb = query[i : i + chunk]
        best: torch.Tensor | None = None
        for j in range(0, ref.shape[0], chunk):
            d = torch.cdist(qb, ref[j : j + chunk]).min(dim=1).values
            best = d if best is None else torch.minimum(best, d)
        assert best is not None  # ref is non-empty (validated by caller)
        out[i : i + chunk] = best
    return out


def nn_distances(
    query: torch.Tensor, ref: torch.Tensor, chunk_size: int = 4096
) -> torch.Tensor:
    """Per-``query``-point nearest-neighbour distance to ``ref`` (VRAM-safe).

    Returns a ``(N,)`` tensor; the caller can mask/clamp/average it (e.g. the
    DTU protocol filters which points count and caps distances at 60 mm before
    averaging).
    """
    if query.ndim != 2 or query.shape[1] != 3:
        raise ValueError(f"expected query (N, 3), got {tuple(query.shape)}")
    if ref.ndim != 2 or ref.shape[1] != 3:
        raise ValueError(f"expected ref (M, 3), got {tuple(ref.shape)}")
    if query.shape[0] == 0 or ref.shape[0] == 0:
        raise ValueError("both point sets must be non-empty")
    return _nn_min_dists(query, ref, chunk_size)


def chamfer_distance_chunked(
    a: torch.Tensor, b: torch.Tensor, chunk_size: int = 4096
) -> dict[str, float]:
    """Bidirectional Chamfer for large clouds, in VRAM-safe chunks.

    Same result and units as :func:`chamfer_distance` but computed in
    ``chunk_size``-sized blocks so millions of points fit in VRAM. Use this for
    real ground-truth clouds; the plain :func:`chamfer_distance` is fine for the
    small synthetic eval.
    """
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError(f"expected a with shape (N, 3), got {tuple(a.shape)}")
    if b.ndim != 2 or b.shape[1] != 3:
        raise ValueError(f"expected b with shape (M, 3), got {tuple(b.shape)}")
    if a.shape[0] == 0 or b.shape[0] == 0:
        raise ValueError("both point sets must be non-empty")

    a_to_b = _nn_min_dists(a, b, chunk_size).mean()
    b_to_a = _nn_min_dists(b, a, chunk_size).mean()
    symmetric = 0.5 * (a_to_b + b_to_a)
    return {
        "a_to_b": float(a_to_b),
        "b_to_a": float(b_to_a),
        "symmetric": float(symmetric),
    }
