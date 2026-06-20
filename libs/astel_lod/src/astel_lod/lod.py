"""LOD tier index selection from a global importance ranking.

All public functions operate on raw numpy arrays and return **index arrays** so
that the caller subsamples its own cloud data without copying it here.

Nested LOD guarantee
--------------------
``generate_lod_indices`` derives every tier from the **same** global importance
sort (descending).  The top-*k* indices are always a subset of the top-*K*
indices when *k* ≤ *K*, so a streaming client that already holds a lower tier
never re-downloads overlapping splats when upgrading.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .importance import splat_importance


def select_lod_indices(
    importance: NDArray[np.floating],
    target_count: int,
) -> NDArray[np.intp]:
    """Return the indices of the top-``target_count`` Gaussians by importance.

    Parameters
    ----------
    importance:
        Shape ``(N,)``.  Per-Gaussian importance scores (e.g. from
        :func:`~astel_lod.importance.splat_importance`).
    target_count:
        Number of Gaussians to select.  Must be > 0.

    Returns
    -------
    NDArray[np.intp]
        1-D integer array of length ``min(target_count, N)``, sorted in
        **ascending index order** so that the result can be used directly as a
        fancy-index slice that preserves the original splat ordering.

    Raises
    ------
    ValueError
        If ``target_count <= 0``.
    """
    if target_count <= 0:
        msg = f"target_count must be > 0, got {target_count}"
        raise ValueError(msg)

    imp = np.asarray(importance, dtype=np.float64)
    n = imp.shape[0]

    if target_count >= n:
        return np.arange(n, dtype=np.intp)

    # argpartition gives the k smallest; we want the k largest → negate.
    # Using argpartition is O(N) average vs O(N log N) for a full sort —
    # worthwhile for large splat clouds.
    k = target_count
    # Indices of the top-k in arbitrary order:
    top_k_unsorted: NDArray[np.intp] = np.argpartition(-imp, k - 1)[:k].astype(np.intp)
    # Sort ascending so the output preserves original cloud order.
    return np.sort(top_k_unsorted)


def generate_lod_indices(
    opacity: NDArray[np.floating],
    log_scales: NDArray[np.floating],
    target_counts: list[int],
) -> list[NDArray[np.intp]]:
    """Compute importance once, then return one index array per target count.

    All tiers are derived from the **same** global descending importance
    ranking, which guarantees the nested-subset property: the index set for a
    smaller target is always a subset of the index set for a larger target.

    Parameters
    ----------
    opacity:
        Shape ``(N,)``.
    log_scales:
        Shape ``(N, 3)``.
    target_counts:
        Requested tier sizes.  Order does not matter; the function processes
        them smallest-first internally and returns results in the **same order
        as the input list**.

    Returns
    -------
    list[NDArray[np.intp]]
        One sorted index array per entry in ``target_counts``, in the same
        order as the input list.  Each array has length
        ``min(target_counts[i], N)``.

    Raises
    ------
    ValueError
        If any entry in ``target_counts`` is ≤ 0 (delegated to
        :func:`select_lod_indices`).
    """
    imp = splat_importance(opacity, log_scales)
    n = imp.shape[0]

    # Compute a single full descending sort once.
    # For each target count k we take the first k entries of this sorted order.
    sorted_desc: NDArray[np.intp] = np.argsort(-imp, kind="stable").astype(np.intp)

    results: list[NDArray[np.intp]] = []
    for k in target_counts:
        if k <= 0:
            msg = f"target_count must be > 0, got {k}"
            raise ValueError(msg)
        actual_k = min(k, n)
        top_k = sorted_desc[:actual_k]
        results.append(np.sort(top_k))

    return results
