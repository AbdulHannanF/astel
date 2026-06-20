"""Fit a DeformationField to observed per-frame point positions.

Uses farthest-point sampling to select K control nodes, Gaussian RBF weights
(normalised to sum-to-one per gaussian), and per-frame weighted least-squares
to solve an affine transform [R|t] for each node.

HONESTY (CLAUDE.md §10.4): FitReport carries the REAL measured reconstruction
error vs. the input frames.  The fitter makes no attempt to hide large
residuals — high-rank or incompressible motion will honestly produce large
per_frame_mean_err values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np

from .field import DeformationField


@dataclass
class FitReport:
    """Reconstruction quality report for a fitted :class:`DeformationField`.

    All error values are in the same units as the input positions.

    Parameters
    ----------
    per_frame_mean_err:
        Mean per-gaussian Euclidean error per frame.
    per_frame_p95_err:
        95th-percentile per-gaussian Euclidean error per frame.
    overall_mean_err:
        Mean of ``per_frame_mean_err`` across all frames.
    overall_p95_err:
        Mean of ``per_frame_p95_err`` across all frames.
    n_nodes:
        Number of LBS control nodes used.
    note:
        Human-readable description of the approximation quality.
    """

    per_frame_mean_err: list[float]
    per_frame_p95_err: list[float]
    overall_mean_err: float
    overall_p95_err: float
    n_nodes: int
    note: str


def _farthest_point_sample(points: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """Return indices of *k* farthest-point-sampled points from *points*.

    Parameters
    ----------
    points:
        Shape ``(N, 3)``.
    k:
        Number of points to select; clamped to N if k >= N.
    seed:
        Index of the first point (deterministic).

    Returns
    -------
    np.ndarray
        Integer indices into *points*, shape ``(min(k, N),)``.
    """
    n = points.shape[0]
    k = min(k, n)
    selected = np.empty(k, dtype=np.int64)
    dists = np.full(n, np.inf, dtype=np.float64)

    idx = seed % n
    for i in range(k):
        selected[i] = idx
        diff = points - points[idx]
        d = np.einsum("nd,nd->n", diff, diff)
        np.minimum(dists, d, out=dists)
        idx = int(np.argmax(dists))

    return selected


def _rbf_weights(
    base_positions: np.ndarray,  # (N, 3)
    node_positions: np.ndarray,  # (K, 3)
) -> np.ndarray:
    """Gaussian RBF weights, normalised so each row sums to 1.

    sigma is set to the median nearest-node distance across all gaussians.
    If all gaussians collapse to the same point (degenerate), sigma=1 is used.
    """
    # Pairwise distances (N, K)
    diff = base_positions[:, np.newaxis, :] - node_positions[np.newaxis, :, :]
    dist2 = np.einsum("nkd,nkd->nk", diff, diff)  # (N, K)

    # sigma = median of the nearest-node distance per gaussian
    nearest_dist = np.sqrt(np.min(dist2, axis=1))  # (N,)
    sigma = float(np.median(nearest_dist))
    if sigma < 1e-12:
        sigma = 1.0

    w = np.exp(-dist2 / (2.0 * sigma**2))  # (N, K)

    # Normalise rows
    row_sums = w.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums < 1e-30, 1.0, row_sums)
    return cast(np.ndarray, (w / row_sums).astype(np.float32))


def _fit_node_transform(
    base: np.ndarray,  # (N, 3)
    target: np.ndarray,  # (N, 3)
    node_w: np.ndarray,  # (N,)  per-gaussian weight for this node
) -> np.ndarray:
    """Solve a weighted affine transform mapping base → target.

    Minimises  Σ_n  w_n * || A @ base[n] + t  - target[n] ||²
    via a homogeneous least-squares system.

    Returns
    -------
    np.ndarray
        Shape ``(3, 4)``: ``[A | t]`` where A is 3×3 and t is the translation.
    """
    # Build system: for each point, the row is [base[n].T | 1] → predicts target[n].
    # We solve for the transpose: X @ [b; 1] ≈ t_row
    # Weighted: scale rows by sqrt(w).
    sqrt_w = np.sqrt(np.maximum(node_w, 0.0))[:, np.newaxis]  # (N, 1)

    # Design matrix: (N, 4) — homogeneous base coords
    ones = np.ones((base.shape[0], 1), dtype=np.float64)
    A_mat = np.concatenate([base, ones], axis=1)
    A_mat = (A_mat * sqrt_w).astype(np.float64)  # weighted

    B_mat = (target * sqrt_w).astype(np.float64)  # (N, 3) weighted

    # Solve: A_mat @ X ≈ B_mat  →  X is (4, 3)
    result, _, _, _ = np.linalg.lstsq(A_mat, B_mat, rcond=None)

    # result: (4, 3)  →  transpose → (3, 4): [row=output_dim, col=input_dim|bias]
    tf = cast(np.ndarray, result.T.astype(np.float32))  # (3, 4)
    return tf


def fit_deformation_field(
    base_positions: np.ndarray,
    frames_positions: np.ndarray,
    n_nodes: int,
    *,
    seed: int = 0,
) -> tuple[DeformationField, FitReport]:
    """Fit an LBS :class:`DeformationField` to observed per-frame positions.

    Parameters
    ----------
    base_positions:
        Rest-pose Gaussian means, shape ``(N, 3)``.
    frames_positions:
        Observed positions for each frame, shape ``(F, N, 3)``.
    n_nodes:
        Number of LBS control nodes K (clamped to N if K >= N).
    seed:
        Seed point index for farthest-point sampling (deterministic).

    Returns
    -------
    tuple[DeformationField, FitReport]
        The fitted field and an honest quality report.
    """
    base = np.asarray(base_positions, dtype=np.float64)
    frames = np.asarray(frames_positions, dtype=np.float64)

    if base.ndim != 2 or base.shape[1] != 3:
        raise ValueError(f"base_positions must be (N, 3), got {base.shape}")
    if frames.ndim != 3 or frames.shape[1:] != base.shape:
        raise ValueError(
            f"frames_positions must be (F, {base.shape[0]}, 3), got {frames.shape}"
        )

    N = base.shape[0]
    F = frames.shape[0]
    K = min(n_nodes, N)

    # --- Step 1: farthest-point sampling for node positions -----------------
    node_idx = _farthest_point_sample(base, K, seed=seed)
    node_positions = base[node_idx].astype(np.float32)  # (K, 3)

    # --- Step 2: Gaussian RBF blend weights ---------------------------------
    weights = _rbf_weights(base.astype(np.float32), node_positions)  # (N, K)

    # --- Step 3: per-frame, per-node weighted affine solve ------------------
    node_transforms = np.zeros((F, K, 3, 4), dtype=np.float32)

    for f in range(F):
        target_f = frames[f]  # (N, 3)
        for k in range(K):
            node_w = weights[:, k].astype(np.float64)  # (N,)
            tf = _fit_node_transform(base, target_f, node_w)  # (3, 4)
            node_transforms[f, k] = tf

    # --- Step 4: build DeformationField -------------------------------------
    field = DeformationField(
        node_positions=node_positions,
        weights=weights,
        node_transforms=node_transforms,
    )

    # --- Step 5: measure REAL reconstruction error --------------------------
    per_frame_mean: list[float] = []
    per_frame_p95: list[float] = []

    for f in range(F):
        pred = field.apply(base.astype(np.float32), frame=f)  # (N, 3)
        err = np.linalg.norm(pred.astype(np.float64) - frames[f], axis=1)  # (N,)
        per_frame_mean.append(float(np.mean(err)))
        per_frame_p95.append(float(np.percentile(err, 95)))

    overall_mean = float(np.mean(per_frame_mean))
    overall_p95 = float(np.mean(per_frame_p95))

    note = (
        f"Affine-LBS approximation with K={K} nodes. "
        "Motion well-described by LBS (global rotation, simple bends) fits "
        "tightly. High-rank or incompressible motion produces honest residuals — "
        "errors above are the true measured reconstruction error, not a bound."
    )

    report = FitReport(
        per_frame_mean_err=per_frame_mean,
        per_frame_p95_err=per_frame_p95,
        overall_mean_err=overall_mean,
        overall_p95_err=overall_p95,
        n_nodes=K,
        note=note,
    )

    return field, report
