"""Bradley-Terry model fitting for pairwise preference aggregation.

Per CORPUS.md §4.4: pairwise results feed a Bradley-Terry model to produce a
per-system "strength" score per axis (and overall), with 95% confidence
intervals (bootstrap acceptable). This module implements:

- A simple iterative MLE fit (Zermelo / minorization-maximization update,
  the standard fixed-point algorithm for Bradley-Terry -- no heavy deps,
  numpy only).
- A bootstrap-resampling CI estimator over the pairwise comparison records.

Bradley-Terry model: for systems i, j with strengths pi_i, pi_j > 0, the
probability that i beats j is pi_i / (pi_i + pi_j). We fit on the log scale
(``strength = exp(theta)``) for numerical stability and normalize so that
strengths sum to the number of systems (i.e. average strength = 1), which
makes scores comparable across axes/fits. Ties are split as half a win for
each side, the standard Bradley-Terry-with-ties convention.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.random import Generator, default_rng

from astel_eval.scoring_models import PairwisePreference


@dataclass(frozen=True, slots=True)
class BTResult:
    """Fitted Bradley-Terry strengths with bootstrap CIs for one axis."""

    systems: tuple[str, ...]
    """Systems in a stable order, matching ``strengths``/``ci_low``/``ci_high``."""

    strengths: tuple[float, ...]
    """Point-estimate strength per system (mean strength normalized to 1.0)."""

    ci_low: tuple[float, ...]
    """2.5th-percentile bootstrap strength per system."""

    ci_high: tuple[float, ...]
    """97.5th-percentile bootstrap strength per system."""

    n_comparisons: int
    """Total number of pairwise records used (including ties)."""

    def strength_of(self, system: str) -> float:
        return self.strengths[self.systems.index(system)]

    def ci_of(self, system: str) -> tuple[float, float]:
        idx = self.systems.index(system)
        return (self.ci_low[idx], self.ci_high[idx])


def _wins_matrix(
    prefs: list[PairwisePreference], systems: tuple[str, ...]
) -> np.ndarray:
    """Build an (n, n) matrix W where W[i, j] = effective wins of i over j.

    Ties contribute 0.5 to each direction.
    """
    idx = {s: k for k, s in enumerate(systems)}
    n = len(systems)
    wins = np.zeros((n, n), dtype=np.float64)
    for p in prefs:
        if p.system_a not in idx or p.system_b not in idx:
            continue
        a, b = idx[p.system_a], idx[p.system_b]
        if p.winner is None:
            wins[a, b] += 0.5
            wins[b, a] += 0.5
        elif p.winner == p.system_a:
            wins[a, b] += 1.0
        else:
            wins[b, a] += 1.0
    return wins


def _fit_strengths(
    wins: np.ndarray,
    *,
    max_iter: int = 200,
    tol: float = 1e-10,
    prior_tie: float = 0.5,
) -> np.ndarray:
    """Fit Bradley-Terry strengths via the Zermelo fixed-point iteration.

    ``wins[i, j]`` = number of times i beat j (ties pre-split as 0.5/0.5).
    Returns strengths normalized so their mean is 1.0. Systems with zero total
    comparisons get strength 1.0 (no information -> neutral prior).

    Smoothing prior
    ---------------
    Real corpora frequently contain *fully separated* pairs -- e.g. Astel wins
    every single head-to-head against a baseline. That is a well-known
    Bradley-Terry MLE pathology: the unregularized loser's strength diverges to
    0 and the fixed-point iteration never reaches ``tol``, so it would run all
    ``max_iter`` iterations (the dominant cost of the eval suite).

    To make the estimate finite/stable *and* fast-converging, we add a small
    symmetric fictitious tie of ``prior_tie`` wins in each direction to every
    pair that was actually compared (i.e. where ``totals[i, j] > 0``). This is a
    weak Beta-style prior that pulls separated estimates toward equality just
    enough to guarantee a finite fixed point, while leaving the *ordering* and
    near-balanced fits essentially unchanged. With the prior in place,
    convergence is fast, so ``max_iter`` is a sane 200 (vs. the old 10_000) and
    we early-stop on relative change.

    The vectorized update precomputes ``numerator = wins.sum(axis=1)`` once and
    computes the denominator with broadcasting:
    ``S = strengths[:, None] + strengths[None, :]; (totals / S).sum(axis=1)``.
    ``totals[i, i] == 0`` so the ``i == j`` term contributes 0 with no
    division-by-zero (``S[i, i] = 2 * strengths[i] > 0``).
    """
    n = wins.shape[0]
    wins = wins.copy()
    totals = wins + wins.T  # total comparisons between each pair

    if prior_tie > 0:
        # Add a symmetric fictitious tie to every actually-compared pair.
        compared = totals > 0
        wins = wins + prior_tie * compared
        totals = totals + 2.0 * prior_tie * compared

    n_played = totals.sum(axis=1)
    played = n_played > 0

    # ``numerator`` is fixed across iterations; precompute once.
    numerator = wins.sum(axis=1)

    strengths = np.ones(n, dtype=np.float64)
    for _ in range(max_iter):
        pair_sum = strengths[:, None] + strengths[None, :]
        denom = (totals / pair_sum).sum(axis=1)

        new_strengths = strengths.copy()
        # Update only systems that actually played; others keep their value.
        valid = played & (denom > 0) & (numerator > 0)
        new_strengths[valid] = numerator[valid] / denom[valid]
        # Played systems with no wins at all keep a tiny positive strength
        # (the prior normally prevents this, but stay defensive).
        no_wins = played & ~(numerator > 0)
        new_strengths[no_wins] = 1e-6

        # Normalize to avoid drift (Bradley-Terry strengths are scale-free).
        mean = new_strengths.mean()
        if mean > 0:
            new_strengths = new_strengths / mean

        # Early-stop on max relative (and absolute) change.
        delta = np.abs(new_strengths - strengths)
        rel = delta / np.maximum(strengths, 1e-12)
        converged = np.max(np.minimum(delta, rel)) < tol
        strengths = new_strengths
        if converged:
            break
    return strengths


def fit_bradley_terry(
    prefs: list[PairwisePreference],
    *,
    axis: str | None = None,
    n_bootstrap: int = 1000,
    rng: Generator | None = None,
) -> BTResult:
    """Fit a Bradley-Terry model over ``prefs`` (optionally filtered to ``axis``).

    Returns strengths for every system that appears as ``system_a`` or
    ``system_b`` in at least one record, plus bootstrap 95% CIs computed by
    resampling the comparison records with replacement.
    """
    if axis is not None:
        prefs = [p for p in prefs if p.axis == axis]

    systems_set: set[str] = set()
    for p in prefs:
        systems_set.add(p.system_a)
        systems_set.add(p.system_b)
    systems = tuple(sorted(systems_set))

    if not systems:
        return BTResult(
            systems=(), strengths=(), ci_low=(), ci_high=(), n_comparisons=0
        )

    wins = _wins_matrix(prefs, systems)
    point = _fit_strengths(wins)

    if rng is None:
        rng = default_rng(seed=0)

    n = len(systems)
    boot_strengths = np.empty((n_bootstrap, n), dtype=np.float64)
    m = len(prefs)
    if m == 0:
        boot_strengths[:] = point
    else:
        for b in range(n_bootstrap):
            sample_idx = rng.integers(0, m, size=m)
            sample = [prefs[i] for i in sample_idx]
            boot_wins = _wins_matrix(sample, systems)
            boot_strengths[b] = _fit_strengths(boot_wins)

    ci_low = np.percentile(boot_strengths, 2.5, axis=0)
    ci_high = np.percentile(boot_strengths, 97.5, axis=0)

    return BTResult(
        systems=systems,
        strengths=tuple(float(x) for x in point),
        ci_low=tuple(float(x) for x in ci_low),
        ci_high=tuple(float(x) for x in ci_high),
        n_comparisons=len(prefs),
    )


def fit_per_axis(
    prefs: list[PairwisePreference],
    *,
    n_bootstrap: int = 1000,
    rng: Generator | None = None,
) -> dict[str, BTResult]:
    """Fit a separate Bradley-Terry model for every axis present in ``prefs``."""
    axes: set[str] = {p.axis for p in prefs}
    results: dict[str, BTResult] = {}
    for axis in sorted(axes):
        results[axis] = fit_bradley_terry(
            prefs, axis=axis, n_bootstrap=n_bootstrap, rng=rng
        )
    return results


def per_case_strengths(
    prefs: list[PairwisePreference],
    case_id: str,
    *,
    axis: str = "overall",
    n_bootstrap: int = 200,
    rng: Generator | None = None,
) -> BTResult:
    """Fit Bradley-Terry strengths using only records for a single ``case_id``.

    Used by the M3 gate evaluator (CORPUS.md §4.4), which compares per-case
    Astel strength against TRELLIS.2 and Meshy-free strengths.
    """
    case_prefs = [p for p in prefs if p.case_id == case_id and p.axis == axis]
    return fit_bradley_terry(case_prefs, n_bootstrap=n_bootstrap, rng=rng)


__all__ = [
    "BTResult",
    "fit_bradley_terry",
    "fit_per_axis",
    "per_case_strengths",
]
