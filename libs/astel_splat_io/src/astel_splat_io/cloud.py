"""The shared in-memory splat representation.

Mirrors ``pipelines/stub/make_sample_splat.py``'s ``SplatCloud`` exactly so
that the stub's procedural sample (and any future pipeline stage) can be fed
directly into these writers without conversion.

All arrays use the raw 3DGS parameterisation (log-scale, opacity logit, SH
band-0 DC colour), not display-space values:

- ``positions``: (N, 3) world-space xyz, float32.
- ``colors_dc``: (N, 3) SH band-0 DC coefficients, float32. ``albedo = 0.5 +
  SH_C0 * f_dc``.
- ``opacity``: (N,) logit, float32. ``alpha = sigmoid(opacity)``.
- ``log_scales``: (N, 3) log of world-space sigma, float32.
- ``quats``: (N, 4) quaternion in (w, x, y, z) order, normalised, float32.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

# SH band-0 (DC) basis constant. albedo = 0.5 + C0 * f_dc, so f_dc = (albedo-0.5)/C0.
SH_C0: float = 0.28209479177387814


@dataclass(frozen=True)
class SplatCloud:
    """A bundle of per-splat attributes in the raw 3DGS parameterisation.

    All arrays share the same leading dimension ``N`` (the splat count).
    """

    positions: NDArray[np.float32]  # (N, 3) world-space xyz
    colors_dc: NDArray[np.float32]  # (N, 3) SH band-0 DC coefficients
    opacity: NDArray[np.float32]  # (N,)   logit
    log_scales: NDArray[np.float32]  # (N, 3) log of world-space sigma
    quats: NDArray[np.float32]  # (N, 4) (w, x, y, z) normalised

    def __post_init__(self) -> None:
        n = self.positions.shape[0]
        if self.positions.shape != (n, 3):
            raise ValueError("positions must be (N, 3)")
        if self.colors_dc.shape != (n, 3):
            raise ValueError("colors_dc must be (N, 3)")
        if self.opacity.shape != (n,):
            raise ValueError("opacity must be (N,)")
        if self.log_scales.shape != (n, 3):
            raise ValueError("log_scales must be (N, 3)")
        if self.quats.shape != (n, 4):
            raise ValueError("quats must be (N, 4)")

    @property
    def count(self) -> int:
        return int(self.positions.shape[0])

    def reordered(self, order: NDArray[np.intp]) -> SplatCloud:
        """Return a copy of this cloud with all per-splat arrays permuted by ``order``.

        ``order`` is a length-N index array such that ``result[i] ==
        self[order[i]]``. Used by exporters (e.g. SPZ/SOG spatial sorts) that
        reorder splats; callers MUST apply the same permutation to any bound
        provenance buffer (manifest-v0 section 5.4).
        """
        return SplatCloud(
            positions=self.positions[order],
            colors_dc=self.colors_dc[order],
            opacity=self.opacity[order],
            log_scales=self.log_scales[order],
            quats=self.quats[order],
        )
