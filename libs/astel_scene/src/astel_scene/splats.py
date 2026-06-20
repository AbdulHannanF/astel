"""ObjectSplats — frozen dataclass holding raw Gaussian-field arrays.

The Astel internal quaternion convention is **(w, x, y, z)** throughout.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ObjectSplats:
    """Raw Gaussian-field arrays for a single object.

    All arrays share the same leading dimension N (number of splats).

    Attributes
    ----------
    positions:
        (N, 3) float32 — Gaussian means.
    quats:
        (N, 4) float32 — rotation quaternions in **(w, x, y, z)** order.
    log_scales:
        (N, 3) float32 — natural log of per-axis scale.
    opacity:
        (N,) float32 — opacity values.
    colors_dc:
        (N, 3) float32 — SH band-0 DC colour coefficients.
    """

    positions: NDArray[np.float32]
    quats: NDArray[np.float32]
    log_scales: NDArray[np.float32]
    opacity: NDArray[np.float32]
    colors_dc: NDArray[np.float32]

    def __post_init__(self) -> None:
        n = self.positions.shape[0]
        if self.positions.ndim != 2 or self.positions.shape[1] != 3:
            raise ValueError(f"positions must be (N, 3), got {self.positions.shape}")
        if self.quats.shape != (n, 4):
            raise ValueError(f"quats must be ({n}, 4), got {self.quats.shape}")
        if self.log_scales.shape != (n, 3):
            raise ValueError(
                f"log_scales must be ({n}, 3), got {self.log_scales.shape}"
            )
        if self.opacity.shape != (n,):
            raise ValueError(f"opacity must be ({n},), got {self.opacity.shape}")
        if self.colors_dc.shape != (n, 3):
            raise ValueError(f"colors_dc must be ({n}, 3), got {self.colors_dc.shape}")

    @property
    def count(self) -> int:
        """Number of splats."""
        return int(self.positions.shape[0])
