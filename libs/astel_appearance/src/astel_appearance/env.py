"""SH-L2 environment lighting representation + a few studio presets.

An :class:`EnvironmentSH` is 9 RGB SH radiance coefficients (the same basis as
:mod:`astel_appearance.sh`). It is what the L4 layer stores as ``l4-env.json``
and what the Relight Studio swaps/rotates to relight an asset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .sh import N_SH_L2, sh_eval_l2

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class EnvironmentSH:
    """RGB SH-L2 radiance environment (``(9, 3)`` coefficients)."""

    sh_rgb: FloatArray  # (9, 3)
    name: str = "environment"

    def __post_init__(self) -> None:
        arr = np.asarray(self.sh_rgb, dtype=np.float64)
        if arr.shape != (N_SH_L2, 3):
            raise ValueError("sh_rgb must be (9, 3)")
        object.__setattr__(self, "sh_rgb", arr)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "astel.l4-env/v0",
            "name": self.name,
            "basis": "real-sh-l2",
            "order": "[Y00,Y1-1,Y10,Y11,Y2-2,Y2-1,Y20,Y21,Y22]",
            "convention": "radiance coeffs; shading E(n)/pi = sum (A_l/pi) L Y",
            "sh_rgb": self.sh_rgb.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EnvironmentSH:
        return cls(
            sh_rgb=np.asarray(data["sh_rgb"], dtype=np.float64),
            name=str(data.get("name", "environment")),
        )


def directional_env(
    direction: ArrayLike,
    color: ArrayLike,
    *,
    ambient: ArrayLike | float = 0.25,
    name: str = "directional",
) -> EnvironmentSH:
    """Build an SH env = ambient term + a single soft key light.

    ``direction`` points *toward* the light. The DC term is set so the ambient
    diffuse shading equals ``ambient``; the key light is projected onto SH as a
    directional radiance lobe of the given ``color``.
    """
    d = np.asarray(direction, dtype=np.float64).reshape(1, 3)
    y = sh_eval_l2(d)[0]  # (9,)
    col = np.asarray(color, dtype=np.float64).reshape(3)
    amb = np.broadcast_to(np.asarray(ambient, dtype=np.float64), (3,))

    sh_rgb = np.zeros((N_SH_L2, 3), dtype=np.float64)
    # DC: diffuse_shading DC term is COSINE_CONV[0]*Y00*L0 = 1*0.282095*L0,
    # so L0 = ambient / 0.282095 gives a flat ambient shading of `ambient`.
    sh_rgb[0, :] = amb / y[0]
    # Key light: add the directional lobe scaled by colour.
    sh_rgb += np.outer(y, col)
    return EnvironmentSH(sh_rgb=sh_rgb, name=name)


def studio_presets() -> dict[str, EnvironmentSH]:
    """A small set of named relight environments for the Relight Studio."""
    return {
        "studio": directional_env(
            [0.4, 0.8, 0.6], [1.1, 1.05, 1.0], ambient=0.35, name="studio"
        ),
        "noon": directional_env(
            [0.0, 1.0, 0.1], [1.3, 1.28, 1.2], ambient=0.45, name="noon"
        ),
        "sunset": directional_env(
            [0.9, 0.25, 0.3], [1.4, 0.7, 0.4], ambient=0.18, name="sunset"
        ),
        "rim": directional_env(
            [-0.6, 0.3, -0.7], [0.9, 0.95, 1.2], ambient=0.12, name="rim"
        ),
    }
