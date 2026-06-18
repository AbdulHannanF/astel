"""Producer-facing L4 helper: raw splat arrays -> appearance artifacts.

Both the CPU stub and the GPU producer need the *same* L4 decomposition but
must stay decoupled from each other (the API must not import the torch-bearing
GPU package). This module works only on raw numpy arrays + JSON-able dicts, so
each producer keeps ownership of file I/O and ``.astel`` binding while sharing
the appearance math here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .decompose import AppearanceLayer, albedo_colors_dc, decompose_appearance
from .normals import surfel_normals
from .webdata import relight_payload

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class AppearanceArtifacts:
    """Everything a producer needs to write + bind the L4 layer."""

    layer: AppearanceLayer
    albedo_colors_dc: FloatArray  # (N, 3) DC coeffs for the albedo splat cloud
    env: dict[str, Any]  # l4-env.json content
    summary: dict[str, Any]  # l4.json content (also folded into the report)
    relight_preview: dict[str, Any]  # l4-relight.json content (web studio)


def build_appearance(
    positions: NDArray[np.floating],
    colors_dc: NDArray[np.floating],
    quats: NDArray[np.floating],
    log_scales: NDArray[np.floating],
    opacity_logit: NDArray[np.floating],
    *,
    max_preview: int = 6000,
    seed: int = 0,
) -> AppearanceArtifacts:
    """Decompose a splat cloud's baked colour into the L4 appearance artifacts."""
    normals = surfel_normals(positions, quats, log_scales)
    layer = decompose_appearance(colors_dc, normals, opacity_logit=opacity_logit)
    return AppearanceArtifacts(
        layer=layer,
        albedo_colors_dc=albedo_colors_dc(layer),
        env=layer.env.to_dict(),
        summary=layer.summary(),
        relight_preview=relight_payload(
            layer, np.asarray(positions, dtype=np.float64),
            max_points=max_preview, seed=seed,
        ),
    )
