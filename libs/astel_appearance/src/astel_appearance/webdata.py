"""Compact JSON payloads the web studios consume (Relight + Physics).

The full L4 albedo/normals live in the ``.astel`` package and the loose
``l4-albedo.ply``; the browser studios only need a *representative downsample*
to relight interactively or to outline a collision proxy. These helpers emit
small, self-describing JSON the front-end can fetch without a splat decoder.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .decompose import AppearanceLayer


def relight_payload(
    layer: AppearanceLayer,
    positions: NDArray[np.floating],
    *,
    max_points: int = 6000,
    seed: int = 0,
) -> dict[str, Any]:
    """A downsampled {position, normal, albedo} payload for the Relight Studio.

    The browser re-shades these points under a live SH environment (the same
    math as :mod:`astel_appearance.sh`), proving the albedo/lighting split. The
    sample is a deterministic uniform random subset (honest preview, not the
    full asset).
    """
    pos = np.asarray(positions, dtype=np.float64)
    n = pos.shape[0]
    if n > max_points:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n, size=max_points, replace=False))
    else:
        idx = np.arange(n)

    p = pos[idx]
    centre = p.mean(axis=0)
    radius = float(np.linalg.norm(p - centre, axis=1).max()) or 1.0

    return {
        "schema": "astel.l4-relight-preview/v0",
        "count": int(idx.size),
        "total": int(n),
        "downsampled": bool(idx.size < n),
        "lighting_confidence": float(layer.lighting_confidence),
        "center": [float(x) for x in centre],
        "radius": radius,
        "env_estimated": layer.env.to_dict(),
        "positions": np.round(p, 5).tolist(),
        "normals": np.round(layer.normals[idx], 4).tolist(),
        "albedo": np.round(layer.albedo[idx], 4).tolist(),
        "notes": layer.notes,
    }
