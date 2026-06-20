"""Test-helper: build synthetic ObjectSplats filling an axis-aligned box."""

from __future__ import annotations

import numpy as np

from astel_scene.splats import ObjectSplats


def box_object(
    n: int,
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    size: tuple[float, float, float] = (1.0, 1.0, 1.0),
    seed: int = 0,
) -> ObjectSplats:
    """Return *n* gaussians uniformly distributed inside an axis-aligned box.

    Parameters
    ----------
    n:
        Number of splats.
    center:
        Box centre ``(cx, cy, cz)``.
    size:
        Box full extents ``(sx, sy, sz)``; each splat's coordinate is
        uniformly sampled from ``[center − size/2, center + size/2]``.
    seed:
        RNG seed for reproducibility.

    Gaussian field defaults
    -----------------------
    * **quats** — identity ``(1, 0, 0, 0)`` for every splat.
    * **log_scales** — zeros (unit scale).
    * **opacity** — ones.
    * **colors_dc** — uniform gray ``(0.5, 0.5, 0.5)`` (SH DC for mid-grey).
    """
    rng = np.random.default_rng(seed)
    c = np.array(center, dtype=np.float32)
    s = np.array(size, dtype=np.float32)

    lo = c - s * 0.5
    hi = c + s * 0.5
    positions = rng.uniform(lo, hi, size=(n, 3)).astype(np.float32)

    quats = np.zeros((n, 4), dtype=np.float32)
    quats[:, 0] = 1.0  # identity: w=1, x=y=z=0

    log_scales = np.zeros((n, 3), dtype=np.float32)
    opacity = np.ones(n, dtype=np.float32)
    colors_dc = np.full((n, 3), 0.5, dtype=np.float32)

    return ObjectSplats(
        positions=positions,
        quats=quats,
        log_scales=log_scales,
        opacity=opacity,
        colors_dc=colors_dc,
    )
