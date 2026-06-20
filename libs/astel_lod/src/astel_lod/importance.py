"""Per-Gaussian importance scoring for LOD tier selection.

The importance of a Gaussian is estimated as:

    importance(i) = opacity(i) * projected_footprint(i)

where::

    projected_footprint(i) = exp(log_scale_x(i))
                            * exp(log_scale_y(i))
                            * exp(log_scale_z(i))
                           = exp(sum(log_scales[i]))

**Formula rationale (honest perceptual proxy, not optimal saliency):**

* *Opacity* gates whether the Gaussian contributes visibly at all.  Gaussians
  with near-zero opacity are invisible regardless of size and should always be
  culled first.
* *Projected footprint* (product of the three world-space semi-axis lengths)
  approximates the screen-space area a Gaussian occupies at a canonical viewing
  distance.  A larger Gaussian covers more pixels and therefore carries more
  visual weight when the budget is limited.

The score is monotone in opacity (for fixed scales) and monotone in each
log-scale component (for fixed opacity and other scales).  It does **not**
account for viewing angle, occlusion, depth, or scene-level saliency — those
require per-frame information unavailable at asset-build time.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def splat_importance(
    opacity: NDArray[np.floating],
    log_scales: NDArray[np.floating],
) -> NDArray[np.float64]:
    """Compute per-Gaussian importance scores.

    Parameters
    ----------
    opacity:
        Shape ``(N,)``.  Values in ``[0, 1]`` (or the raw pre-sigmoid
        activations — any finite float; the formula is monotone in this
        argument).
    log_scales:
        Shape ``(N, 3)``.  Log of the three world-space semi-axis lengths
        (the standard 3DGS / 2DGS parameterisation).

    Returns
    -------
    NDArray[np.float64]
        Shape ``(N,)``, dtype float64.  ``importance[i] = opacity[i] *
        exp(log_scales[i, 0] + log_scales[i, 1] + log_scales[i, 2])``.
        Always finite for finite inputs.
    """
    op = np.asarray(opacity, dtype=np.float64)  # (N,)
    ls = np.asarray(log_scales, dtype=np.float64)  # (N, 3)

    # Sum log-scales per Gaussian then exponentiate → product of semi-axes.
    log_footprint: NDArray[np.float64] = ls.sum(axis=1)
    footprint: NDArray[np.float64] = np.exp(log_footprint)

    return op * footprint
