"""Real spherical harmonics (band 0-2) and irradiance, the L4 lighting basis.

Conventions follow Ramamoorthi & Hanrahan, "An Efficient Representation for
Irradiance Environment Maps" (SIGGRAPH 2001):

* :func:`sh_eval_l2` returns the 9 real SH basis values ``Y_l^m(n)`` for unit
  directions ``n``, in the order
  ``[Y00, Y1-1, Y10, Y11, Y2-2, Y2-1, Y20, Y21, Y22]``.
* A *radiance* environment is represented by 9 SH coefficients ``L`` (per
  channel). The cosine-convolved **irradiance** of a Lambertian surface with
  normal ``n`` is ``E(n) = Σ A_l L_l^m Y_l^m(n)`` with ``A_0 = π``,
  ``A_1 = 2π/3``, ``A_2 = π/4``.
* The Lambertian **diffuse shading factor** (exit radiance per unit albedo) is
  ``E(n)/π``; this module computes that directly via :func:`diffuse_shading`,
  using the folded constants ``Â_l = A_l/π = {1, 2/3, 1/4}`` so that the
  reflected colour of a surface is simply ``albedo * diffuse_shading``.

Everything is plain numpy (no torch); the same constants are mirrored in the
web ``apps/web/src/lib/sh.ts`` port (parity-tested).
"""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

# --- Real SH basis normalisation constants (l = 0, 1, 2) ---
_C0 = 0.28209479177387814  # 0.5 * sqrt(1/pi)
_C1 = 0.4886025119029199  # 0.5 * sqrt(3/pi)
_C2 = 1.0925484305920792  # 0.5 * sqrt(15/pi)
_C3 = 0.31539156525252005  # 0.25 * sqrt(5/pi)
_C4 = 0.5462742152960396  # 0.25 * sqrt(15/pi)

#: Folded cosine-convolution constants Â_l = A_l / π for the diffuse shading
#: factor E(n)/π, indexed by SH coefficient (0..8): l=0 -> 1, l=1 -> 2/3,
#: l=2 -> 1/4.
COSINE_CONV: FloatArray = np.array(
    [1.0, 2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 0.25, 0.25, 0.25, 0.25, 0.25],
    dtype=np.float64,
)

N_SH_L2 = 9


def _as_unit(dirs: NDArray[np.floating]) -> FloatArray:
    d = np.asarray(dirs, dtype=np.float64)
    if d.ndim != 2 or d.shape[1] != 3:
        raise ValueError("dirs must be (N, 3)")
    norm = np.linalg.norm(d, axis=1, keepdims=True)
    norm = np.where(norm == 0.0, 1.0, norm)
    return d / norm


def sh_eval_l2(dirs: NDArray[np.floating]) -> FloatArray:
    """Evaluate the 9 real SH basis functions for ``(N, 3)`` directions.

    Returns ``(N, 9)`` in order ``[Y00, Y1-1, Y10, Y11, Y2-2, Y2-1, Y20,
    Y21, Y22]``. Inputs are normalised defensively.
    """
    n = _as_unit(dirs)
    x, y, z = n[:, 0], n[:, 1], n[:, 2]
    out = np.empty((n.shape[0], N_SH_L2), dtype=np.float64)
    out[:, 0] = _C0
    out[:, 1] = -_C1 * y
    out[:, 2] = _C1 * z
    out[:, 3] = -_C1 * x
    out[:, 4] = _C2 * x * y
    out[:, 5] = -_C2 * y * z
    out[:, 6] = _C3 * (3.0 * z * z - 1.0)
    out[:, 7] = -_C2 * x * z
    out[:, 8] = _C4 * (x * x - y * y)
    return out


def diffuse_shading(
    env_sh: NDArray[np.floating], normals: NDArray[np.floating]
) -> FloatArray:
    """Lambertian diffuse shading factor ``E(n)/π`` for an SH radiance env.

    Parameters
    ----------
    env_sh:
        ``(9,)`` (single channel) or ``(9, C)`` SH radiance coefficients.
    normals:
        ``(N, 3)`` unit surface normals.

    Returns ``(N,)`` for a single channel or ``(N, C)`` for ``C`` channels.
    The reflected colour of a Lambertian surface is ``albedo * shading``.
    """
    basis = sh_eval_l2(normals)  # (N, 9)
    weighted = basis * COSINE_CONV[None, :]  # (N, 9)
    env = np.asarray(env_sh, dtype=np.float64)
    if env.shape[0] != N_SH_L2:
        raise ValueError("env_sh leading dim must be 9 (SH-L2)")
    return weighted @ env  # (N,) or (N, C)


def fit_environment_sh(
    normals: NDArray[np.floating],
    values: NDArray[np.floating],
    *,
    weights: NDArray[np.floating] | None = None,
    ridge: float = 1e-3,
) -> FloatArray:
    """Least-squares fit of SH radiance coefficients to ``shading`` samples.

    Solves ``min_L Σ w_i (Σ Â_l Y_l(n_i) L_l - value_i)^2 + ridge·‖L‖²`` so
    that :func:`diffuse_shading(L, n) ≈ value`. ``values`` may be ``(N,)`` or
    ``(N, C)``; the return shape matches (``(9,)`` or ``(9, C)``).

    The small Tikhonov ``ridge`` keeps the normal-equation matrix invertible
    when the sampled normals do not span all SH directions (e.g. a single
    captured hemisphere). This is a *low-frequency* fit by construction (SH-L2
    only represents up to quadratic angular variation) — the basis of the
    single-observation intrinsic split in :mod:`astel_appearance.decompose`.
    """
    basis = sh_eval_l2(normals) * COSINE_CONV[None, :]  # (N, 9): the design matrix
    v = np.asarray(values, dtype=np.float64)
    single = v.ndim == 1
    if single:
        v = v[:, None]
    if weights is None:
        w = np.ones(basis.shape[0], dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
    wb = basis * w[:, None]  # (N, 9)
    ata = basis.T @ wb + ridge * np.eye(N_SH_L2)  # (9, 9)
    atb = wb.T @ v  # (9, C)
    coeffs = np.linalg.solve(ata, atb)  # (9, C)
    return cast(FloatArray, coeffs[:, 0] if single else coeffs)


def yaw_rotation(angle_rad: float) -> FloatArray:
    """A 3x3 rotation about ``+Y`` (up axis), for relight environment spin.

    Rotating the *environment* by ``R`` is equivalent to evaluating the shading
    at ``R^{-1} n``; the Relight Studio uses this to spin the HDRI live without
    an SH rotation matrix.
    """
    c, s = float(np.cos(angle_rad)), float(np.sin(angle_rad))
    return cast(
        FloatArray,
        np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64),
    )
