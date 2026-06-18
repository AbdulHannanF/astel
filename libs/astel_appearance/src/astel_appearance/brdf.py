"""Cook-Torrance microfacet BRDF (GGX) — the per-splat PBR forward model.

This is the analytic shading model the L4 ``{albedo, metallic, roughness}``
parameters feed, used to:

* render a physically-based preview for engines that consume colored splats
  (the "PBR approximation" export, CLAUDE.md §3 L4), and
* provide a specular term for the Relight Studio beyond pure Lambertian diffuse.

Plain numpy, vectorised over ``(N, ...)`` splat batches. The metallic workflow
follows the glTF 2.0 / Disney convention: dielectric F0 = 0.04, metals tint the
specular by the base colour and have no diffuse.
"""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

#: Default dielectric (non-metal) Fresnel reflectance at normal incidence.
DIELECTRIC_F0 = 0.04


def _dot(a: NDArray[np.floating], b: NDArray[np.floating]) -> FloatArray:
    prod = np.asarray(a, dtype=np.float64) * np.asarray(b, dtype=np.float64)
    return cast(FloatArray, np.clip(np.sum(prod, axis=-1), 0.0, 1.0))


def fresnel_schlick(
    cos_theta: NDArray[np.floating], f0: NDArray[np.floating]
) -> FloatArray:
    """Schlick's Fresnel approximation ``F0 + (1-F0)(1-cosθ)^5``."""
    c = np.clip(np.asarray(cos_theta, dtype=np.float64), 0.0, 1.0)
    f = np.asarray(f0, dtype=np.float64)
    return f + (1.0 - f) * (1.0 - c) ** 5


def ggx_ndf(
    n_dot_h: NDArray[np.floating], roughness: NDArray[np.floating]
) -> FloatArray:
    """Trowbridge-Reitz (GGX) normal distribution function ``D``.

    Uses ``α = roughness²`` (the perceptual-roughness remap). Normalised so that
    ``∫ D (n·h) dω_h = 1`` over the hemisphere.
    """
    a = np.clip(np.asarray(roughness, dtype=np.float64), 1e-3, 1.0) ** 2
    a2 = a * a
    ndh = np.clip(np.asarray(n_dot_h, dtype=np.float64), 0.0, 1.0)
    denom = ndh * ndh * (a2 - 1.0) + 1.0
    return a2 / (np.pi * denom * denom)


def smith_g(
    n_dot_v: NDArray[np.floating],
    n_dot_l: NDArray[np.floating],
    roughness: NDArray[np.floating],
) -> FloatArray:
    """Smith geometry term with the Schlick-GGX height-correlated remap (k=α/2)."""
    a = np.clip(np.asarray(roughness, dtype=np.float64), 1e-3, 1.0) ** 2
    k = a / 2.0

    def g1(c: FloatArray) -> FloatArray:
        c = np.clip(c, 1e-6, 1.0)
        return c / (c * (1.0 - k) + k)

    return g1(np.asarray(n_dot_v, dtype=np.float64)) * g1(
        np.asarray(n_dot_l, dtype=np.float64)
    )


def cook_torrance(
    albedo: NDArray[np.floating],
    metallic: NDArray[np.floating],
    roughness: NDArray[np.floating],
    normal: NDArray[np.floating],
    view_dir: NDArray[np.floating],
    light_dir: NDArray[np.floating],
    light_color: NDArray[np.floating],
) -> FloatArray:
    """Outgoing radiance of a Cook-Torrance surface for one directional light.

    All direction arrays are ``(N, 3)`` (``view_dir``/``light_dir`` point *away*
    from the surface, toward camera/light). ``albedo``/``light_color`` are
    ``(N, 3)``; ``metallic``/``roughness`` are ``(N,)``. Returns ``(N, 3)``
    reflected radiance ``(diffuse + specular) * (n·l) * light_color``.
    """
    n = np.asarray(normal, dtype=np.float64)
    v = np.asarray(view_dir, dtype=np.float64)
    light = np.asarray(light_dir, dtype=np.float64)
    h = v + light
    h = h / np.clip(np.linalg.norm(h, axis=-1, keepdims=True), 1e-9, None)

    ndl = _dot(n, light)[:, None]
    ndv = _dot(n, v)[:, None]
    ndh = _dot(n, h)
    vdh = _dot(v, h)

    base = np.asarray(albedo, dtype=np.float64)
    met = np.asarray(metallic, dtype=np.float64)[:, None]
    f0 = (1.0 - met) * DIELECTRIC_F0 + met * base  # (N, 3)

    fresnel = fresnel_schlick(vdh[:, None], f0)  # (N, 3)
    d = ggx_ndf(ndh, roughness)[:, None]
    g = smith_g(ndv[:, 0], ndl[:, 0], roughness)[:, None]

    spec = (d * g * fresnel) / np.clip(4.0 * ndv * ndl, 1e-6, None)

    k_d = (1.0 - fresnel) * (1.0 - met)  # metals have no diffuse
    diffuse = k_d * base / np.pi

    radiance = (diffuse + spec) * ndl * np.asarray(light_color, dtype=np.float64)
    return np.clip(radiance, 0.0, None)
