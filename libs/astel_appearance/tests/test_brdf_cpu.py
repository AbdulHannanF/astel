"""Cook-Torrance GGX BRDF analytic identities."""

from __future__ import annotations

import numpy as np

from astel_appearance.brdf import (
    DIELECTRIC_F0,
    cook_torrance,
    fresnel_schlick,
    ggx_ndf,
    smith_g,
)


def test_fresnel_endpoints() -> None:
    f0 = np.full((5, 3), DIELECTRIC_F0)
    # Normal incidence -> F == F0 (cosθ broadcast against the channel dim).
    assert np.allclose(fresnel_schlick(np.ones((5, 1)), f0), f0)
    # Grazing -> F == 1.
    assert np.allclose(fresnel_schlick(np.zeros((5, 1)), f0), 1.0)


def test_ggx_normalizes_over_hemisphere() -> None:
    # integral over hemisphere of D(h) (n.h) dω = 1 ; MC with uniform sampling.
    rng = np.random.default_rng(0)
    n = 200_000
    z = rng.uniform(0.0, 1.0, n)  # cos theta in [0,1] (upper hemisphere)
    ndh = z
    for rough in (0.3, 0.6, 0.9):
        d = ggx_ndf(ndh, np.full(n, rough))
        integral = (d * ndh).mean() * 2.0 * np.pi  # pdf = 1/(2pi)
        assert abs(integral - 1.0) < 0.02, (rough, integral)


def test_smith_g_in_unit_range() -> None:
    rng = np.random.default_rng(3)
    ndv = rng.uniform(0.05, 1.0, 1000)
    ndl = rng.uniform(0.05, 1.0, 1000)
    g = smith_g(ndv, ndl, np.full(1000, 0.5))
    assert np.all(g >= 0.0) and np.all(g <= 1.0)


def test_metal_has_no_diffuse_dielectric_does() -> None:
    # One splat, light + view along the normal (+z).
    normal = np.array([[0.0, 0.0, 1.0]])
    up = np.array([[0.0, 0.0, 1.0]])
    albedo = np.array([[0.8, 0.1, 0.1]])
    rough = np.array([0.9])  # rough -> weak specular, diffuse-dominated
    white = np.array([[1.0, 1.0, 1.0]])

    dielectric = cook_torrance(
        albedo, np.array([0.0]), rough, normal, up, up, white
    )
    metal = cook_torrance(
        albedo, np.array([1.0]), rough, normal, up, up, white
    )
    # Dielectric reflects its red diffuse; a rough metal reflects far less red.
    assert dielectric[0, 0] > metal[0, 0]
    assert dielectric[0, 0] > 0.1


def test_radiance_nonnegative() -> None:
    rng = np.random.default_rng(7)
    n = 500
    normals = np.tile([0.0, 0.0, 1.0], (n, 1))
    v = rng.standard_normal((n, 3))
    v[:, 2] = np.abs(v[:, 2]) + 0.1
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    light = rng.standard_normal((n, 3))
    light[:, 2] = np.abs(light[:, 2]) + 0.1
    light /= np.linalg.norm(light, axis=1, keepdims=True)
    out = cook_torrance(
        rng.uniform(0, 1, (n, 3)),
        rng.uniform(0, 1, n),
        rng.uniform(0.1, 1, n),
        normals,
        v,
        light,
        np.ones((n, 3)),
    )
    assert np.all(out >= 0.0)
    assert np.all(np.isfinite(out))
