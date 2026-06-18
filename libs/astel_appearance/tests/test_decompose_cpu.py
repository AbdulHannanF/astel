"""L4 intrinsic decomposition: the relight round-trip invariant + recovery."""

from __future__ import annotations

from typing import cast

import numpy as np

from astel_appearance.decompose import (
    colors_dc_from_rgb,
    decompose_appearance,
    observed_rgb_from_dc,
    relight_rgb,
)
from astel_appearance.env import directional_env
from astel_appearance.sh import diffuse_shading


def _random_normals(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, 3))
    return cast(np.ndarray, v / np.linalg.norm(v, axis=1, keepdims=True))


def test_dc_rgb_roundtrip() -> None:
    rng = np.random.default_rng(0)
    rgb = rng.uniform(0.0, 1.0, (100, 3))
    assert np.allclose(observed_rgb_from_dc(colors_dc_from_rgb(rgb)), rgb, atol=1e-6)


def test_relight_invariant_reproduces_observed() -> None:
    # The core honesty guarantee: relighting under the *estimated* env returns
    # the captured colour, wherever albedo isn't clipped and shading > eps.
    n = 3000
    normals = _random_normals(n, 1)
    env_true = directional_env([0.3, 0.7, 0.5], [0.9, 0.9, 0.9], ambient=0.6)
    rng = np.random.default_rng(2)
    albedo_true = rng.uniform(0.2, 0.5, (n, 3))
    shading = diffuse_shading(env_true.sh_rgb, normals)  # (n, 3)
    observed = np.clip(albedo_true * shading, 0.0, 1.0)

    layer = decompose_appearance(colors_dc_from_rgb(observed), normals)
    relit = relight_rgb(layer, layer.env)

    inside = np.all((layer.albedo > 1e-4) & (layer.albedo < 1.0 - 1e-4), axis=1)
    assert inside.mean() > 0.9
    assert np.allclose(relit[inside], observed[inside], atol=1e-3)


def test_uniform_albedo_recovered_high_confidence() -> None:
    # Constant albedo lit by an SH-L2 env: luminance is fully explained by the
    # lighting model, so the fit recovers ~constant albedo at high confidence.
    n = 5000
    normals = _random_normals(n, 5)
    albedo_true = np.array([0.45, 0.32, 0.55])
    env_true = directional_env([0.2, 0.9, 0.3], [0.8, 0.8, 0.8], ambient=0.55)
    shading = diffuse_shading(env_true.sh_rgb, normals)
    observed = np.clip(albedo_true[None, :] * shading, 0.0, 1.0)

    layer = decompose_appearance(colors_dc_from_rgb(observed), normals)
    assert layer.lighting_confidence > 0.95
    # Recovered albedo should be near-constant (low spatial variance).
    assert np.all(layer.albedo.std(axis=0) < 0.02)


def test_relight_changes_appearance() -> None:
    n = 2000
    normals = _random_normals(n, 9)
    env_true = directional_env([0.4, 0.8, 0.2], [0.9, 0.9, 0.9], ambient=0.5)
    observed = np.clip(0.4 * diffuse_shading(env_true.sh_rgb, normals), 0, 1)
    layer = decompose_appearance(colors_dc_from_rgb(observed), normals)

    sunset = directional_env([0.9, 0.2, 0.3], [1.4, 0.6, 0.3], ambient=0.2)
    relit = relight_rgb(layer, sunset)
    # A different environment must produce a different image.
    assert np.mean(np.abs(relit - observed)) > 0.02


def test_degenerate_black_input_is_noop() -> None:
    n = 500
    normals = _random_normals(n, 3)
    observed = np.zeros((n, 3))
    layer = decompose_appearance(colors_dc_from_rgb(observed), normals)
    assert np.all(np.isfinite(layer.albedo))
    assert any("degenerate" in note.lower() for note in layer.notes)


def test_priors_emitted_and_flagged() -> None:
    n = 100
    normals = _random_normals(n, 4)
    observed = np.full((n, 3), 0.5)
    layer = decompose_appearance(colors_dc_from_rgb(observed), normals)
    assert np.all(layer.metallic == 0.0)
    assert np.all(layer.roughness == 0.6)
    summary = layer.summary()
    assert "prior" in summary["metallic"]
    assert summary["count"] == n
