"""EnvironmentSH presets/serialization + web relight payload."""

from __future__ import annotations

import numpy as np

from astel_appearance.decompose import (
    colors_dc_from_rgb,
    decompose_appearance,
)
from astel_appearance.env import EnvironmentSH, directional_env, studio_presets
from astel_appearance.sh import diffuse_shading
from astel_appearance.webdata import relight_payload


def test_env_roundtrip() -> None:
    env = directional_env([0.0, 1.0, 0.0], [1.0, 0.9, 0.8], ambient=0.3)
    back = EnvironmentSH.from_dict(env.to_dict())
    assert np.allclose(back.sh_rgb, env.sh_rgb)
    assert back.name == env.name


def test_directional_env_ambient_shading() -> None:
    # With zero key-light colour, shading == ambient everywhere (flat).
    env = directional_env([0, 1, 0], [0.0, 0.0, 0.0], ambient=0.4)
    dirs = np.array([[0, 0, 1.0], [1, 0, 0.0], [0, 1, 0.0]])
    shading = diffuse_shading(env.sh_rgb, dirs)
    assert np.allclose(shading, 0.4, atol=1e-9)


def test_directional_env_brighter_toward_light() -> None:
    env = directional_env([0, 0, 1.0], [1.0, 1.0, 1.0], ambient=0.2)
    toward = diffuse_shading(env.sh_rgb, np.array([[0, 0, 1.0]]))[0]
    away = diffuse_shading(env.sh_rgb, np.array([[0, 0, -1.0]]))[0]
    assert np.all(toward > away)


def test_presets_present() -> None:
    presets = studio_presets()
    assert {"studio", "noon", "sunset", "rim"} <= set(presets)
    for env in presets.values():
        assert env.sh_rgb.shape == (9, 3)


def test_relight_payload_shape_and_downsample() -> None:
    rng = np.random.default_rng(0)
    n = 10_000
    normals = rng.standard_normal((n, 3))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    positions = rng.standard_normal((n, 3))
    observed = np.clip(rng.uniform(0.2, 0.6, (n, 3)), 0, 1)
    layer = decompose_appearance(colors_dc_from_rgb(observed), normals)

    payload = relight_payload(layer, positions, max_points=2000, seed=0)
    assert payload["count"] == 2000
    assert payload["total"] == n
    assert payload["downsampled"] is True
    assert len(payload["positions"]) == 2000
    assert len(payload["normals"]) == 2000
    assert len(payload["albedo"]) == 2000
    assert payload["env_estimated"]["schema"] == "astel.l4-env/v0"

    # Deterministic for a fixed seed.
    again = relight_payload(layer, positions, max_points=2000, seed=0)
    assert again["positions"] == payload["positions"]
