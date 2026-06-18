"""Surfel normals + the producer-facing build_appearance integration."""

from __future__ import annotations

import numpy as np

from astel_appearance.normals import surfel_normals
from astel_appearance.produce import build_appearance


def test_surfel_normal_is_thin_axis() -> None:
    # Identity orientation, thin axis = z (smallest log-scale) -> normal ~ ±z.
    n = 50
    positions = np.zeros((n, 3))
    positions[:, 2] = np.linspace(-1, 1, n)  # spread along z so outward flips
    quats = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1))
    log_scales = np.tile([np.log(0.1), np.log(0.1), np.log(0.01)], (n, 1))
    normals = surfel_normals(positions, quats, log_scales)
    assert np.allclose(np.abs(normals[:, 2]), 1.0, atol=1e-6)
    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-6)


def test_build_appearance_shapes_and_roundtrip() -> None:
    rng = np.random.default_rng(0)
    n = 2000
    positions = rng.standard_normal((n, 3))
    quats = rng.standard_normal((n, 4))
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    log_scales = np.log(rng.uniform(0.01, 0.1, (n, 3)))
    opacity = rng.uniform(2.0, 5.0, n)  # high logits -> ~opaque
    colors_dc = rng.uniform(-1.0, 1.0, (n, 3))

    art = build_appearance(
        positions, colors_dc, quats, log_scales, opacity, max_preview=500
    )
    assert art.albedo_colors_dc.shape == (n, 3)
    assert art.layer.count == n
    assert art.env["schema"] == "astel.l4-env/v0"
    assert art.summary["schema"] == "astel.l4-appearance/v0"
    assert art.relight_preview["count"] == 500
    assert 0.0 <= art.summary["lighting_confidence"] <= 1.0
