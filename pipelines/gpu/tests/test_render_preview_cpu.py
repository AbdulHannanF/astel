"""CPU tests for the pure turntable-camera seam (no gsplat/CUDA)."""

from __future__ import annotations

import math

import torch

from astel_gpu.render_preview import turntable_centres


def test_centres_lie_on_sphere_of_radius() -> None:
    centres = turntable_centres(8, radius=3.0, elevation_deg=15.0)
    dists = centres.norm(dim=-1)
    assert torch.allclose(dists, torch.full((8,), 3.0), atol=1e-4)


def test_elevation_sets_up_axis_height() -> None:
    # +Z is the up axis; every camera sits at radius*sin(elev) in Z.
    centres = turntable_centres(6, radius=2.0, elevation_deg=30.0)
    expected_z = 2.0 * math.sin(math.radians(30.0))
    assert torch.allclose(centres[:, 2], torch.full((6,), expected_z), atol=1e-5)


def test_azimuth_spans_the_orbit() -> None:
    # First camera starts on the +X side of the up axis; distinct azimuths.
    centres = turntable_centres(4, radius=1.0, elevation_deg=0.0)
    # 4 evenly-spaced azimuths in the XY plane: +X, +Y, -X, -Y.
    assert torch.allclose(centres[0], torch.tensor([1.0, 0.0, 0.0]), atol=1e-6)
    assert torch.allclose(centres[1], torch.tensor([0.0, 1.0, 0.0]), atol=1e-6)
    assert torch.allclose(centres[2], torch.tensor([-1.0, 0.0, 0.0]), atol=1e-6)


def test_count_matches_request() -> None:
    assert turntable_centres(12, 3.0, 15.0).shape == (12, 3)
