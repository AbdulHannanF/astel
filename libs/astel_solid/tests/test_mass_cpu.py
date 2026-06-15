"""Mass properties validated against analytic solids."""

from __future__ import annotations

import math

import numpy as np
from _shapes import unit_cube

from astel_solid.mass import compute_mass_properties


def test_unit_cube_volume_com_inertia() -> None:
    mesh = unit_cube(side=1.0, center=(0, 0, 0))
    mp = compute_mass_properties(mesh, density=1.0)

    assert math.isclose(mp.volume, 1.0, rel_tol=1e-6)
    assert math.isclose(mp.mass, 1.0, rel_tol=1e-6)
    assert np.allclose(mp.center_of_mass, 0.0, atol=1e-6)

    # Solid cube side L, mass m: I = m·(L²+L²)/12 on the diagonal = m/6 for L=1.
    expected = np.eye(3) * (1.0 / 6.0)
    assert np.allclose(mp.inertia_tensor, expected, atol=1e-6)


def test_cube_com_tracks_offset() -> None:
    mesh = unit_cube(side=2.0, center=(1.0, -2.0, 3.0))
    mp = compute_mass_properties(mesh, density=1.0)
    assert math.isclose(mp.volume, 8.0, rel_tol=1e-6)
    assert np.allclose(mp.center_of_mass, [1.0, -2.0, 3.0], atol=1e-6)
    # Inertia about COM is offset-invariant: m·(L²+L²)/12, L=2, m=8 → 8·8/12.
    diag = 8.0 * (4.0 + 4.0) / 12.0
    assert np.allclose(mp.inertia_tensor, np.eye(3) * diag, atol=1e-6)


def test_inertia_is_positive_and_offset_invariant_far_from_origin() -> None:
    """A solid far from the origin must still yield a valid COM-frame inertia.

    Regression for the catastrophic-cancellation bug: computing inertia about the
    origin and subtracting the parallel-axis term V·|com|² produced NEGATIVE
    principal moments (physically impossible) once the COM sat far from the
    origin. The COM-frame inertia is translation-invariant, so a cube at a large
    offset must match the same cube centred at the origin, and its diagonal must
    be strictly positive.
    """
    centered = compute_mass_properties(unit_cube(side=2.0, center=(0, 0, 0)))
    far = compute_mass_properties(unit_cube(side=2.0, center=(50.0, -30.0, 20.0)))

    diag = np.diag(far.inertia_tensor)
    assert np.all(diag > 0.0), f"inertia diagonal must be positive, got {diag}"
    # Translation-invariance: far-from-origin inertia == centred inertia.
    assert np.allclose(far.inertia_tensor, centered.inertia_tensor, atol=1e-6)
    # Inertia tensors are symmetric positive-definite (all eigenvalues > 0).
    assert np.all(np.linalg.eigvalsh(far.inertia_tensor) > 0.0)


def test_density_scales_mass_and_inertia() -> None:
    mesh = unit_cube(side=1.0)
    light = compute_mass_properties(mesh, density=1.0)
    heavy = compute_mass_properties(mesh, density=3.0)
    assert math.isclose(heavy.mass, 3.0 * light.mass, rel_tol=1e-9)
    assert np.allclose(heavy.inertia_tensor, 3.0 * light.inertia_tensor, atol=1e-9)
    assert math.isclose(heavy.volume, light.volume, rel_tol=1e-9)
