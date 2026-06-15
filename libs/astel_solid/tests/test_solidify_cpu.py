"""End-to-end solidify on a sampled sphere vs. analytic mass properties."""

from __future__ import annotations

import math

import numpy as np
from _shapes import fibonacci_sphere

from astel_solid.solidify import solidify, surfel_normals


def test_solidify_sphere_matches_analytic_within_tolerance() -> None:
    r = 0.5
    pts, normals = fibonacci_sphere(6000, radius=r)
    result = solidify(pts, normals, resolution=64, density=1.0)

    # The mass MATH is exact (see test_mass_cpu's cube); here we validate the
    # discretized splat→SDF→mesh pipeline lands in the analytic ballpark. A 64³
    # marching-cubes sphere is a faceted inscribed polyhedron whose zero level
    # sits slightly inside, so volume/inertia run a few-to-~15% low — honest,
    # not a bug. Tolerances reflect that discretization, not the math.
    analytic_vol = 4.0 / 3.0 * math.pi * r**3
    assert abs(result.mass.volume - analytic_vol) / analytic_vol < 0.06

    assert np.allclose(result.mass.center_of_mass, 0.0, atol=0.01)  # <2% of r

    # Solid sphere inertia: (2/5)·m·r² isotropic.
    m = result.mass.mass
    expected_diag = 2.0 / 5.0 * m * r**2
    diag = np.diag(result.mass.inertia_tensor)
    assert np.all(np.abs(diag - expected_diag) / expected_diag < 0.15)
    # Diagonal is near-isotropic (the three axes agree closely).
    assert (diag.max() - diag.min()) / diag.mean() < 0.05
    # Off-diagonal terms negligible vs. the diagonal.
    off = result.mass.inertia_tensor - np.diag(diag)
    assert np.max(np.abs(off)) < 0.02 * expected_diag


def test_surfel_normals_pick_thin_axis_and_orient_outward() -> None:
    # Two points on +x and -x; identity quats; thin axis = x (smallest log-scale).
    positions = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float32)
    quats = np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    log_scales = np.array(
        [[-5.0, -1.0, -1.0], [-5.0, -1.0, -1.0]], dtype=np.float32
    )  # x is thinnest
    normals = surfel_normals(positions, quats, log_scales)
    # Identity rotation => x-axis column is (1,0,0); oriented outward from centroid.
    assert np.allclose(normals[0], [1.0, 0.0, 0.0], atol=1e-5)
    assert np.allclose(normals[1], [-1.0, 0.0, 0.0], atol=1e-5)


def test_surfel_normals_are_unit_length() -> None:
    pts, _ = fibonacci_sphere(200, radius=1.0)
    rng = np.random.default_rng(0)
    quats = rng.standard_normal((pts.shape[0], 4)).astype(np.float32)
    log_scales = rng.standard_normal((pts.shape[0], 3)).astype(np.float32)
    normals = surfel_normals(pts, quats, log_scales)
    lengths = np.linalg.norm(normals, axis=1)
    assert np.allclose(lengths, 1.0, atol=1e-5)
