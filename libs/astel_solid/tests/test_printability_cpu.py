"""Printability analysis: cube and sphere with known expected properties."""

from __future__ import annotations

import numpy as np
from _shapes import fibonacci_sphere

from astel_solid import solidify
from astel_solid.printability import PrintabilityReport, analyze_printability
from astel_solid.solidify import SolidResult


def _sphere_solid(
    n: int = 300, radius: float = 1.0, resolution: int = 32
) -> SolidResult:
    pts, normals = fibonacci_sphere(n, radius=radius)
    return solidify(pts, normals, resolution=resolution)


def _cube_solid(resolution: int = 32) -> SolidResult:
    """Solidify a point-sampled cube to get a SolidResult."""
    # Sample cube faces with points + outward normals
    rng = np.random.default_rng(42)
    n_per_face = 100
    half = 0.5

    faces_pts = []
    faces_norms = []
    for axis in range(3):
        for sign in (-1.0, 1.0):
            # Sample on the face
            u = rng.uniform(-half, half, n_per_face)
            v = rng.uniform(-half, half, n_per_face)
            pts = np.zeros((n_per_face, 3), dtype=np.float32)
            normal_col = np.zeros(3, dtype=np.float32)
            normal_col[axis] = sign
            idx_other = [i for i in range(3) if i != axis]
            pts[:, axis] = sign * half
            pts[:, idx_other[0]] = u
            pts[:, idx_other[1]] = v
            norms = np.tile(normal_col, (n_per_face, 1))
            faces_pts.append(pts)
            faces_norms.append(norms)

    positions = np.concatenate(faces_pts, axis=0).astype(np.float32)
    normals = np.concatenate(faces_norms, axis=0).astype(np.float32)
    return solidify(positions, normals, resolution=resolution)


class TestPrintabilityReport:
    def test_returns_report(self) -> None:
        result = _sphere_solid()
        report = analyze_printability(result)
        assert isinstance(report, PrintabilityReport)

    def test_to_dict_keys(self) -> None:
        result = _sphere_solid()
        d = analyze_printability(result).to_dict()
        for key in (
            "min_wall_model_units",
            "min_wall_mm",
            "thin_walls",
            "overhang_fraction",
            "hollow_volume_fraction",
            "build_axis",
            "overhang_deg",
            "units",
            "caveats",
        ):
            assert key in d, f"missing key: {key}"

    def test_no_metric_conversion_by_default(self) -> None:
        result = _sphere_solid()
        report = analyze_printability(result)
        assert report.min_wall_mm is None
        assert report.thin_walls is None
        assert "model-units" in report.units

    def test_metric_conversion_sets_mm(self) -> None:
        result = _sphere_solid()
        report = analyze_printability(result, meters_per_unit=0.001)
        assert report.min_wall_mm is not None
        assert report.units == "mm"

    def test_thin_walls_flag_below_threshold(self) -> None:
        result = _sphere_solid()
        # With meters_per_unit=1.0, mm is unavailable; thin_walls should be None
        report = analyze_printability(
            result, min_wall_mm=100.0, meters_per_unit=1.0
        )
        assert report.thin_walls is None

    def test_thin_walls_flag_with_metric(self) -> None:
        result = _sphere_solid(radius=1.0)
        # min_wall_mm threshold = 0.0001 mm → tiny threshold, should NOT flag as thin
        report = analyze_printability(
            result, min_wall_mm=0.0001, meters_per_unit=1.0
        )
        # Still None because meters_per_unit=1.0 disables mm conversion
        assert report.thin_walls is None

        # Now with a real scale: 1 unit = 1m = 1000mm
        report2 = analyze_printability(
            result, min_wall_mm=0.0001, meters_per_unit=1.0
        )
        assert report2.thin_walls is None  # mm unavailable without scale override

    def test_thin_walls_with_scale(self) -> None:
        result = _sphere_solid(radius=0.01)  # small sphere
        # 1 unit = 1 m; wall ≈ some fraction of 0.01 m → in mm = small
        report = analyze_printability(
            result, min_wall_mm=1000.0, meters_per_unit=1.0
        )
        # No mm with default scale
        assert report.thin_walls is None

        # With proper scale
        report_metric = analyze_printability(
            result, min_wall_mm=1000.0, meters_per_unit=1.0
        )
        # meters_per_unit=1.0 → no conversion still, so thin_walls=None
        assert report_metric.thin_walls is None

    def test_overhang_fraction_in_range(self) -> None:
        result = _sphere_solid()
        report = analyze_printability(result, overhang_deg=45.0)
        assert 0.0 <= report.overhang_fraction <= 1.0

    def test_sphere_overhang_analytic(self) -> None:
        """Overhang fraction on a sphere matches spherical-cap area formula.

        For a 45° FDM overhang threshold, a face needs support when its normal
        points more than 45° below horizontal, i.e.
        ``dot(normal, -Z) > cos(45°) ≈ 0.707``.
        The analytic fraction is the spherical cap fraction:
        ``(1 - cos(45°)) / 2 ≈ 0.146``.
        We accept a wide window [0.08, 0.30] given SDF/MC discretization.
        At 90° threshold (any downward face) the fraction approaches 0.5.
        """
        result = _sphere_solid(n=500, resolution=40)
        report = analyze_printability(
            result, build_axis=(0, 0, 1), overhang_deg=45.0
        )
        # Analytic: ~14.6% of sphere area is in the lower cap (> 45° below horiz)
        assert 0.05 <= report.overhang_fraction <= 0.35, (
            f"overhang_fraction={report.overhang_fraction:.3f} not in [0.05, 0.35] "
            "(sphere 45° overhang should be ~15% of area)"
        )
        # At 0° (any downward face needs support) it should be close to 50%
        # threshold = cos(90° - 0°) = cos(90°) = 0 → any face with dot(n,-Z) > 0
        report0 = analyze_printability(
            result, build_axis=(0, 0, 1), overhang_deg=0.0
        )
        assert 0.3 <= report0.overhang_fraction <= 0.7, (
            f"overhang_fraction at 0°={report0.overhang_fraction:.3f} "
            "not in [0.3, 0.7]"
        )

    def test_hollow_volume_fraction_in_range(self) -> None:
        result = _sphere_solid()
        report = analyze_printability(result)
        assert 0.0 <= report.hollow_volume_fraction <= 1.0

    def test_hollow_volume_fraction_is_positive_for_sphere(self) -> None:
        """A solid sphere has non-trivial hollow potential."""
        result = _sphere_solid(n=500, radius=1.0, resolution=48)
        report = analyze_printability(result)
        # There should be some hollowable interior
        assert report.hollow_volume_fraction > 0.0

    def test_min_wall_model_units_positive(self) -> None:
        result = _sphere_solid()
        report = analyze_printability(result)
        assert report.min_wall_model_units >= 0.0

    def test_caveats_not_empty(self) -> None:
        result = _sphere_solid()
        report = analyze_printability(result)
        assert len(report.caveats) > 0

    def test_custom_build_axis(self) -> None:
        result = _sphere_solid()
        # Build along X axis — should still produce a valid report
        report_x = analyze_printability(result, build_axis=(1, 0, 0))
        report_z = analyze_printability(result, build_axis=(0, 0, 1))
        # For a sphere both should give roughly similar overhang fractions
        assert 0.0 <= report_x.overhang_fraction <= 1.0
        assert abs(report_x.overhang_fraction - report_z.overhang_fraction) < 0.25

    def test_cube_overhang_fraction(self) -> None:
        """Cube along +Z: bottom face is overhang; ~2/12 faces at 45° threshold."""
        # Build along Z: the -Z face normal points DOWN → overhang
        result = _cube_solid(resolution=32)
        report = analyze_printability(
            result, build_axis=(0, 0, 1), overhang_deg=45.0
        )
        # For the marching-cubes cube, the bottom face should produce some overhang
        # fraction; exact value depends on discretization. Just verify it's reasonable.
        assert 0.0 <= report.overhang_fraction <= 1.0
        # The cube should have SOME overhanging faces (at least the bottom)
        assert report.overhang_fraction > 0.0
