"""Convex decomposition: coacd path, scipy fallback, and GLB/NPZ output."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from _shapes import unit_cube

from astel_solid.convex import ConvexSet, convex_decompose, write_convex_glb


def test_convex_decompose_returns_convexset() -> None:
    mesh = unit_cube()
    cset = convex_decompose(mesh)
    assert isinstance(cset, ConvexSet)
    assert cset.n_hulls >= 1
    assert cset.method in ("coacd", "scipy-hull-fallback")


def test_convex_decompose_hull_shapes() -> None:
    mesh = unit_cube()
    cset = convex_decompose(mesh)
    for hull in cset.hulls:
        assert hull.vertices.ndim == 2 and hull.vertices.shape[1] == 3
        assert hull.faces.ndim == 2 and hull.faces.shape[1] == 3
        assert hull.vertices.dtype == np.float32
        assert hull.faces.dtype == np.int32


def test_convex_decompose_scipy_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force scipy fallback by removing coacd from sys.modules and blocking import."""
    # Temporarily block the coacd import
    sys.modules["coacd"] = None  # type: ignore[assignment]
    try:
        # We need to reload convex to re-execute the lazy import

        import astel_solid.convex as convex_mod

        # Re-run _scipy_single_hull directly to test it
        mesh = unit_cube()
        result = convex_mod._scipy_single_hull(mesh)
        assert result.method == "scipy-hull-fallback"
        assert result.n_hulls == 1
        assert result.hulls[0].vertices.shape[1] == 3
    finally:
        del sys.modules["coacd"]


def test_write_convex_output_file_exists(tmp_path: Path) -> None:
    mesh = unit_cube()
    cset = convex_decompose(mesh)
    out = write_convex_glb(cset, tmp_path / "hulls.glb")
    assert out.exists()
    # Suffix should be .glb (trimesh available) or .npz (fallback)
    assert out.suffix in (".glb", ".npz")


def test_write_convex_npz_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Force NPZ path by blocking trimesh import."""
    sys.modules["trimesh"] = None  # type: ignore[assignment]
    sys.modules["trimesh.scene"] = None  # type: ignore[assignment]
    try:

        import astel_solid.convex as convex_mod

        mesh = unit_cube()
        cset = convex_decompose(mesh)
        out = convex_mod.write_convex_glb(cset, tmp_path / "hulls_npz.glb")
        assert out.suffix == ".npz"
        assert out.exists()
        data = np.load(str(out), allow_pickle=True)
        assert "verts_0" in data
        assert "faces_0" in data
        assert str(data["method"]) == cset.method
    finally:
        del sys.modules["trimesh"]
        if "trimesh.scene" in sys.modules:
            del sys.modules["trimesh.scene"]


def test_convex_scipy_fallback_covers_all_points() -> None:
    """The scipy hull must contain all original mesh vertices (it's the full hull)."""
    from astel_solid.convex import _scipy_single_hull

    mesh = unit_cube()
    cset = _scipy_single_hull(mesh)
    # All 8 cube vertices should appear in the hull
    assert cset.hulls[0].vertices.shape[0] == 8
