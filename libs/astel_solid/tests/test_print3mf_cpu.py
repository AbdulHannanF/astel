"""3MF writer: round-trip vertex/triangle counts and zip structure."""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

from _shapes import unit_cube

from astel_solid.print3mf import write_3mf


def test_3mf_is_valid_zip(tmp_path: Path) -> None:
    mesh = unit_cube()
    out = tmp_path / "cube.3mf"
    write_3mf(mesh, out)
    assert out.exists()
    assert zipfile.is_zipfile(out)


def test_3mf_contains_required_parts(tmp_path: Path) -> None:
    mesh = unit_cube()
    out = tmp_path / "cube.3mf"
    write_3mf(mesh, out)
    with zipfile.ZipFile(out) as zf:
        names = set(zf.namelist())
    assert "[Content_Types].xml" in names
    assert "_rels/.rels" in names
    assert "3D/3dmodel.model" in names


def test_3mf_vertex_triangle_counts_round_trip(tmp_path: Path) -> None:
    """Parse the model XML and verify vertex/triangle counts match the mesh."""
    mesh = unit_cube()
    out = tmp_path / "cube.3mf"
    write_3mf(mesh, out)
    with zipfile.ZipFile(out) as zf:
        model_xml = zf.read("3D/3dmodel.model").decode("utf-8")

    vertex_count = len(re.findall(r"<vertex\s", model_xml))
    triangle_count = len(re.findall(r"<triangle\s", model_xml))

    assert vertex_count == mesh.n_vertices  # 8 for a cube
    assert triangle_count == mesh.n_faces   # 12 for a cube


def test_3mf_accepts_path_string(tmp_path: Path) -> None:
    mesh = unit_cube()
    out = str(tmp_path / "cube_str.3mf")
    write_3mf(mesh, out)
    assert zipfile.is_zipfile(out)


def test_3mf_namespace_present(tmp_path: Path) -> None:
    mesh = unit_cube()
    out = tmp_path / "cube_ns.3mf"
    write_3mf(mesh, out)
    with zipfile.ZipFile(out) as zf:
        model_xml = zf.read("3D/3dmodel.model").decode("utf-8")
    assert "http://schemas.microsoft.com/3dmanufacturing/core/2015/02" in model_xml
    assert 'unit="millimeter"' in model_xml
