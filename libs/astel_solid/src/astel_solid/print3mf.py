"""3MF writer for the print path (internal surface only, not the asset).

3MF is emitted ONLY for 3D printing / external slicers — per CLAUDE.md §1.2 the
product asset is always splats; this mesh is a derived print artifact.

Implements the 3MF Core spec (http://schemas.microsoft.com/3dmanufacturing/core/2015/02)
as a hand-rolled OPC zip:
  [Content_Types].xml
  _rels/.rels
  3D/3dmodel.model

No extra dependencies: stdlib ``zipfile`` + string XML only, matching the spirit
of ``stl.py``.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np

from .isosurface import TriMesh

_NAMESPACE = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"

_CT_RELS = "application/vnd.openxmlformats-package.relationships+xml"
_CT_MODEL = "application/vnd.ms-package.3dmanufacturing-3dmodel+xml"
_CT_NS = "http://schemas.openxmlformats.org/package/2006/content-types"
_RELS_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
_RELS_TYPE = (
    "http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"
)

_CONTENT_TYPES = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    f'<Types xmlns="{_CT_NS}">\n'
    f'  <Default Extension="rels" ContentType="{_CT_RELS}"/>\n'
    f'  <Default Extension="model" ContentType="{_CT_MODEL}"/>\n'
    "</Types>\n"
)

_RELS = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    f'<Relationships xmlns="{_RELS_NS}">\n'
    f'  <Relationship Id="r1" Type="{_RELS_TYPE}"'
    ' Target="/3D/3dmodel.model"/>\n'
    "</Relationships>\n"
)


def _build_model_xml(mesh: TriMesh) -> str:
    """Serialise the mesh into the 3dmodel.model XML string."""
    v = mesh.vertices.astype(np.float32)
    f = mesh.faces

    vertex_lines: list[str] = []
    for x, y, z in v:
        vertex_lines.append(f'        <vertex x="{x:.6g}" y="{y:.6g}" z="{z:.6g}"/>')

    triangle_lines: list[str] = []
    for v1, v2, v3 in f:
        triangle_lines.append(f'        <triangle v1="{v1}" v2="{v2}" v3="{v3}"/>')

    vertices_block = "\n".join(vertex_lines)
    triangles_block = "\n".join(triangle_lines)

    return (
        f'<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<model unit="millimeter" xml:lang="en-US"\n'
        f'       xmlns="{_NAMESPACE}">\n'
        f"  <resources>\n"
        f'    <object id="1" type="model">\n'
        f"      <mesh>\n"
        f"        <vertices>\n"
        f"{vertices_block}\n"
        f"        </vertices>\n"
        f"        <triangles>\n"
        f"{triangles_block}\n"
        f"        </triangles>\n"
        f"      </mesh>\n"
        f"    </object>\n"
        f"  </resources>\n"
        f'  <build>\n'
        f'    <item objectid="1"/>\n'
        f'  </build>\n'
        f"</model>\n"
    )


def write_3mf(mesh: TriMesh, path: str | Path) -> None:
    """Write ``mesh`` as a 3MF archive to ``path``.

    The 3MF is an OPC zip with:
    - ``[Content_Types].xml`` — MIME-type map
    - ``_rels/.rels`` — relationship pointing at the model part
    - ``3D/3dmodel.model`` — vertices + triangles in the 3MF Core namespace

    Unit is "millimeter". Coordinates are taken verbatim from the mesh (world
    space in model units). When metric grounding is needed, the caller is
    responsible for scaling prior to writing.
    """
    model_xml = _build_model_xml(mesh)
    path = Path(path)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", _CONTENT_TYPES.encode("utf-8"))
        zf.writestr("_rels/.rels", _RELS.encode("utf-8"))
        zf.writestr("3D/3dmodel.model", model_xml.encode("utf-8"))
