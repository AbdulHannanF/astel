"""Binary STL writer for the print path (internal surface only, not the asset).

STL is emitted ONLY for 3D printing / external slicers — per CLAUDE.md §1.2 the
product asset is always splats; this mesh is a derived print artifact. Writes the
standard 80-byte-header binary STL (little-endian): per triangle a face normal,
three vertices, and a 2-byte attribute count.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .isosurface import TriMesh

_STL_TRI = np.dtype(
    [
        ("normal", "<f4", (3,)),
        ("v1", "<f4", (3,)),
        ("v2", "<f4", (3,)),
        ("v3", "<f4", (3,)),
        ("attr", "<u2"),
    ]
)


def _face_normals(
    a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> np.ndarray:
    n = np.cross(b - a, c - a)
    length = np.linalg.norm(n, axis=1, keepdims=True)
    length = np.where(length == 0.0, 1.0, length)
    return (n / length).astype(np.float32)


def write_binary_stl(mesh: TriMesh, path: str | Path) -> int:
    """Write ``mesh`` as a binary STL. Returns the byte count written."""
    v = mesh.vertices.astype(np.float32)
    a = v[mesh.faces[:, 0]]
    b = v[mesh.faces[:, 1]]
    c = v[mesh.faces[:, 2]]

    records = np.zeros(mesh.n_faces, dtype=_STL_TRI)
    records["normal"] = _face_normals(a, b, c)
    records["v1"] = a
    records["v2"] = b
    records["v3"] = c

    header = b"astel-solid L5 print mesh (derived; not the product asset)"
    header = header.ljust(80, b"\x00")[:80]
    count = np.uint32(mesh.n_faces).tobytes()

    payload = header + count + records.tobytes()
    path = Path(path)
    path.write_bytes(payload)
    return len(payload)
