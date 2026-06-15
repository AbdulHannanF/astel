"""astel-solid — L5 solidification & print path (CLAUDE.md §3 L5).

Splat cloud → SDF → watertight surface → mass properties → print exports. The
derived surface is internal scaffolding for printing / physics / collision only;
the product asset is always Gaussian splats (CLAUDE.md §1.2).
"""

from __future__ import annotations

from .isosurface import TriMesh, extract_isosurface
from .mass import MassProperties, compute_mass_properties
from .sdf import SdfGrid, oriented_point_sdf
from .solidify import SolidResult, solidify, surfel_normals
from .stl import write_binary_stl

__all__ = [
    "MassProperties",
    "SdfGrid",
    "SolidResult",
    "TriMesh",
    "compute_mass_properties",
    "extract_isosurface",
    "oriented_point_sdf",
    "solidify",
    "surfel_normals",
    "write_binary_stl",
]
