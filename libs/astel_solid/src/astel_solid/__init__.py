"""astel-solid — L5 solidification & print path (CLAUDE.md §3 L5).

Splat cloud → SDF → watertight surface → mass properties → print exports. The
derived surface is internal scaffolding for printing / physics / collision only;
the product asset is always Gaussian splats (CLAUDE.md §1.2).
"""

from __future__ import annotations

from .convex import ConvexHull, ConvexSet, convex_decompose, write_convex_glb
from .isosurface import TriMesh, extract_isosurface
from .mass import MassProperties, compute_mass_properties
from .print3mf import write_3mf
from .printability import PrintabilityReport, analyze_printability
from .sdf import SdfGrid, oriented_point_sdf
from .solidify import SolidResult, solidify, surfel_normals
from .stl import write_binary_stl

__all__ = [
    "ConvexHull",
    "ConvexSet",
    "MassProperties",
    "PrintabilityReport",
    "SdfGrid",
    "SolidResult",
    "TriMesh",
    "analyze_printability",
    "compute_mass_properties",
    "convex_decompose",
    "extract_isosurface",
    "oriented_point_sdf",
    "solidify",
    "surfel_normals",
    "write_3mf",
    "write_binary_stl",
    "write_convex_glb",
]
