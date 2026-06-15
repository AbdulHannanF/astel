"""Mass properties of a closed triangle mesh via the divergence theorem.

Decomposes the solid into signed tetrahedra (each triangle with the origin) and
accumulates the exact polynomial integrals (Blow & Binstock / Eberly), giving the
volume, centre of mass, and inertia tensor about the COM for a uniform density.
Pure numpy, fully vectorised. Used to populate L5/L6 physics setup (mass, COM,
inertia) bound to the splat asset — never a standalone deliverable.

Validates against analytic solids: a unit-density sphere of radius ``r`` yields
``V = 4/3·π·r³`` and a diagonal inertia ``(2/5)·m·r²``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .isosurface import TriMesh

# Integral of x·xᵀ over the canonical tetrahedron (origin, e1, e2, e3).
_C_CANONICAL: NDArray[np.float64] = (
    np.array([[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]]) / 120.0
)


@dataclass(frozen=True)
class MassProperties:
    """Uniform-density mass properties of a closed mesh (SI-ish: metres, kg/m³)."""

    volume: float
    mass: float
    density: float
    center_of_mass: NDArray[np.float64]
    inertia_tensor: NDArray[np.float64]  # (3,3) about the COM


def compute_mass_properties(mesh: TriMesh, *, density: float = 1.0) -> MassProperties:
    """Volume, COM, and COM-frame inertia tensor for a closed, outward-wound mesh.

    ``density`` is mass per unit volume; ``mass = density · volume``. Assumes the
    mesh is watertight and outward-wound (as produced by
    :func:`astel_solid.isosurface.extract_isosurface`).
    """
    v = mesh.vertices.astype(np.float64)
    a = v[mesh.faces[:, 0]]  # (F,3)
    b = v[mesh.faces[:, 1]]
    c = v[mesh.faces[:, 2]]

    # Signed 6·volume of each tetra (origin, a, b, c) = det[a b c].
    det = np.einsum("ij,ij->i", a, np.cross(b, c))  # (F,)
    volume = float(det.sum() / 6.0)
    if abs(volume) < 1e-18:
        raise ValueError("degenerate mesh: ~zero volume")

    # First moment ∫ x dV = Σ det·(a+b+c)/24  →  COM = M1 / V.
    m1 = (det[:, None] * (a + b + c) / 24.0).sum(axis=0)  # (3,)
    com = m1 / volume

    # Compute the inertia in the COM frame DIRECTLY by recentering the vertices
    # to the COM first. The textbook alternative (inertia about the origin, then
    # subtract the parallel-axis term V·(|com|²·I − com⊗com)) is a difference of
    # two large near-equal matrices when the mesh sits far from the origin: with
    # the marching-cubes discretization bias it loses all significance and can
    # even emit NEGATIVE principal moments (physically impossible). Recentering
    # makes the second-moment integral O(object size), not O(distance-to-origin),
    # so the result is stable wherever the mesh happens to live in space.
    ac, bc, cc = a - com, b - com, c - com
    det_c = np.einsum("ij,ij->i", ac, np.cross(bc, cc))  # signed 6·vol, COM frame

    # Second moment about the COM: S = ∫ x'·x'ᵀ dV = Σ det · J·C·Jᵀ.
    j = np.stack([ac, bc, cc], axis=2)  # (F,3,3): columns are the recentered verts
    s_tet = j @ _C_CANONICAL @ np.transpose(j, (0, 2, 1))  # (F,3,3)
    s = (det_c[:, None, None] * s_tet).sum(axis=0)  # (3,3) about the COM

    # Inertia about the COM: I = trace(S)·Id − S (no parallel-axis shift needed).
    mass = density * volume
    inertia_com = density * (np.trace(s) * np.eye(3) - s)

    return MassProperties(
        volume=abs(volume),
        mass=abs(mass),
        density=density,
        center_of_mass=com,
        inertia_tensor=inertia_com,
    )
