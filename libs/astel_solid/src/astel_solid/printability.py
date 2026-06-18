"""Printability analysis of a solidified mesh (L5 print-path gate).

Analyses the ``SolidResult`` produced by :func:`astel_solid.solidify` and
returns a ``PrintabilityReport`` covering the most common FDM/SLA print issues:
thin walls, overhangs, and potential for hollowing material savings.

All geometry is read from the existing ``SolidResult`` (mesh + SDF grid); the
mesh is never mutated.

HONESTY (CLAUDE.md §10.4):
- When ``meters_per_unit == 1.0``, wall thickness and volume are reported in
  *model units*, not millimetres.  This is stated explicitly in the report.
- When ``meters_per_unit`` is supplied, conversions are exact; bias from
  marching-cubes discretization is documented in the report caveat.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .solidify import SolidResult


@dataclass(frozen=True)
class PrintabilityReport:
    """Printability summary for a solidified asset.

    Fields
    ------
    min_wall_model_units
        Thinnest wall estimated from the interior SDF (interior distance × 2).
        Always present; in model units regardless of ``meters_per_unit``.
    min_wall_mm
        Same in millimetres — only set when ``meters_per_unit != 1.0``.
    thin_walls
        ``True`` iff ``min_wall_mm`` is set AND falls below ``min_wall_mm``
        threshold passed by the caller. ``None`` when threshold not provided
        or metric conversion unavailable.
    overhang_fraction
        Area-weighted fraction of faces whose downward angle vs
        ``build_axis`` exceeds ``overhang_deg``.  These faces need support
        structures.
    hollow_volume_fraction
        Fraction of the solid volume that could be saved by shelling (eroding
        the interior SDF by half the min-wall distance).  Purely informational.
    build_axis
        Unit build direction used for overhang computation.
    overhang_deg
        Overhang angle threshold in degrees.
    units
        ``"mm"`` when metric conversion was applied; otherwise
        ``"model-units (not metric)"`` to be explicit.
    caveats
        Honesty notes: discretization bias, scale grounding status, etc.
    """

    min_wall_model_units: float
    min_wall_mm: float | None
    thin_walls: bool | None
    overhang_fraction: float
    hollow_volume_fraction: float
    build_axis: tuple[float, float, float]
    overhang_deg: float
    units: str
    caveats: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_wall_model_units": self.min_wall_model_units,
            "min_wall_mm": self.min_wall_mm,
            "thin_walls": self.thin_walls,
            "overhang_fraction": self.overhang_fraction,
            "hollow_volume_fraction": self.hollow_volume_fraction,
            "build_axis": list(self.build_axis),
            "overhang_deg": self.overhang_deg,
            "units": self.units,
            "caveats": self.caveats,
        }


def _face_normals(
    verts: NDArray[np.float32], faces: NDArray[np.int64]
) -> NDArray[np.float32]:
    """Return (F,3) unit outward face normals via cross-product of edges."""
    a = verts[faces[:, 0]]
    b = verts[faces[:, 1]]
    c = verts[faces[:, 2]]
    n = np.cross(b - a, c - a).astype(np.float64)
    length = np.linalg.norm(n, axis=1, keepdims=True)
    length = np.where(length == 0.0, 1.0, length)
    return (n / length).astype(np.float32)


def _face_areas(
    verts: NDArray[np.float32], faces: NDArray[np.int64]
) -> NDArray[np.float64]:
    """Return (F,) triangle areas."""
    a = verts[faces[:, 0]].astype(np.float64)
    b = verts[faces[:, 1]].astype(np.float64)
    c = verts[faces[:, 2]].astype(np.float64)
    cross = np.cross(b - a, c - a)
    return np.array(np.linalg.norm(cross, axis=1) * 0.5, dtype=np.float64)


def analyze_printability(
    result: SolidResult,
    *,
    build_axis: tuple[float, float, float] = (0, 0, 1),
    min_wall_mm: float | None = None,
    meters_per_unit: float = 1.0,
    overhang_deg: float = 45.0,
) -> PrintabilityReport:
    """Analyse ``result`` for common FDM/SLA printability issues.

    Parameters
    ----------
    result
        The ``SolidResult`` from :func:`~astel_solid.solidify.solidify`.
    build_axis
        Unit vector for the print build direction (gravity-opposing axis).
        Overhangs are faces whose downward component exceeds ``overhang_deg``.
    min_wall_mm
        When provided, ``thin_walls`` is set to ``True`` iff the measured
        thinnest wall (in mm after metric conversion) is below this threshold.
        Requires ``meters_per_unit != 1.0`` for a meaningful mm comparison.
    meters_per_unit
        Scale factor to convert model units → metres (1 model unit = N metres).
        When ``1.0`` (default), no metric conversion is performed and wall
        thickness is reported in model units only.
    overhang_deg
        Angle threshold in degrees below which faces are considered overhangs
        (angle measured between the face normal and the build axis DOWN vector).
    """
    mesh = result.mesh
    grid = result.grid

    caveats: list[str] = []

    # ------------------------------------------------------------------
    # Wall thickness: interior SDF minimum × 2
    # The SDF is negative inside (IMLS / Hoppe convention: negative inside,
    # positive outside). The interior minimum represents the local signed
    # distance to the nearest surface; at the thinnest wall both sides are
    # surfaces, so wall ≈ 2 × |min interior distance|.
    # ------------------------------------------------------------------
    interior_mask = grid.values < 0.0
    if interior_mask.any():
        interior_vals = grid.values[interior_mask]  # negative values
        min_interior_dist = float(np.abs(interior_vals).min())  # closest to surface
        # Scale by voxel spacing (values are in world units already)
        min_wall_model = min_interior_dist * 2.0
    else:
        # No interior found (open mesh / all-positive SDF) — conservative estimate
        min_wall_model = 0.0
        caveats.append(
            "SDF has no interior voxels; wall thickness estimate is 0 (open mesh or "
            "padding too small for this resolution)."
        )

    if meters_per_unit != 1.0:
        mm_per_unit = meters_per_unit * 1000.0
        min_wall_mm_value: float | None = min_wall_model * mm_per_unit
        units = "mm"
    else:
        min_wall_mm_value = None
        units = "model-units (not metric)"
        caveats.append(
            "meters_per_unit=1.0 (default): wall thickness and volume are in MODEL "
            "units, not millimetres. Pass meters_per_unit to enable metric conversion."
        )

    caveats.append(
        "Wall thickness derived from marching-cubes SDF at resolution "
        f"{list(grid.values.shape)}; carries voxel-spacing discretization bias."
    )

    if min_wall_mm is not None and min_wall_mm_value is not None:
        thin_walls: bool | None = min_wall_mm_value < min_wall_mm
    elif min_wall_mm is not None and min_wall_mm_value is None:
        thin_walls = None
        caveats.append(
            "thin_walls threshold was requested in mm, but meters_per_unit=1.0 so "
            "metric conversion is unavailable. Provide meters_per_unit to enable."
        )
    else:
        thin_walls = None

    # ------------------------------------------------------------------
    # Overhangs: area-weighted fraction of faces angled against build_axis
    # A face is an overhang when its downward component cos > cos(90° - threshold)
    # i.e. when angle(face_normal, build_axis) > (90° + overhang_deg) ... actually
    # standard: overhang if angle(face_normal, -build_axis) < (90° - overhang_deg).
    # Equivalently: dot(face_normal, -build_axis) > cos(90° - overhang_deg).
    # ------------------------------------------------------------------
    bx = np.asarray(build_axis, dtype=np.float64)
    bx_len = float(np.linalg.norm(bx))
    bx = np.array([0.0, 0.0, 1.0]) if bx_len == 0.0 else bx / bx_len

    down = -bx  # direction gravity / print direction acts against
    threshold_cos = float(np.cos(np.radians(90.0 - overhang_deg)))  # cos(45) ≈ 0.707

    fnormals = _face_normals(mesh.vertices, mesh.faces).astype(np.float64)  # (F,3)
    areas = _face_areas(mesh.vertices, mesh.faces)  # (F,)
    dot_down = fnormals @ down  # (F,)  positive = facing down = overhang candidate

    overhang_mask = dot_down > threshold_cos
    total_area = float(areas.sum())
    if total_area > 0.0:
        overhang_fraction = float(areas[overhang_mask].sum() / total_area)
    else:
        overhang_fraction = 0.0

    # ------------------------------------------------------------------
    # Hollowing estimate: erode interior SDF by half the min-wall distance,
    # count voxels inside the eroded boundary vs. total interior voxels.
    # hollow_volume_fraction ≈ fraction of solid volume saveable by shelling.
    # ------------------------------------------------------------------
    if min_wall_model > 0.0 and interior_mask.any():
        erosion_dist = min_wall_model / 2.0  # half-wall inward margin
        # Interior SDF is negative; more negative = deeper inside.
        # Eroded interior = voxels with SDF < -(erosion_dist).
        eroded_mask = grid.values < -erosion_dist
        n_interior = float(np.count_nonzero(interior_mask))
        n_eroded = float(np.count_nonzero(eroded_mask))
        hollow_volume_fraction = float(n_eroded / n_interior) if n_interior > 0 else 0.0
    else:
        hollow_volume_fraction = 0.0

    return PrintabilityReport(
        min_wall_model_units=min_wall_model,
        min_wall_mm=min_wall_mm_value,
        thin_walls=thin_walls,
        overhang_fraction=overhang_fraction,
        hollow_volume_fraction=hollow_volume_fraction,
        build_axis=(float(bx[0]), float(bx[1]), float(bx[2])),
        overhang_deg=overhang_deg,
        units=units,
        caveats=caveats,
    )
