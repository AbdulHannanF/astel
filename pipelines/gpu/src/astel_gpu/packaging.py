"""Full layer-stack artifact writer for the GPU producer (parity with the stub).

The API's stub producer (``services/api/.../producer.py``) emits a complete
``.astel`` artifact contract: ``l0.ply``, ``l3.ply``, ``l3.spz``, ``l3.sog``,
``l3.glb``, ``package.astel`` and ``quality-report.json``. The GPU producer historically
emitted only ``l3.ply`` + ``quality-report.json`` + a metrics file, so a GPU
generation produced a *thinner* asset than the CPU stub. This module closes that
gap: given the refined L3 :class:`~astel_splat_io.cloud.SplatCloud`, it writes the
identical artifact set so the web viewer, Layer Inspector and ``.astel`` consumers
see the same contract whether the producer is the stub or the real GPU pipeline.

The writer is deliberately **torch-free** (it operates on the numpy-backed
``SplatCloud`` and the pydantic ``QualityReport``), so it is a CPU-testable seam:
the GPU producer converts its trained ``GaussianParams`` to a ``SplatCloud`` (the
one torch step, in :mod:`astel_gpu.export`) and then hands off to here.

HONESTY (CLAUDE.md §10.4): the typed package ``QualityReport`` carries explicit
``None`` + ``reason`` for every unmeasured field — geometric error vs. reality is
genuinely undefined for both the self-consistency smoke and the generative paths
(no ground-truth scan), so it is never fabricated. The real, measured
self-consistency PSNR lives in the ``quality-report.json`` dict's ``fidelity``
block, flagged for what it is.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import astel_dynamics
import astel_lod
import numpy as np
from astel_format.builder import build_minimal_package
from astel_format.models import (
    GeometricError,
    HallucinationReport,
    LayerArticulation,
    QualityReport,
    ScaleConfidence,
)
from astel_splat_io.cloud import SplatCloud
from astel_splat_io.gltf import write_gltf
from astel_splat_io.ply import write_ply
from astel_splat_io.sog import write_sog
from astel_splat_io.spz import write_spz

logger = logging.getLogger(__name__)

#: Voxel resolution (longest-axis) for the producer's best-effort L5 solidify.
#: Kept modest for producer responsiveness; the print/physics-grade pass can run
#: a finer grid on demand.
L5_SDF_RESOLUTION: int = 48

#: Max convex hulls for the L5 collision proxy decomposition.
L5_MAX_HULLS: int = 32

#: L0 "seed" cloud = a deterministic strided subsample of L3 (mirrors the stub).
#: The GPU producer has no separate SfM/latent seed stage in these paths, so the
#: cheap preview tier is an evenly-strided slice of the refined cloud, documented
#: as such in the report caveats.
L0_SUBSAMPLE_DIVISOR: int = 24

GENERATOR_NAME = "astel-gpu"

#: Map the L6 physics-material LLM's joint vocabulary (``astel_llm.JOINT_TYPES``:
#: ``fixed/hinge/slider/ball/free``) onto the ``.astel`` manifest's
#: ``LayerArticulation.type`` enum (``revolute/prismatic/fixed/free`` from
#: layer.schema.json). The LLM speaks URDF-ish names; the manifest speaks the
#: splat-asset joint enum, so the two must be reconciled or the binding raises.
#: ``hinge`` -> ``revolute`` (1-DOF rotation), ``slider`` -> ``prismatic`` (1-DOF
#: translation), ``ball`` -> ``free`` (the manifest has no spherical joint, so a
#: 3-DOF ball is reported as free rather than over-constrained to a 1-DOF joint).
#: An unmapped joint type binds as ``type=None`` (recorded, under-specified) — a
#: previous version passed the raw LLM string straight through, which raised a
#: pydantic ValidationError on any non-``fixed/free`` joint and (under the broad
#: best-effort guard) silently dropped the entire L6 mass join.
_JOINT_TYPE_MAP: dict[str, Literal["revolute", "prismatic", "fixed", "free"]] = {
    "fixed": "fixed",
    "hinge": "revolute",
    "slider": "prismatic",
    "ball": "free",
    "free": "free",
}


def seed_cloud(cloud: SplatCloud, divisor: int = L0_SUBSAMPLE_DIVISOR) -> SplatCloud:
    """Derive a sparse L0 seed cloud by deterministic strided subsampling."""
    step = max(1, divisor)
    order = np.arange(0, cloud.count, step, dtype=np.intp)
    return cloud.reordered(order)


def build_package_quality_report(
    *, modality: str, origin_note: str
) -> QualityReport:
    """Typed ``.astel`` package quality report for a GPU-produced asset.

    Honest by construction: geometric error vs. reality is ``None`` (no GT scan
    exists for either the self-consistency smoke or a generated object), scale is
    the ungrounded identity (1 unit == 1 m, zero-width CI named as a non-estimate),
    and the asset is reported as fully generated (0% measured). ``origin_note``
    names which GPU path produced it.
    """
    no_gt_reason = (
        "No ground-truth geometry exists for this asset (self-consistency / "
        "generative path), so Chamfer-vs-L1 is not measured. Geometric accuracy "
        "vs. reality arrives only with the COLMAP/real-capture path (M2)."
    )
    return QualityReport(
        geometric_error=GeometricError(
            units="mm",
            reference_layer="l0",
            chamfer_mm=None,
            mean_mm=None,
            p95_mm=None,
            method="not-measured",
            reason=no_gt_reason,
        ),
        # Ungrounded scale: identity with a zero-width CI, named as a non-estimate
        # (the schema forbids non-positive bounds, so 1.0/1.0/1.0 is the honest
        # encoding of "no metric grounding performed here").
        scale_confidence=ScaleConfidence(
            meters_per_unit=1.0,
            ci_low=1.0,
            ci_high=1.0,
            ci_method="gpu-no-estimate",
        ),
        hallucination=HallucinationReport(
            measured_fraction=0.0,
            generated_fraction=1.0,
        ),
        origin="generated",
        caveats=[
            f"GPU generative path; modality={modality}. {origin_note}",
            "All quality metrics in this package report are unmeasured (null) by "
            "design: this path has no ground-truth geometry and no metric-scale "
            "grounding. The measured self-consistency PSNR is reported separately "
            "in quality-report.json (fidelity), flagged as self-consistency, not "
            "accuracy vs. reality.",
        ],
    )


def build_l6_articulation(
    raw_articulation: list[dict[str, Any]],
    raw_regions: list[dict[str, Any]],
) -> list[LayerArticulation]:
    """Map the LLM's articulation hints onto manifest ``LayerArticulation`` entries.

    The L6 physics-material LLM emits articulation as ``{parent, child,
    joint_type}`` using region *names* and the URDF-ish joint vocabulary
    (``astel_llm.JOINT_TYPES``). The manifest ``LayerArticulation`` wants integer
    region *indices* (into the regions list, parallel to the per-gaussian region
    map) and the splat-asset joint enum. This resolves names -> indices (via
    :data:`_JOINT_TYPE_MAP`) and maps the vocabulary. Honest by construction
    (CLAUDE.md §10.4):

    * an unknown region name -> ``None`` (the link is recorded, its endpoint is
      flagged unresolved rather than invented),
    * an unknown joint type -> ``type=None`` (recorded, under-specified) instead
      of crashing the bind,
    * ``axis`` stays ``None`` — the LLM provides no joint axis, so none is
      fabricated.
    """
    name_to_index = {
        str(r["region"]): i
        for i, r in enumerate(raw_regions)
        if r.get("region") is not None
    }
    out: list[LayerArticulation] = []
    for art in raw_articulation:
        # The manifest schema forbids null members (each field is optional but
        # typed), so an unresolved region name or unmapped joint type is OMITTED
        # rather than emitted as null; ``axis`` is never set (the LLM gives none).
        fields: dict[str, Any] = {}
        joint = _JOINT_TYPE_MAP.get(str(art.get("joint_type", "")))
        if joint is not None:
            fields["type"] = joint
        parent_idx = name_to_index.get(str(art.get("parent")))
        if parent_idx is not None:
            fields["parent_region"] = parent_idx
        child_idx = name_to_index.get(str(art.get("child")))
        if child_idx is not None:
            fields["child_region"] = child_idx
        out.append(LayerArticulation(**fields))
    return out


def meters_per_unit_from_longest_axis(
    longest_axis_m: float, positions: np.ndarray
) -> float:
    """Metric scale (metres per model-unit) from a longest-axis size estimate.

    ``longest_axis_m`` is the asset's real longest-axis length in metres (the
    Generation Spec's VLM size estimate); ``positions`` is the L3 cloud in model
    (native) units. The model-space longest axis is the largest AABB extent.
    Returns ``longest_axis_m / model_longest_axis`` so that
    ``model_extent × meters_per_unit == longest_axis_m``. Falls back to ``1.0``
    (ungrounded) when the estimate or the model extent is non-positive — a scale
    is never fabricated (CLAUDE.md §10.4).
    """
    if longest_axis_m <= 0.0 or positions.size == 0:
        return 1.0
    extent = positions.max(axis=0) - positions.min(axis=0)
    model_longest = float(np.max(extent))
    if model_longest <= 0.0:
        return 1.0
    return longest_axis_m / model_longest


def compute_l6_masses(
    l6_regions: list[dict[str, Any]],
    total_volume_model_units: float,
    meters_per_unit: float,
) -> dict[str, Any]:
    """Compute per-region mass estimates by joining L6 region data with L5 volume.

    Parameters
    ----------
    l6_regions:
        List of region dicts from ``l6.json`` (each with at least
        ``density_kg_m3``). Typically the ``spec.regions`` list from the
        physics-material stage.
    total_volume_model_units:
        Total watertight volume in model (native) units, from L5 solidify.
    meters_per_unit:
        Scale factor for the coordinate system. When ``1.0`` (ungrounded),
        the mass computation assumes 1 unit == 1 m and is flagged accordingly
        (CLAUDE.md §10.4 honesty contract).

    Returns a JSON-serialisable dict with ``total_mass_kg``, mass caveats,
    and either per-region mass entries (one region) or a note that per-region
    segmentation is not yet available (multiple regions with no region-map).
    """
    metric_volume_m3 = total_volume_model_units * (meters_per_unit**3)
    scale_grounded = meters_per_unit != 1.0

    densities = [
        float(r.get("density_kg_m3", 0.0)) for r in l6_regions if r.get("density_kg_m3")
    ]
    if not densities:
        return {
            "error": "no density_kg_m3 values found in l6_regions",
            "scale_grounded": scale_grounded,
        }

    n_regions = len(l6_regions)

    caveats: list[str] = []
    if not scale_grounded:
        caveats.append(
            "masses assume 1 unit = 1 m; metric grounding not performed on this "
            "asset. Apply real meters_per_unit from SfM/metric-depth before using "
            "mass values in a simulation."
        )

    if n_regions == 1:
        density = densities[0]
        mass_kg = density * metric_volume_m3
        region_label = str(l6_regions[0].get("region", "region_0"))
        result: dict[str, Any] = {
            "metric_volume_m3": metric_volume_m3,
            "total_mass_kg": mass_kg,
            "regions": [
                {
                    "region": region_label,
                    "density_kg_m3": density,
                    "mass_kg": mass_kg,
                }
            ],
            "scale_grounded": scale_grounded,
        }
    else:
        # Multiple regions but no per-region volume segmentation: use mean
        # density for the total mass estimate only.  Per-region masses would
        # require a region-map → SDF intersection, which is future work.
        mean_density = sum(densities) / len(densities)
        total_mass_kg = mean_density * metric_volume_m3
        caveats.append(
            f"per_region_volume: not-segmented. {n_regions} regions present but "
            "no region-map→SDF intersection is available, so total mass uses the "
            "mean region density. Per-region mass segmentation (region-map + SDF "
            "intersection) is future work."
        )
        result = {
            "metric_volume_m3": metric_volume_m3,
            "total_mass_kg": total_mass_kg,
            "mean_density_kg_m3": mean_density,
            "per_region_volume": "not-segmented",
            "regions": [
                {"region": str(r.get("region", f"region_{i}")),
                 "density_kg_m3": float(r.get("density_kg_m3", 0.0))}
                for i, r in enumerate(l6_regions)
            ],
            "scale_grounded": scale_grounded,
        }

    if caveats:
        result["caveats"] = caveats
    return result


#: Engine-setup sidecar schema version (the flat descriptor the Unity/UE5
#: plugins consume — see docs/architecture/coordinate-conventions.md and the
#: plugin READMEs). Denormalised on purpose: engine importers should not have to
#: walk the nested manifest + chase file-path-referenced sidecars in C#/C++.
ENGINE_SETUP_SCHEMA = "astel.engine-setup/v0"


def build_engine_setup(
    *,
    meters_per_unit: float,
    splat_file: str,
    solidity: dict[str, Any] | None,
    l6_mass: dict[str, Any] | None,
    l6_regions: list[dict[str, Any]] | None,
    articulation: list[LayerArticulation] | None,
    handedness: str = "right",
    up_axis: str = "+Y",
    forward_axis: str = "-Z",
) -> dict[str, Any]:
    """Assemble the flat ``engine.json`` physics-setup descriptor for engine plugins.

    This is the *denormalised* view of the L5 collision + L6 physics-material
    layers that the Unity and UE5 plugins read directly, instead of walking the
    nested ``manifest.json`` and chasing its file-path-referenced sidecars
    (``l5-mass.json`` / ``l6.json``) in C#/C++. Pure + CPU-testable.

    Honest by construction (CLAUDE.md §1.3, §10.4):

    - ``l5`` is ``None`` when solidify produced no watertight surface; ``l6`` is
      ``None`` when no physics-material layer exists. The plugins handle both.
    - ``mass_kg`` is the **metric** mass from the L6↔L5 join (``l6-mass.json``)
      when present; without it, mass is left ``0.0`` (the plugins fall back to a
      unit mass) and a caveat is recorded — a model-unit "mass at unit density"
      is never passed off as kilograms.
    - ``center_of_mass`` / ``inertia_diagonal`` stay in **model units**; the
      plugins scale position-like quantities by ``meters_per_unit`` and flip
      handedness per the coordinate-convention doc.
    - ``scale_grounded`` mirrors whether a real metric scale was applied.
    """
    notes: list[str] = []
    setup: dict[str, Any] = {
        "schema": ENGINE_SETUP_SCHEMA,
        "meters_per_unit": meters_per_unit,
        "splat_file": splat_file,
        "coordinate_system": {
            "handedness": handedness,
            "up_axis": up_axis,
            "forward_axis": forward_axis,
            "note": (
                "Astel 3DGS world frame. Apply the per-engine transform from "
                "docs/architecture/coordinate-conventions.md."
            ),
        },
        "scale_grounded": meters_per_unit != 1.0,
        "l5": None,
        "l6": None,
        "notes": notes,
    }

    if not setup["scale_grounded"]:
        notes.append(
            "scale ungrounded (meters_per_unit == 1.0): mass/length values assume "
            "1 model unit = 1 m."
        )

    # --- L5 mass properties (from the solidify summary + the L6 mass join) ---
    if solidity is not None:
        mass_kg = 0.0
        if l6_mass is not None and "total_mass_kg" in l6_mass:
            mass_kg = float(l6_mass["total_mass_kg"])
        else:
            notes.append(
                "mass_kg unavailable (no L6 physics-material mass join); engines "
                "should treat the body as unit-mass until L6 is present."
            )
        com = [float(x) for x in solidity.get("center_of_mass", [0.0, 0.0, 0.0])]
        inertia = [float(x) for x in solidity.get("inertia_diagonal", [0.0, 0.0, 0.0])]
        volume_model_units = float(solidity.get("volume", 0.0))
        setup["l5"] = {
            "mass_props": {
                "volume_m3": volume_model_units * (meters_per_unit**3),
                "mass_kg": mass_kg,
                "center_of_mass": com,
                "inertia_diagonal": inertia,
            }
        }

    # --- L6 per-region materials + articulation (region names -> int indices) ---
    if l6_regions:
        regions_out = [
            {
                "name": str(r.get("region", f"region_{i}")),
                "material": r.get("material"),
                "density_kg_m3": float(r.get("density_kg_m3", 0.0)),
                "friction": float(r.get("friction", 0.0)),
                "restitution": float(r.get("restitution", 0.0)),
            }
            for i, r in enumerate(l6_regions)
        ]
        articulation_out: list[dict[str, Any]] = []
        for art in articulation or []:
            entry: dict[str, Any] = {"joint_type": art.type}
            if art.parent_region is not None:
                entry["region_a"] = art.parent_region
            if art.child_region is not None:
                entry["region_b"] = art.child_region
            if art.axis is not None:
                entry["axis"] = list(art.axis)
            articulation_out.append(entry)
        setup["l6"] = {"regions": regions_out, "articulation": articulation_out}

    return setup


def _try_solidify(
    l3_cloud: SplatCloud,
    out_dir: Path,
    *,
    resolution: int,
    max_hulls: int = L5_MAX_HULLS,
) -> dict[str, Any] | None:
    """Best-effort L5: splat → SDF → watertight surface → mass props + exports.

    Writes:
    - ``l5.stl`` (always, binary STL)
    - ``l5.3mf`` (always, OPC 3MF — no extra deps)
    - ``l5-convex.glb`` or ``l5-convex.npz`` (best-effort; trimesh optional)

    Returns a JSON-able solidity summary (or ``None`` if the cloud doesn't yield
    a closed surface). Wrapped broadly: solidification must never fail an asset.
    Volume/inertia are in MODEL units (not metric unless scale-grounded).
    """
    try:
        from astel_solid import (
            analyze_printability,
            convex_decompose,
            solidify,
            surfel_normals,
            write_3mf,
            write_binary_stl,
            write_convex_glb,
        )

        normals = surfel_normals(
            l3_cloud.positions, l3_cloud.quats, l3_cloud.log_scales
        )
        result = solidify(
            l3_cloud.positions, normals, resolution=resolution, density=1.0
        )
        write_binary_stl(result.mesh, out_dir / "l5.stl")
        write_3mf(result.mesh, out_dir / "l5.3mf")

        diag = [float(result.mass.inertia_tensor[i, i]) for i in range(3)]
        summary: dict[str, Any] = {
            "source": "L5 solidify (splat -> SDF -> watertight surface)",
            "units": "model-units (not metric unless scale-grounded)",
            "volume": float(result.mass.volume),
            "mass_at_unit_density": float(result.mass.mass),
            "center_of_mass": [float(x) for x in result.mass.center_of_mass],
            "inertia_diagonal": diag,
            "mesh": {
                "vertices": result.mesh.n_vertices,
                "faces": result.mesh.n_faces,
            },
            "sdf_resolution": list(result.grid.values.shape),
            "stl": "l5.stl",
            "3mf": "l5.3mf",
            "note": (
                "Derived INTERNAL surface for print/physics/collision; the asset "
                "remains splats (CLAUDE.md §1.2). Volume/inertia carry the "
                "marching-cubes discretization bias."
            ),
        }

        # Convex decomposition (best-effort — never fatal)
        try:
            cset = convex_decompose(result.mesh, max_hulls=max_hulls)
            convex_path = write_convex_glb(
                cset, out_dir / "l5-convex.glb"
            )
            convex_name = convex_path.name
            summary["convex"] = {
                "file": convex_name,
                "n_hulls": cset.n_hulls,
                "method": cset.method,
            }
        except Exception:
            logger.warning(
                "L5 convex decomposition failed (best-effort); skipping",
                exc_info=True,
            )
            convex_name = None

        # Printability analysis (best-effort — never fatal)
        try:
            pr = analyze_printability(result)
            summary["printability"] = pr.to_dict()
        except Exception:
            logger.warning(
                "L5 printability analysis failed (best-effort); skipping",
                exc_info=True,
            )

        (out_dir / "l5-mass.json").write_text(json.dumps(summary, indent=2))
        return summary
    except Exception:
        logger.exception("L5 solidify failed (best-effort); skipping l5 artifacts")
        return None


def _try_appearance(
    l3_cloud: SplatCloud, out_dir: Path
) -> dict[str, Any] | None:
    """Best-effort L4: decompose baked colour into albedo + estimated SH env.

    Writes ``l4-albedo.ply`` (un-lit base colour), ``l4-env.json`` (estimated
    illumination), ``l4.json`` (summary) and ``l4-relight.json`` (web studio
    preview). Returns the L4 summary dict (or ``None`` on failure — appearance
    decomposition must never fail an asset; the asset stays splats either way).
    """
    try:
        from astel_appearance import build_appearance

        art = build_appearance(
            l3_cloud.positions,
            l3_cloud.colors_dc,
            l3_cloud.quats,
            l3_cloud.log_scales,
            l3_cloud.opacity,
        )
        albedo_cloud = SplatCloud(
            positions=l3_cloud.positions,
            colors_dc=art.albedo_colors_dc.astype(np.float32),
            opacity=l3_cloud.opacity,
            log_scales=l3_cloud.log_scales,
            quats=l3_cloud.quats,
        )
        write_ply(albedo_cloud, out_dir / "l4-albedo.ply")
        (out_dir / "l4-env.json").write_text(json.dumps(art.env, indent=2))
        (out_dir / "l4.json").write_text(json.dumps(art.summary, indent=2))
        (out_dir / "l4-relight.json").write_text(
            json.dumps(art.relight_preview)
        )
        summary: dict[str, Any] = art.summary
        return summary
    except Exception:
        logger.exception("L4 appearance failed (best-effort); skipping l4 artifacts")
        return None


def _write_lod(l3_cloud: SplatCloud, out_dir: Path) -> dict[str, Any] | None:
    """Best-effort LOD emission: importance-ranked tier PLYs + ``l3.lod.json``.

    Always includes a "full" tier pointing at the master ``l3.ply`` (no new
    file written). For each named tier in :data:`astel_lod.TIER_BUDGETS` whose
    budget is STRICTLY LESS THAN N (the cloud is large enough to downsample),
    writes ``l3.lod.<name>.ply`` and adds a tier entry. Deduplication: if two
    budgets are equal (should not happen with the current constants, but guarded
    for forward-safety) or equal N, only one tier is emitted.

    Returns the descriptor dict (``astel.lod/v0``) or ``None`` on failure —
    LOD emission must never fail an asset.
    """
    try:
        n = l3_cloud.count

        # Collect unique downsample budgets strictly below N (ascending).
        seen_counts: set[int] = {n}  # "full" tier reserves N
        downsample_tiers: list[tuple[str, int]] = []
        for tier_name, budget in sorted(
            astel_lod.TIER_BUDGETS.items(), key=lambda kv: kv[1]
        ):
            if budget < n and budget not in seen_counts:
                seen_counts.add(budget)
                downsample_tiers.append((tier_name, budget))

        # Build all LOD index arrays in one pass (generate_lod_indices computes
        # importance once internally from opacity + log_scales).
        tiers: list[dict[str, Any]] = []

        # "full" tier — master file, no copy needed.
        tiers.append({"name": "full", "count": n, "file": "l3.ply"})

        if downsample_tiers:
            budgets = [b for _, b in downsample_tiers]
            index_arrays = astel_lod.generate_lod_indices(
                l3_cloud.opacity, l3_cloud.log_scales, budgets
            )
            for (tier_name, _budget), indices in zip(
                downsample_tiers, index_arrays, strict=True
            ):
                actual_count = len(indices)
                # Honest: only emit if count is distinct from what we already have.
                if actual_count in seen_counts:
                    continue
                seen_counts.add(actual_count)
                lod_cloud = l3_cloud.reordered(indices)
                lod_filename = f"l3.lod.{tier_name}.ply"
                write_ply(lod_cloud, out_dir / lod_filename)
                tiers.append(
                    {"name": tier_name, "count": actual_count, "file": lod_filename}
                )

        descriptor: dict[str, Any] = astel_lod.build_lod_descriptor(tiers)
        astel_lod.write_descriptor(descriptor, out_dir / "l3.lod.json")
        return descriptor
    except Exception:
        logger.exception("LOD emission failed (best-effort); skipping LOD artifacts")
        return None


def write_dynamics_layer(
    field: astel_dynamics.DeformationField,
    timeline: astel_dynamics.Timeline,
    out_dir: Path,
    *,
    representation: str = "deformation_field",
) -> tuple[Path, Path]:
    """Write L7 deformation field + timeline into ``out_dir``.

    Writes ``l7-deformation.bin`` (via :func:`astel_dynamics.write_deformation_bin`)
    and ``l7-timeline.json`` (via :func:`astel_dynamics.write_timeline_json`).
    Returns ``(deformation_path, timeline_path)``.

    This is the public helper that the video/dynamics pipeline stage calls after
    fitting the deformation field; the paths are then passed to
    :func:`write_layer_stack` (via ``l7_deformation_path`` / ``l7_timeline_path``)
    to bind the L7 layer into the ``.astel`` package.
    """
    deformation_path = out_dir / "l7-deformation.bin"
    timeline_path = out_dir / "l7-timeline.json"
    astel_dynamics.write_deformation_bin(field, deformation_path)
    astel_dynamics.write_timeline_json(timeline, timeline_path)
    return deformation_path, timeline_path


def write_layer_stack(
    l3_cloud: SplatCloud,
    out_dir: Path,
    *,
    task_id: str,
    modality: str,
    prompt: str,
    seed: int,
    report_dict: dict[str, Any],
    package_report: QualityReport,
    l2_cloud: SplatCloud | None = None,
    solidify_l5: bool = True,
    appearance_l4: bool = True,
    meters_per_unit: float = 1.0,
    longest_axis_m: float | None = None,
    l7_deformation_path: Path | None = None,
    l7_timeline_path: Path | None = None,
    l7_representation: str | None = None,
) -> list[str]:
    """Write the full ``.astel`` artifact contract for ``l3_cloud`` into ``out_dir``.

    Emits ``l3.ply``, ``l0.ply``, ``l3.spz``, ``l3.sog``, ``l3.glb``,
    ``engine.json``, ``package.astel`` and
    ``quality-report.json`` (plus ``l2.ply`` when ``l2_cloud`` is given — the
    pre-refinement generator output). When ``solidify_l5`` and the L3 cloud yields
    a watertight surface, also emits ``l5.stl`` + ``l5-mass.json`` and injects a
    ``solidity`` summary into the report (best-effort — never fatal). When
    ``l6.json`` is present in ``out_dir``, computes the L6↔L5 mass join and binds
    both L5 and L6 layers into the ``.astel`` package. When ``longest_axis_m`` (the
    Generation Spec's metric size estimate, metres) is supplied, ``meters_per_unit``
    is derived from it and the L3 extent so the masses and the package manifest are
    metrically grounded. Returns the sorted list of written file names. Torch-free:
    a CPU-testable seam.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[str] = []

    # Metric grounding: when a real longest-axis size estimate is supplied (the
    # Generation Spec's VLM estimate), derive meters_per_unit from it + the L3
    # extent so the L6 mass join and the package manifest are metrically grounded
    # (scale_grounded:true). Without an estimate the scale stays the ungrounded
    # identity (1 unit == 1 m), honestly flagged downstream.
    if longest_axis_m is not None:
        meters_per_unit = meters_per_unit_from_longest_axis(
            longest_axis_m, l3_cloud.positions
        )

    l0_cloud = seed_cloud(l3_cloud)

    l3_ply = out_dir / "l3.ply"
    write_ply(l3_cloud, l3_ply)
    artifacts.append("l3.ply")

    # LOD emission (CLAUDE.md §8.6): importance-ranked tier PLYs + descriptor.
    # Best-effort, never fatal. Only emits downsampled tiers when the cloud is
    # large enough (budget < N); for small/test clouds, only "full" tier is
    # recorded. The "full" tier always points at l3.ply — no duplicate copy.
    lod_descriptor = _write_lod(l3_cloud, out_dir)
    if lod_descriptor is not None:
        artifacts.append("l3.lod.json")
        for tier in lod_descriptor.get("tiers", []):
            tier_file = str(tier.get("file", ""))
            if tier_file and tier_file != "l3.ply" and tier_file not in artifacts:
                artifacts.append(tier_file)
        report_dict["lod"] = lod_descriptor

    l0_ply = out_dir / "l0.ply"
    write_ply(l0_cloud, l0_ply)
    artifacts.append("l0.ply")

    if l2_cloud is not None:
        write_ply(l2_cloud, out_dir / "l2.ply")
        artifacts.append("l2.ply")

    # L4 appearance (CLAUDE.md §3 L4): split the baked L3 colour into per-splat
    # albedo + an estimated SH environment so the asset relights. Best-effort,
    # CPU-pure (torch-free, via astel_appearance).
    appearance: dict[str, Any] | None = None
    if appearance_l4:
        appearance = _try_appearance(l3_cloud, out_dir)
        if appearance is not None:
            artifacts.append("l4-albedo.ply")
            artifacts.append("l4-env.json")
            artifacts.append("l4.json")
            artifacts.append("l4-relight.json")
            report_dict["appearance"] = appearance

    # L5 solidity (CLAUDE.md §3 L5): derive an internal watertight surface +
    # mass properties from the splats. Best-effort — a noisy/open cloud may not
    # solidify, and that must never fail the asset (the surface is scaffolding,
    # not the deliverable; the asset is always splats, §1.2).
    solidity: dict[str, Any] | None = None
    if solidify_l5:
        solidity = _try_solidify(l3_cloud, out_dir, resolution=L5_SDF_RESOLUTION)
        if solidity is not None:
            artifacts.append("l5.stl")
            artifacts.append("l5.3mf")
            artifacts.append("l5-mass.json")
            # Convex file may be .glb or .npz depending on trimesh availability
            if "convex" in solidity and isinstance(solidity["convex"], dict):
                convex_file = solidity["convex"].get("file")
                if convex_file:
                    artifacts.append(str(convex_file))
            report_dict["solidity"] = solidity

    # L6 mass join: when l6.json was written by the physics-material stage and
    # solidify succeeded (volume known), join density data with volume to produce
    # l6-mass.json (CLAUDE.md §3 L6 — density × L5 volume = mass).
    l6_json_path = out_dir / "l6.json"
    l6_regions_path: Path | None = None
    l6_articulation_list: list[LayerArticulation] = []
    l6_regions_raw: list[dict[str, Any]] = []
    l6_mass_result: dict[str, Any] | None = None

    if l6_json_path.exists():
        try:
            l6_data = json.loads(l6_json_path.read_text(encoding="utf-8"))
            spec_data = l6_data.get("spec", {})
            raw_regions: list[dict[str, Any]] = spec_data.get("regions", [])
            raw_articulation: list[dict[str, Any]] = spec_data.get("articulation", [])
            l6_regions_path = l6_json_path
            l6_regions_raw = raw_regions

            # Map LLM articulation hints (region names + URDF-ish joint vocabulary)
            # onto the manifest's int region indices + joint enum. See
            # build_l6_articulation for the honesty contract (unresolved -> None,
            # never a crash, never an invented axis).
            l6_articulation_list = build_l6_articulation(
                raw_articulation, raw_regions
            )

            # Mass join only possible when we have a volume
            if solidity is not None and raw_regions:
                volume = float(solidity.get("volume", 0.0))
                mass_result = compute_l6_masses(raw_regions, volume, meters_per_unit)
                l6_mass_result = mass_result
                l6_mass_path = out_dir / "l6-mass.json"
                l6_mass_path.write_text(json.dumps(mass_result, indent=2))
                artifacts.append("l6-mass.json")
                report_dict["l6"] = mass_result
        except Exception:
            logger.warning(
                "L6 mass join failed (best-effort); l6.json present but not joined",
                exc_info=True,
            )

    # Compressed-delivery exports of the L3 cloud (SPZ byte-exact; SOG
    # best-effort per astel_splat_io.sog's documented caveats).
    write_spz(l3_cloud, out_dir / "l3.spz")
    artifacts.append("l3.spz")
    write_sog(l3_cloud, out_dir / "l3.sog")
    artifacts.append("l3.sog")
    # KHR_gaussian_splatting glTF (RC schema) — the broadly-interoperable
    # interop export (any glTF viewer/engine). Same 3DGS frame as the .ply
    # master; only the quaternion order differs (see astel_splat_io.gltf).
    write_gltf(l3_cloud, out_dir / "l3.glb")
    artifacts.append("l3.glb")

    # Propagate origin from the typed package report into the served report dict
    # so both the package manifest and quality-report.json agree.
    report_dict.setdefault("origin", package_report.origin)

    # Resolve L4 file paths for package binding (only when decomposition ran)
    l4_env_path: Path | None = None
    l4_albedo_path: Path | None = None
    l4_summary_path: Path | None = None
    if appearance is not None:
        l4_env_path = out_dir / "l4-env.json"
        l4_albedo_path = out_dir / "l4-albedo.ply"
        l4_summary_path = out_dir / "l4.json"

    # Resolve L5 file paths for package binding (only when solidify succeeded)
    l5_isosurface_path: Path | None = None
    l5_convex_set_path: Path | None = None
    l5_mass_props_path: Path | None = None
    if solidity is not None:
        l5_isosurface_path = out_dir / "l5.stl"
        l5_mass_props_path = out_dir / "l5-mass.json"
        convex_info = solidity.get("convex")
        if isinstance(convex_info, dict) and convex_info.get("file"):
            l5_convex_set_path = out_dir / str(convex_info["file"])

    # Full .astel package binding L0 + L3 (+ L5/L6 when available) with
    # per-gaussian provenance. Both the smoke and generative paths are fully
    # generated (provenance 0.0 == generated under the manifest convention
    # "1=measured, 0=generated").
    package = build_minimal_package(
        asset_id=task_id,
        created=datetime.now(UTC).isoformat(),
        generator_name=GENERATOR_NAME,
        generator_version="0.1.0",
        source_modality=modality,
        l3_ply_path=l3_ply,
        l3_count=l3_cloud.count,
        l3_provenance=[0.0] * l3_cloud.count,
        l0_ply_path=l0_ply,
        l0_count=l0_cloud.count,
        l0_provenance=[0.0] * l0_cloud.count,
        quality_report=package_report,
        prompt=prompt or None,
        seed=seed,
        meters_per_unit=meters_per_unit,
        l4_env_path=l4_env_path,
        l4_albedo_path=l4_albedo_path,
        l4_summary_path=l4_summary_path,
        l5_isosurface_path=l5_isosurface_path,
        l5_convex_set_path=l5_convex_set_path,
        l5_mass_props_path=l5_mass_props_path,
        l6_regions_path=l6_regions_path,
        l6_articulation=l6_articulation_list if l6_articulation_list else None,
        l7_deformation_path=l7_deformation_path,
        l7_timeline_path=l7_timeline_path,
        l7_representation=l7_representation,
    )
    package.write(out_dir / "package.astel")
    artifacts.append("package.astel")

    # engine.json — the flat physics-setup descriptor the Unity/UE5 plugins read
    # (denormalised view of L5 collision + L6 material/mass/articulation). Emitted
    # alongside the package so an engine importer never has to walk the nested
    # manifest + chase its file-referenced sidecars in C#/C++.
    engine_setup = build_engine_setup(
        meters_per_unit=meters_per_unit,
        splat_file="l3.spz",
        solidity=solidity,
        l6_mass=l6_mass_result,
        l6_regions=l6_regions_raw or None,
        articulation=l6_articulation_list or None,
    )
    (out_dir / "engine.json").write_text(json.dumps(engine_setup, indent=2))
    artifacts.append("engine.json")

    (out_dir / "quality-report.json").write_text(json.dumps(report_dict, indent=2))
    artifacts.append("quality-report.json")

    return sorted(artifacts)
