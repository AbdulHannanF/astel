"""Full layer-stack artifact writer for the GPU producer (parity with the stub).

The API's stub producer (``services/api/.../producer.py``) emits a complete
``.astel`` artifact contract: ``l0.ply``, ``l3.ply``, ``l3.spz``, ``l3.sog``,
``package.astel`` and ``quality-report.json``. The GPU producer historically
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
from typing import Any

from astel_format.builder import build_minimal_package
from astel_format.models import (
    GeometricError,
    HallucinationReport,
    QualityReport,
    ScaleConfidence,
)
from astel_splat_io.cloud import SplatCloud
from astel_splat_io.ply import write_ply
from astel_splat_io.sog import write_sog
from astel_splat_io.spz import write_spz

logger = logging.getLogger(__name__)

#: Voxel resolution (longest-axis) for the producer's best-effort L5 solidify.
#: Kept modest for producer responsiveness; the print/physics-grade pass can run
#: a finer grid on demand.
L5_SDF_RESOLUTION: int = 48

#: L0 "seed" cloud = a deterministic strided subsample of L3 (mirrors the stub).
#: The GPU producer has no separate SfM/latent seed stage in these paths, so the
#: cheap preview tier is an evenly-strided slice of the refined cloud, documented
#: as such in the report caveats.
L0_SUBSAMPLE_DIVISOR: int = 24

GENERATOR_NAME = "astel-gpu"


def seed_cloud(cloud: SplatCloud, divisor: int = L0_SUBSAMPLE_DIVISOR) -> SplatCloud:
    """Derive a sparse L0 seed cloud by deterministic strided subsampling."""
    import numpy as np

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
        caveats=[
            f"origin=measured(gpu); modality={modality}. {origin_note}",
            "All quality metrics in this package report are unmeasured (null) by "
            "design: this path has no ground-truth geometry and no metric-scale "
            "grounding. The measured self-consistency PSNR is reported separately "
            "in quality-report.json (fidelity), flagged as self-consistency, not "
            "accuracy vs. reality.",
        ],
    )


def _try_solidify(
    l3_cloud: SplatCloud, out_dir: Path, *, resolution: int
) -> dict[str, Any] | None:
    """Best-effort L5: splat → SDF → watertight surface → mass props + ``l5.stl``.

    Returns a JSON-able solidity summary (or ``None`` if the cloud doesn't yield a
    closed surface). Wrapped broadly: solidification must never fail an asset.
    Volume/inertia are in MODEL units (not metric unless scale-grounded).
    """
    try:
        from astel_solid import solidify, surfel_normals, write_binary_stl

        normals = surfel_normals(
            l3_cloud.positions, l3_cloud.quats, l3_cloud.log_scales
        )
        result = solidify(
            l3_cloud.positions, normals, resolution=resolution, density=1.0
        )
        write_binary_stl(result.mesh, out_dir / "l5.stl")

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
            "note": (
                "Derived INTERNAL surface for print/physics/collision; the asset "
                "remains splats (CLAUDE.md §1.2). Volume/inertia carry the "
                "marching-cubes discretization bias."
            ),
        }
        (out_dir / "l5-mass.json").write_text(json.dumps(summary, indent=2))
        return summary
    except Exception:
        logger.exception("L5 solidify failed (best-effort); skipping l5 artifacts")
        return None


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
) -> list[str]:
    """Write the full ``.astel`` artifact contract for ``l3_cloud`` into ``out_dir``.

    Emits ``l3.ply``, ``l0.ply``, ``l3.spz``, ``l3.sog``, ``package.astel`` and
    ``quality-report.json`` (plus ``l2.ply`` when ``l2_cloud`` is given — the
    pre-refinement generator output). When ``solidify_l5`` and the L3 cloud yields
    a watertight surface, also emits ``l5.stl`` + ``l5-mass.json`` and injects a
    ``solidity`` summary into the report (best-effort — never fatal). Returns the
    sorted list of written file names. Torch-free: a CPU-testable seam.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[str] = []

    l0_cloud = seed_cloud(l3_cloud)

    l3_ply = out_dir / "l3.ply"
    write_ply(l3_cloud, l3_ply)
    artifacts.append("l3.ply")

    l0_ply = out_dir / "l0.ply"
    write_ply(l0_cloud, l0_ply)
    artifacts.append("l0.ply")

    if l2_cloud is not None:
        write_ply(l2_cloud, out_dir / "l2.ply")
        artifacts.append("l2.ply")

    # L5 solidity (CLAUDE.md §3 L5): derive an internal watertight surface +
    # mass properties from the splats. Best-effort — a noisy/open cloud may not
    # solidify, and that must never fail the asset (the surface is scaffolding,
    # not the deliverable; the asset is always splats, §1.2).
    if solidify_l5:
        solidity = _try_solidify(l3_cloud, out_dir, resolution=L5_SDF_RESOLUTION)
        if solidity is not None:
            artifacts.append("l5.stl")
            artifacts.append("l5-mass.json")
            report_dict["solidity"] = solidity

    # Compressed-delivery exports of the L3 cloud (SPZ byte-exact; SOG
    # best-effort per astel_splat_io.sog's documented caveats).
    write_spz(l3_cloud, out_dir / "l3.spz")
    artifacts.append("l3.spz")
    write_sog(l3_cloud, out_dir / "l3.sog")
    artifacts.append("l3.sog")

    # Full .astel package binding L0 + L3 with per-gaussian provenance. Both the
    # smoke and generative paths are fully generated (provenance 0.0 == generated
    # under the manifest convention "1=measured, 0=generated").
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
    )
    package.write(out_dir / "package.astel")
    artifacts.append("package.astel")

    (out_dir / "quality-report.json").write_text(json.dumps(report_dict, indent=2))
    artifacts.append("quality-report.json")

    return sorted(artifacts)
