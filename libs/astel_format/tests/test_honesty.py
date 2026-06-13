"""Honesty-contract tests (manifest-v0.md section 6).

"Every numeric field is either a real measurement or explicit `null` with a
`reason`. There is no '0 means we didn't check.'"
"""

from __future__ import annotations

import json
from pathlib import Path

from astel_format import (
    AstelPackage,
    HallucinationReport,
    QualityReport,
    ScaleConfidence,
    build_minimal_package,
)
from astel_format.models import GeometricError, ViewMetrics


def test_null_geometric_error_with_reason_is_accepted_and_preserved(
    small_ply_path: Path, small_ply_count: int, tmp_path: Path
) -> None:
    qr = QualityReport(
        geometric_error=GeometricError(
            units="mm",
            reference_layer="l1",
            chamfer_mm=None,
            mean_mm=None,
            p95_mm=None,
            reason="no measured reference (pure text-to-3D)",
        ),
        scale_confidence=ScaleConfidence(meters_per_unit=1.0, ci_low=0.5, ci_high=2.0),
        hallucination=HallucinationReport(
            measured_fraction=0.0, generated_fraction=1.0, unknown_fraction=0.0
        ),
        view_metrics=ViewMetrics(psnr=None, ssim=None, lpips=None, n_holdout_views=0),
        caveats=["entire asset generated from a text prompt; no held-out views"],
    )

    pkg = build_minimal_package(
        asset_id="018f6e2e-0000-7000-8000-000000000000",
        created="2026-06-13T00:00:00Z",
        generator_name="astel-pipeline-stub",
        generator_version="0.1.0",
        source_modality="text",
        l3_ply_path=small_ply_path,
        l3_count=small_ply_count,
        l3_provenance=[0.0] * small_ply_count,  # fully generated -> 0.0
        quality_report=qr,
    )

    out = tmp_path / "asset.astel"
    pkg.write(out)

    loaded = AstelPackage.read(out)
    geo = loaded.manifest.quality_report.geometric_error

    assert geo.chamfer_mm is None
    assert geo.mean_mm is None
    assert geo.p95_mm is None
    assert geo.reason == "no measured reference (pure text-to-3D)"
    assert geo.reference_layer == "l1"
    assert geo.units == "mm"

    manifest_dict = loaded.to_manifest_dict()
    geo_dict = manifest_dict["quality_report"]["geometric_error"]
    # nulls preserved as explicit JSON null, not dropped.
    assert geo_dict["chamfer_mm"] is None
    assert geo_dict["mean_mm"] is None
    assert geo_dict["p95_mm"] is None
    assert "reason" in geo_dict

    # view_metrics nulls also preserved.
    vm_dict = manifest_dict["quality_report"]["view_metrics"]
    assert vm_dict["psnr"] is None
    assert vm_dict["ssim"] is None
    assert vm_dict["lpips"] is None
    assert vm_dict["n_holdout_views"] == 0

    # 0.0 provenance for a fully-generated asset is the reserved
    # "fully generated" sentinel, not "unknown" (manifest-v0.md section 5.2).
    [channel] = loaded.manifest.provenance.channels
    accessor = loaded.manifest.buffers.accessors[channel.accessor]
    buffer_view = loaded.manifest.buffers.buffer_views[accessor.buffer_view]
    buffer_entry = loaded.manifest.buffers.buffers[buffer_view.buffer]
    prov_bytes = loaded.files[buffer_entry.uri]
    assert all(b == 0 for b in prov_bytes)


def test_quality_report_json_roundtrip_preserves_null_with_reason() -> None:
    raw = {
        "geometric_error": {
            "units": "mm",
            "reference_layer": "l1",
            "chamfer_mm": None,
            "mean_mm": None,
            "p95_mm": None,
            "reason": "no measured reference",
        },
        "scale_confidence": {"meters_per_unit": 1.0, "ci_low": 0.9, "ci_high": 1.1},
        "hallucination": {"measured_fraction": 0.1, "generated_fraction": 0.9},
    }
    qr = QualityReport.model_validate(raw)
    out = qr.model_dump(mode="json", exclude_unset=True)
    assert out == raw
    assert json.dumps(out, sort_keys=True) == json.dumps(raw, sort_keys=True)
