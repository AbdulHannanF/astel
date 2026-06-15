"""GPU smoke test for the synthetic controlled-ground-truth eval.

Skips on any machine that cannot run a gsplat kernel via the
``requires_gsplat_runtime`` fixture — see ``conftest.py``.
"""

from __future__ import annotations

import math


def test_synthetic_eval_psnr_climbs_and_chamfer_finite(
    requires_gsplat_runtime: None,
) -> None:
    from astel_gpu.synthetic_eval import run_synthetic_eval

    _final_params, eval_metrics, quality_report = run_synthetic_eval(
        iters=100,
        n_gaussians=1_000,
        n_views=6,
        image_size=128,
        seed=12345,
    )

    assert eval_metrics["origin"] == "measured"
    assert eval_metrics["final_psnr_db"] > eval_metrics["init_psnr_db"]

    chamfer_mm = eval_metrics["chamfer_mm"]
    for key in ("a_to_b", "b_to_a", "symmetric"):
        value = chamfer_mm[key]
        assert math.isfinite(value)
        assert value > 0.0

    assert eval_metrics["longest_axis_m"] == 0.20

    assert quality_report["schema"] == "astel.quality-report/v0"
    assert quality_report["origin"] == "measured"
    geo = quality_report["geometric_error"]
    # Raw all-means Chamfer is always reported and matches eval_metrics.
    assert geo["chamfer_raw_all_means_mm"] == chamfer_mm["symmetric"]
    assert geo["method"] in (
        "synthetic-gt-chamfer-opacity-filtered",
        "synthetic-gt-chamfer-raw-all-means",
    )
    assert geo["n_contributing_gaussians"] >= 0
    assert math.isfinite(geo["chamfer_mm_vs_l1"])
    assert geo["chamfer_mm_vs_l1"] > 0.0
    # Headline equals the opacity-filtered value when any gaussians contribute.
    filtered = eval_metrics["chamfer_filtered_mm"]
    if filtered is not None:
        assert geo["chamfer_mm_vs_l1"] == filtered["symmetric"]
    assert quality_report["scale"]["longest_axis_m"] == 0.20
    assert quality_report["scale"]["confidence"] == 1.0
    assert quality_report["scale"]["method"] == "synthetic-known"
    assert quality_report["provenance"]["measured_ratio"] == 1.0
    assert quality_report["provenance"]["generated_ratio"] == 0.0
    assert len(quality_report["caveats"]) >= 1
