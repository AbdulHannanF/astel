"""CPU tests for capture_eval helpers (no gsplat/CUDA needed)."""

from __future__ import annotations

import numpy as np
import torch

from astel_gpu.capture_eval import (
    build_capture_quality_report,
    build_init_cloud,
    estimate_object_half_extent,
    split_train_test,
)
from astel_gpu.dtu import DtuScan


def test_build_init_cloud_shapes_and_bounds() -> None:
    center = np.array([10.0, -20.0, 600.0])
    spread = 50.0
    cloud = build_init_cloud(2000, center, spread, torch.device("cpu"), seed=0)
    assert cloud.means.shape == (2000, 3)
    assert cloud.scales.shape == (2000, 3)
    assert cloud.quats.shape == (2000, 4)
    assert cloud.opacities.shape == (2000,)
    assert cloud.colors.shape == (2000, 3)
    # all means within the init box around center
    lo = torch.tensor(center - spread, dtype=torch.float32)
    hi = torch.tensor(center + spread, dtype=torch.float32)
    assert bool(((cloud.means >= lo) & (cloud.means <= hi)).all())


def test_estimate_object_half_extent() -> None:
    # Three cameras at distance 100 from the origin, R = I so centre = -t.
    distance = 100.0
    centres = np.array([[0, 0, distance], [distance, 0, 0], [0, distance, 0]])
    viewmats = []
    for c in centres:
        vm = np.eye(4)
        vm[:3, 3] = -c  # R = I -> centre = -R^T t = -t = c
        viewmats.append(vm)
    ks = np.tile(np.array([[1000.0, 0, 200], [0, 1000.0, 150], [0, 0, 1]]), (3, 1, 1))
    scan = DtuScan(
        viewmats=np.stack(viewmats),
        ks=ks,
        images=np.zeros((3, 300, 400, 3), dtype=np.float32),
        width=400,
        height=300,
        object_center=np.zeros(3),
    )
    # half_extent = median(distance * (height/2) / fy) * 0.5 = 100*150/1000*0.5
    assert estimate_object_half_extent(scan) == 7.5


def test_capture_quality_report_structure() -> None:
    report = build_capture_quality_report(
        count=200_000,
        psnr_db=23.3,
        n_holdout_views=7,
        accuracy_mm=5.0,
        completeness_mm=3.0,
        n_data_eval=40000,
        n_gt_eval=900000,
        fitted_longest_axis_mm=126.0,
        gt_longest_axis_mm=126.7,
        scan_name="scan1",
    )
    assert report["origin"] == "measured"
    assert report["modality"] == "capture-dtu/scan1"
    ge = report["geometric_error"]
    assert ge["accuracy_data_to_gt_mm"] == 5.0
    assert ge["completeness_gt_to_data_mm"] == 3.0
    assert ge["chamfer_mm_vs_l1"] == 4.0  # overall = (acc + comp) / 2
    assert ge["max_dist_cap_mm"] == 60.0
    assert report["fidelity"]["n_holdout_views"] == 7
    # Scale is inherited (metric poses), so there is no scale ERROR to report.
    assert report["scale"]["relative_error"] is None
    assert report["provenance"]["measured_ratio"] == 1.0
    assert len(report["caveats"]) >= 4


def test_split_train_test() -> None:
    train, test = split_train_test(49, holdout_every=8)
    assert test == [0, 8, 16, 24, 32, 40, 48]
    assert len(train) == 42
    assert set(train).isdisjoint(test)
    assert sorted(train + test) == list(range(49))
