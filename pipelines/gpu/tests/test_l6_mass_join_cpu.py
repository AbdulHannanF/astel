"""CPU tests for compute_l6_masses and build_package_quality_report.origin."""

from __future__ import annotations

import numpy as np
import pytest

from astel_gpu.packaging import (
    build_l6_articulation,
    build_package_quality_report,
    compute_l6_masses,
    meters_per_unit_from_longest_axis,
)

# ---- compute_l6_masses -------------------------------------------------------


def test_single_region_grounded_mass() -> None:
    """Single region + real scale: mass == density × metric_volume."""
    regions = [
        {
            "region": "body",
            "material": "oak wood",
            "material_class": "rigid",
            "density_kg_m3": 700.0,
            "friction": 0.5,
            "restitution": 0.3,
        }
    ]
    volume_model = 0.001  # 1 litre in model units (meters when grounded)
    mpu = 1.0  # grounded (for this test we use 1.0 but treat as genuinely real)
    result = compute_l6_masses(regions, volume_model, mpu)
    # 700 kg/m³ × 0.001 m³ = 0.7 kg
    assert pytest.approx(result["total_mass_kg"], rel=1e-6) == 0.7
    assert result["metric_volume_m3"] == pytest.approx(volume_model, rel=1e-9)
    assert len(result["regions"]) == 1
    assert result["regions"][0]["region"] == "body"
    assert pytest.approx(result["regions"][0]["mass_kg"], rel=1e-6) == 0.7


def test_single_region_scale_grounded_flag_true_when_non_unit() -> None:
    regions = [{"region": "head", "density_kg_m3": 7850.0}]
    result = compute_l6_masses(regions, 0.0005, 0.01)  # 1 unit = 1 cm
    assert result["scale_grounded"] is True
    # volume_m3 = 0.0005 * 0.01^3 = 5e-13 m³
    expected_volume = 0.0005 * (0.01**3)
    assert pytest.approx(result["metric_volume_m3"], rel=1e-9) == expected_volume
    expected_mass = 7850.0 * expected_volume
    assert pytest.approx(result["total_mass_kg"], rel=1e-6) == expected_mass


def test_single_region_ungrounded_flag_and_caveat() -> None:
    """When meters_per_unit==1.0 (default/ungrounded), scale_grounded is False."""
    regions = [{"region": "body", "density_kg_m3": 1000.0}]
    result = compute_l6_masses(regions, 0.001, 1.0)
    assert result["scale_grounded"] is False
    caveats = result.get("caveats", [])
    assert any("1 unit = 1 m" in c for c in caveats), (
        "Expected ungrounded caveat in: " + repr(caveats)
    )


def test_multi_region_per_region_volume_not_segmented() -> None:
    """Multiple regions: total mass uses mean density, per_region_volume is noted."""
    regions = [
        {"region": "handle", "density_kg_m3": 700.0},  # wood
        {"region": "blade", "density_kg_m3": 7850.0},  # steel
    ]
    volume_model = 0.001
    mpu = 1.0
    result = compute_l6_masses(regions, volume_model, mpu)

    mean_density = (700.0 + 7850.0) / 2.0  # 4275.0
    expected_mass = mean_density * volume_model * (mpu**3)
    assert pytest.approx(result["total_mass_kg"], rel=1e-6) == expected_mass
    assert result["per_region_volume"] == "not-segmented"
    assert "mean_density_kg_m3" in result
    assert pytest.approx(result["mean_density_kg_m3"], rel=1e-6) == mean_density

    # Two region entries present (density only, no individual mass)
    assert len(result["regions"]) == 2

    # caveat about segmentation must be present
    caveats = result.get("caveats", [])
    assert any("not-segmented" in c for c in caveats), repr(caveats)


def test_multi_region_ungrounded_has_both_caveats() -> None:
    regions = [
        {"region": "a", "density_kg_m3": 500.0},
        {"region": "b", "density_kg_m3": 1500.0},
    ]
    result = compute_l6_masses(regions, 0.01, 1.0)
    caveats = result.get("caveats", [])
    assert any("1 unit = 1 m" in c for c in caveats), repr(caveats)
    assert any("not-segmented" in c for c in caveats), repr(caveats)


def test_no_density_values_returns_error() -> None:
    regions = [{"region": "body", "material_class": "rigid"}]  # no density_kg_m3
    result = compute_l6_masses(regions, 0.001, 1.0)
    assert "error" in result


# ---- build_package_quality_report.origin ------------------------------------


def test_build_package_quality_report_origin_is_generated() -> None:
    qr = build_package_quality_report(modality="text", origin_note="unit-test")
    assert qr.origin == "generated"


def test_build_package_quality_report_no_stale_caveat() -> None:
    """The old 'origin=measured(gpu)' string must NOT appear in caveats."""
    qr = build_package_quality_report(modality="text", origin_note="unit-test")
    caveats = qr.caveats or []
    for caveat in caveats:
        assert "origin=measured(gpu)" not in caveat, (
            "Stale misleading caveat found: " + repr(caveat)
        )


# ---- build_l6_articulation ---------------------------------------------------

_REGIONS = [
    {"region": "box", "density_kg_m3": 700.0},
    {"region": "lid", "density_kg_m3": 700.0},
]


def test_articulation_maps_joint_vocab_and_region_indices() -> None:
    """LLM joint names map to the manifest enum; region names -> indices."""
    artic = build_l6_articulation(
        [{"parent": "box", "child": "lid", "joint_type": "hinge"}], _REGIONS
    )
    assert len(artic) == 1
    assert artic[0].type == "revolute"  # hinge -> revolute
    assert artic[0].parent_region == 0  # "box"
    assert artic[0].child_region == 1  # "lid"
    assert artic[0].axis is None  # never invented (the LLM gives no axis)


@pytest.mark.parametrize(
    ("llm_joint", "manifest_type"),
    [
        ("fixed", "fixed"),
        ("hinge", "revolute"),
        ("slider", "prismatic"),
        ("ball", "free"),
        ("free", "free"),
    ],
)
def test_articulation_full_joint_vocab_maps(
    llm_joint: str, manifest_type: str
) -> None:
    """Every astel_llm.JOINT_TYPES value maps to a valid manifest joint enum.

    This is the regression guard: passing the raw LLM string straight through
    used to raise a ValidationError for hinge/slider/ball (not in the manifest
    enum), silently aborting the entire L6 bind under the best-effort guard.
    """
    artic = build_l6_articulation(
        [{"parent": "box", "child": "lid", "joint_type": llm_joint}], _REGIONS
    )
    assert artic[0].type == manifest_type


def test_articulation_unknown_region_is_none_not_crash() -> None:
    artic = build_l6_articulation(
        [{"parent": "box", "child": "ghost", "joint_type": "hinge"}], _REGIONS
    )
    assert artic[0].parent_region == 0
    assert artic[0].child_region is None  # "ghost" is not a region -> unresolved


def test_articulation_unknown_joint_type_is_none_not_crash() -> None:
    artic = build_l6_articulation(
        [{"parent": "box", "child": "lid", "joint_type": "wormhole"}], _REGIONS
    )
    assert artic[0].type is None  # unmapped -> None, recorded not crashed
    assert artic[0].parent_region == 0


def test_articulation_empty_returns_empty() -> None:
    assert build_l6_articulation([], _REGIONS) == []


# ---- meters_per_unit_from_longest_axis ---------------------------------------


def test_meters_per_unit_from_longest_axis_basic() -> None:
    """A 2-unit-wide model that is really 1 m long -> 0.5 m per unit."""
    positions = np.array(
        [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.5, 0.0]], dtype=np.float32
    )
    assert meters_per_unit_from_longest_axis(1.0, positions) == pytest.approx(0.5)


def test_meters_per_unit_uses_longest_axis_extent() -> None:
    """The grounding uses the LARGEST AABB extent, not an arbitrary axis."""
    positions = np.array(
        [[0.0, 0.0, 0.0], [1.0, 4.0, 0.0]], dtype=np.float32
    )  # extents (1, 4, 0); longest = 4
    assert meters_per_unit_from_longest_axis(2.0, positions) == pytest.approx(0.5)


def test_meters_per_unit_non_positive_estimate_is_ungrounded() -> None:
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    assert meters_per_unit_from_longest_axis(0.0, positions) == 1.0
    assert meters_per_unit_from_longest_axis(-3.0, positions) == 1.0


def test_meters_per_unit_degenerate_extent_is_ungrounded() -> None:
    """All points coincident (zero extent) -> ungrounded fallback, no div-by-zero."""
    positions = np.zeros((10, 3), dtype=np.float32)
    assert meters_per_unit_from_longest_axis(1.0, positions) == 1.0


def test_meters_per_unit_empty_positions_is_ungrounded() -> None:
    assert meters_per_unit_from_longest_axis(1.0, np.zeros((0, 3))) == 1.0
