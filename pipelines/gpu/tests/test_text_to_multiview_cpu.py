"""CPU tests for the multi-view camera spec (no diffusion / GPU)."""

from __future__ import annotations

from astel_gpu.text_to_multiview import DEFAULT_AZIMUTHS, default_spec


def test_default_spec_six_uses_trained_azimuths() -> None:
    spec = default_spec(6)
    assert spec.azimuth_deg == DEFAULT_AZIMUTHS
    assert spec.num_views == 6
    assert spec.elevation_deg == 0.0


def test_default_spec_other_counts_are_evenly_spaced() -> None:
    spec = default_spec(4)
    assert spec.num_views == 4
    assert spec.azimuth_deg == (0, 90, 180, 270)


def test_default_spec_eight() -> None:
    assert default_spec(8).azimuth_deg == (0, 45, 90, 135, 180, 225, 270, 315)
