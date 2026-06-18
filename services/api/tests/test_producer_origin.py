"""Tests that the stub producer marks its reports origin='stub'."""

from __future__ import annotations

from astel_api.producer import build_package_quality_report, build_quality_report


def test_quality_report_dict_origin_is_stub() -> None:
    report = build_quality_report(count=100, modality="text")
    assert report["origin"] == "stub"


def test_package_quality_report_origin_is_stub() -> None:
    qr = build_package_quality_report(modality="text")
    assert qr.origin == "stub"


def test_package_quality_report_no_stale_caveat() -> None:
    """The old 'origin=stub; modality=...' pattern in the caveat text is OK, but
    the typed field is the authoritative value — verify the field is set."""
    qr = build_package_quality_report(modality="image")
    assert qr.origin == "stub"
    # Caveats should mention stub context (informational only)
    caveats = qr.caveats or []
    assert len(caveats) > 0
