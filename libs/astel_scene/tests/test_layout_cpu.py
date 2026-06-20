"""Layout serialisation tests: to_dict/from_dict and JSON round-trips."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from astel_scene.layout import Placement, SceneLayout, SceneObject

_SCHEMA = "astel.scene-layout/v0"


def _sample_layout() -> SceneLayout:
    return SceneLayout(
        objects=[
            SceneObject(
                object_id="chair",
                prompt="a wooden chair",
                placement=Placement(
                    object_id="chair",
                    yaw_deg=45.0,
                    uniform_scale=1.5,
                    translation=(1.0, 0.0, -2.0),
                    ground_contact=True,
                ),
            ),
            SceneObject(
                object_id="table",
                prompt="a round coffee table",
                placement=Placement(
                    object_id="table",
                    yaw_deg=0.0,
                    uniform_scale=2.0,
                    translation=(0.0, 0.0, 0.0),
                    ground_contact=False,
                ),
            ),
        ],
        up_axis="+Y",
        ground_y=-0.5,
    )


# ---------------------------------------------------------------------------
# to_dict / from_dict round-trip
# ---------------------------------------------------------------------------


def test_schema_key_present() -> None:
    d = _sample_layout().to_dict()
    assert d["schema"] == _SCHEMA


def test_to_dict_from_dict_round_trip() -> None:
    original = _sample_layout()
    d = original.to_dict()
    restored = SceneLayout.from_dict(d)

    assert restored.up_axis == original.up_axis
    assert restored.ground_y == original.ground_y
    assert len(restored.objects) == len(original.objects)

    for orig_obj, rest_obj in zip(original.objects, restored.objects, strict=True):
        assert rest_obj.object_id == orig_obj.object_id
        assert rest_obj.prompt == orig_obj.prompt
        assert rest_obj.placement.object_id == orig_obj.placement.object_id
        assert rest_obj.placement.yaw_deg == orig_obj.placement.yaw_deg
        assert rest_obj.placement.uniform_scale == orig_obj.placement.uniform_scale
        assert rest_obj.placement.translation == orig_obj.placement.translation
        assert rest_obj.placement.ground_contact == orig_obj.placement.ground_contact


def test_from_dict_wrong_schema_raises() -> None:
    d = _sample_layout().to_dict()
    d["schema"] = "wrong.schema/v99"
    with pytest.raises(ValueError, match="schema"):
        SceneLayout.from_dict(d)


def test_ground_contact_default_is_true() -> None:
    d = _sample_layout().to_dict()
    # Remove ground_contact from first placement to test default
    del d["objects"][0]["placement"]["ground_contact"]
    restored = SceneLayout.from_dict(d)
    assert restored.objects[0].placement.ground_contact is True


def test_empty_objects_list_round_trips() -> None:
    layout = SceneLayout(objects=[], up_axis="+Y", ground_y=0.0)
    d = layout.to_dict()
    restored = SceneLayout.from_dict(d)
    assert restored.objects == []


# ---------------------------------------------------------------------------
# write_json / read_json round-trip
# ---------------------------------------------------------------------------


def test_write_read_json_round_trip() -> None:
    original = _sample_layout()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "layout.json"
        original.write_json(path)
        assert path.exists()
        restored = SceneLayout.read_json(path)

    assert restored.up_axis == original.up_axis
    assert restored.ground_y == original.ground_y
    assert len(restored.objects) == len(original.objects)
    assert restored.objects[0].object_id == "chair"
    assert restored.objects[1].placement.ground_contact is False


def test_json_file_is_valid_json() -> None:
    layout = _sample_layout()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "layout.json"
        layout.write_json(path)
        raw = path.read_text(encoding="utf-8")
    # Must parse without error
    parsed = json.loads(raw)
    assert parsed["schema"] == _SCHEMA


def test_json_round_trip_preserves_floats() -> None:
    """Floating-point values (ground_y, yaw_deg, etc.) survive JSON serialisation."""
    layout = SceneLayout(
        objects=[
            SceneObject(
                object_id="obj",
                prompt="test",
                placement=Placement(
                    object_id="obj",
                    yaw_deg=123.456,
                    uniform_scale=0.987654,
                    translation=(1.111, 2.222, 3.333),
                    ground_contact=True,
                ),
            )
        ],
        up_axis="+Y",
        ground_y=-7.89,
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "layout.json"
        layout.write_json(path)
        restored = SceneLayout.read_json(path)

    pl = restored.objects[0].placement
    assert abs(pl.yaw_deg - 123.456) < 1e-9
    assert abs(pl.uniform_scale - 0.987654) < 1e-9
    assert abs(pl.translation[0] - 1.111) < 1e-9
    assert abs(restored.ground_y - (-7.89)) < 1e-9
