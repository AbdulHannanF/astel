"""Scene-layout schema — dataclasses + JSON serialisation.

Schema key: ``"astel.scene-layout/v0"``
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_SCHEMA = "astel.scene-layout/v0"


@dataclass
class Placement:
    """Rigid placement of one object in the scene.

    Attributes
    ----------
    object_id:
        Unique identifier matching the parent :class:`SceneObject`.
    yaw_deg:
        Rotation about the up-axis (+Y) in degrees.
    uniform_scale:
        Uniform scale factor applied to the object before placement.
    translation:
        (tx, ty, tz) world-space translation applied after yaw and scale.
    ground_contact:
        When *True* the object is dropped onto the ground plane after
        the rigid transform (default ``True``).
    """

    object_id: str
    yaw_deg: float
    uniform_scale: float
    translation: tuple[float, float, float]
    ground_contact: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "yaw_deg": self.yaw_deg,
            "uniform_scale": self.uniform_scale,
            "translation": list(self.translation),
            "ground_contact": self.ground_contact,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Placement:
        tx, ty, tz = d["translation"]
        return cls(
            object_id=str(d["object_id"]),
            yaw_deg=float(d["yaw_deg"]),
            uniform_scale=float(d["uniform_scale"]),
            translation=(float(tx), float(ty), float(tz)),
            ground_contact=bool(d.get("ground_contact", True)),
        )


@dataclass
class SceneObject:
    """An object in the scene with its text description and placement.

    Attributes
    ----------
    object_id:
        Unique identifier (must match ``placement.object_id``).
    prompt:
        Natural-language description of the object (used for generation).
    placement:
        Rigid placement parameters.
    """

    object_id: str
    prompt: str
    placement: Placement

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "prompt": self.prompt,
            "placement": self.placement.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SceneObject:
        return cls(
            object_id=str(d["object_id"]),
            prompt=str(d["prompt"]),
            placement=Placement.from_dict(d["placement"]),
        )


@dataclass
class SceneLayout:
    """Complete scene layout: a list of objects + coordinate conventions.

    Attributes
    ----------
    objects:
        Ordered list of :class:`SceneObject` entries.  The ordering is
        *significant*: ``compose_scene`` matches objects by index.
    up_axis:
        Coordinate convention for "up".  Must be ``"+Y"`` (the Astel
        internal convention); other values are recorded but not interpreted.
    ground_y:
        Y-coordinate of the ground plane.
    """

    objects: list[SceneObject] = field(default_factory=list)
    up_axis: str = "+Y"
    ground_y: float = 0.0

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": _SCHEMA,
            "up_axis": self.up_axis,
            "ground_y": self.ground_y,
            "objects": [o.to_dict() for o in self.objects],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SceneLayout:
        schema = d.get("schema", "")
        if schema != _SCHEMA:
            raise ValueError(
                f"Unknown scene-layout schema: {schema!r}. Expected {_SCHEMA!r}."
            )
        return cls(
            objects=[SceneObject.from_dict(o) for o in d["objects"]],
            up_axis=str(d.get("up_axis", "+Y")),
            ground_y=float(d.get("ground_y", 0.0)),
        )

    def write_json(self, path: str | Path, *, indent: int = 2) -> None:
        """Serialise the layout to a JSON file at *path*."""
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=indent), encoding="utf-8"
        )

    @classmethod
    def read_json(cls, path: str | Path) -> SceneLayout:
        """Deserialise a layout from a JSON file at *path*."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)
