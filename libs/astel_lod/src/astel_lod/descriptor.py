"""LOD descriptor: the ``astel.lod/v0`` sidecar manifest.

The descriptor records which files hold each LOD tier and how many Gaussians
each contains.  It is persisted as a JSON file next to the tier ``.ply`` /
``.spz`` files in the ``.astel`` package.

Schema ``astel.lod/v0``
-----------------------
::

    {
      "schema": "astel.lod/v0",
      "tiers": [
        {"name": "lowpoly",   "count": 100000,  "file": "lod_lowpoly.ply"},
        {"name": "standard",  "count": 1000000, "file": "lod_standard.ply"},
        {"name": "cinematic", "count": 5000000, "file": "lod_cinematic.ply"}
      ]
    }

Tiers are always stored sorted by ``count`` ascending (smallest tier first),
matching the streaming order: a client requests tiers in ascending count order
and upgrades incrementally.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_SCHEMA = "astel.lod/v0"


def build_lod_descriptor(tiers: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a validated ``astel.lod/v0`` descriptor dict.

    Parameters
    ----------
    tiers:
        List of tier dicts, each with keys ``"name"`` (str), ``"count"``
        (int), and ``"file"`` (str).  Order does not matter; the result is
        sorted by ``count`` ascending.

    Returns
    -------
    dict
        ``{"schema": "astel.lod/v0", "tiers": [...]}`` with tiers sorted by
        ``count`` ascending.

    Raises
    ------
    ValueError
        If ``counts`` are not strictly increasing after sorting (i.e. two tiers
        have the same count, or a duplicate was supplied).
    """
    sorted_tiers = sorted(tiers, key=lambda t: int(t["count"]))

    counts = [int(t["count"]) for t in sorted_tiers]
    for i in range(1, len(counts)):
        if counts[i] <= counts[i - 1]:
            msg = (
                f"LOD tier counts must be strictly increasing; "
                f"got {counts[i - 1]} then {counts[i]} "
                f"(tiers: {[t['name'] for t in sorted_tiers]})"
            )
            raise ValueError(msg)

    return {
        "schema": _SCHEMA,
        "tiers": [
            {"name": str(t["name"]), "count": int(t["count"]), "file": str(t["file"])}
            for t in sorted_tiers
        ],
    }


def write_descriptor(desc: dict[str, Any], path: str | Path) -> None:
    """Serialise ``desc`` to a JSON file at ``path``.

    Parameters
    ----------
    desc:
        A descriptor as returned by :func:`build_lod_descriptor`.
    path:
        Destination file path.  Parent directory must exist.
    """
    Path(path).write_text(json.dumps(desc, indent=2), encoding="utf-8")


def read_descriptor(path: str | Path) -> dict[str, Any]:
    """Read a JSON LOD descriptor from ``path``.

    Parameters
    ----------
    path:
        Path to a ``.json`` descriptor file.

    Returns
    -------
    dict
        The parsed descriptor.

    Raises
    ------
    ValueError
        If the file does not contain a valid ``astel.lod/v0`` schema field.
    """
    raw: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("schema") != _SCHEMA:
        msg = f"Expected schema {_SCHEMA!r}, got {raw.get('schema')!r} in {path}"
        raise ValueError(msg)
    return raw
