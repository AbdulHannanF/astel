"""Provenance sidecar (`*.astl.json` + companion `.bin`) per manifest-v0 section 11.3.

For bare `.spz`/`.sog` exports, the splat file stays standard; a companion
`*.astl.json` sidecar carries the manifest subset (provenance accessor,
scale, quality summary, L4/L6 refs). The provenance buffer is `SCALAR`
`UNORM8` by default (`q = round(p * 255)`, `p = q / 255`), tightly packed,
`count` = splat count, index-aligned to the exported splat order
(manifest-v0 section 5.2-5.4).

If an exporter reorders splats (e.g. an SPZ/SOG spatial sort), it MUST apply
the same permutation to the provenance buffer before writing the sidecar —
see :meth:`astel_splat_io.cloud.SplatCloud.reordered` and the
``permutation`` parameter below.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

UNORM8_COMPONENT_TYPE = "UNORM8"


def encode_provenance_u8(provenance: NDArray[np.floating[Any]]) -> NDArray[np.uint8]:
    """Encode float provenance in [0, 1] to UNORM8: ``q = round(p * 255)``."""
    if np.any((provenance < 0.0) | (provenance > 1.0)):
        raise ValueError("provenance values must be in [0, 1]")
    return np.clip(np.round(provenance * 255.0), 0, 255).astype(np.uint8)


def decode_provenance_u8(encoded: NDArray[np.uint8]) -> NDArray[np.float32]:
    """Decode UNORM8 provenance: ``p = q / 255``."""
    return (encoded.astype(np.float32) / 255.0).astype(np.float32)


def write_provenance_sidecar(
    provenance: NDArray[np.floating[Any]],
    splat_path: str | Path,
    sidecar_path: str | Path,
    *,
    layer: str = "l3",
    permutation: NDArray[np.intp] | None = None,
) -> tuple[int, int]:
    """Write the `*.astl.json` sidecar + companion `.bin` for ``splat_path``.

    ``provenance`` is one float in [0, 1] per gaussian, **in the splat's
    in-memory (pre-export) order**. If the exporter that produced
    ``splat_path`` reordered splats, pass the same ``permutation`` it applied
    (e.g. via :meth:`SplatCloud.reordered`'s ``order`` argument) so the
    provenance buffer stays index-aligned to the exported splat order
    (manifest-v0 section 5.4 — reordering geometry without reordering
    provenance is a CI-failing condition).

    Returns ``(sidecar_bytes_written, bin_bytes_written)``.

    The companion `.bin` is written alongside ``sidecar_path`` with the same
    stem and a ``.provenance.bin`` suffix, e.g. ``asset.astl.json`` ->
    ``asset.provenance.bin``.
    """
    provenance = np.asarray(provenance, dtype=np.float32)
    if permutation is not None:
        provenance = provenance[permutation]

    encoded = encode_provenance_u8(provenance)

    sidecar_path = Path(sidecar_path)
    bin_path = sidecar_path.with_name(
        sidecar_path.stem.split(".")[0] + ".provenance.bin"
    )

    sidecar: dict[str, Any] = {
        "format_version": "0.1.0",
        "splat_file": Path(splat_path).name,
        "provenance": {
            "semantic": "measured_vs_generated",
            "range": [0.0, 1.0],
            "convention": "1=measured, 0=generated",
            "precision": "u8",
            "channels": [
                {
                    "layer": layer,
                    "accessor": 0,
                    "count": int(provenance.shape[0]),
                }
            ],
            "export_carriers": {
                "spz_sidecar": True,
            },
        },
        "buffers": [
            {
                "uri": bin_path.name,
                "byte_length": int(encoded.nbytes),
            }
        ],
        "buffer_views": [
            {
                "buffer": 0,
                "byte_offset": 0,
                "byte_length": int(encoded.nbytes),
                "byte_stride": 1,
            }
        ],
        "accessors": [
            {
                "buffer_view": 0,
                "component_type": UNORM8_COMPONENT_TYPE,
                "type": "SCALAR",
                "count": int(provenance.shape[0]),
                "normalized": True,
            }
        ],
    }

    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_bytes = json.dumps(sidecar, indent=2).encode("utf-8")
    sidecar_path.write_bytes(sidecar_bytes)

    bin_path.parent.mkdir(parents=True, exist_ok=True)
    bin_bytes = encoded.tobytes()
    bin_path.write_bytes(bin_bytes)

    return len(sidecar_bytes), len(bin_bytes)


def read_provenance_sidecar(sidecar_path: str | Path) -> NDArray[np.float32]:
    """Read back the provenance buffer written by :func:`write_provenance_sidecar`."""
    sidecar_path = Path(sidecar_path)
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))

    accessor = sidecar["accessors"][0]
    if accessor["component_type"] != UNORM8_COMPONENT_TYPE:
        raise NotImplementedError(
            f"only UNORM8 provenance accessors are supported, got "
            f"{accessor['component_type']!r}"
        )
    count = int(accessor["count"])

    buffer_uri = sidecar["buffers"][0]["uri"]
    bin_path = sidecar_path.with_name(buffer_uri)
    encoded = np.frombuffer(bin_path.read_bytes(), dtype=np.uint8, count=count)
    return decode_provenance_u8(encoded)
