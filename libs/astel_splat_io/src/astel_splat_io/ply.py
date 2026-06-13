"""Binary little-endian INRIA-layout PLY — the archival master format.

Re-derived from ``pipelines/stub/make_sample_splat.py`` (the canonical
``SplatCloud`` parameterisation and PLY property order). This is the format
the 3DGS ecosystem (and Astel's Spark-based web viewer) expects::

    x y z
    f_dc_0 f_dc_1 f_dc_2          # SH band-0 (DC) colour, NOT 0..1 RGB
    opacity                       # logit; sigmoid() -> alpha at render
    scale_0 scale_1 scale_2       # log-scale; exp() -> world-space sigma
    rot_0 rot_1 rot_2 rot_3       # quaternion (w, x, y, z), normalised

``f_rest_*`` (higher-order SH) are intentionally omitted: band-0 only is valid
and keeps the file small. Readers that expect them default to zero.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from astel_splat_io.cloud import SplatCloud

# The PLY property order is load-bearing: many readers (incl. ours) key on it.
PLY_PROPERTIES: tuple[str, ...] = (
    "x",
    "y",
    "z",
    "f_dc_0",
    "f_dc_1",
    "f_dc_2",
    "opacity",
    "scale_0",
    "scale_1",
    "scale_2",
    "rot_0",
    "rot_1",
    "rot_2",
    "rot_3",
)


def _ply_header(count: int) -> bytes:
    lines = ["ply", "format binary_little_endian 1.0", f"element vertex {count}"]
    lines += [f"property float {name}" for name in PLY_PROPERTIES]
    lines.append("end_header")
    return ("\n".join(lines) + "\n").encode("ascii")


def cloud_to_ply_bytes(cloud: SplatCloud) -> bytes:
    """Serialise a :class:`SplatCloud` to binary-little-endian PLY bytes."""
    interleaved = np.concatenate(
        [
            cloud.positions,
            cloud.colors_dc,
            cloud.opacity[:, None],
            cloud.log_scales,
            cloud.quats,
        ],
        axis=1,
    ).astype("<f4")  # explicit little-endian float32

    if interleaved.shape[1] != len(PLY_PROPERTIES):
        raise AssertionError(
            f"column count {interleaved.shape[1]} != {len(PLY_PROPERTIES)} properties"
        )

    header = _ply_header(cloud.count)
    return header + interleaved.tobytes(order="C")


def write_ply(cloud: SplatCloud, path: str | Path) -> int:
    """Write ``cloud`` to ``path`` as binary PLY. Returns bytes written."""
    data = cloud_to_ply_bytes(cloud)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return len(data)


def _parse_ply_header(data: bytes) -> tuple[list[str], int, int]:
    """Parse a binary-little-endian PLY header.

    Returns ``(property_names, vertex_count, header_byte_length)``.
    """
    if not data.startswith(b"ply"):
        raise ValueError("not a PLY file (missing 'ply' magic)")

    end_marker = b"end_header\n"
    end_idx = data.find(end_marker)
    if end_idx == -1:
        raise ValueError("PLY header missing 'end_header'")
    header_len = end_idx + len(end_marker)

    header_text = data[:end_idx].decode("ascii")
    lines = header_text.splitlines()

    if "format binary_little_endian 1.0" not in lines:
        raise ValueError("only binary_little_endian PLY is supported")

    count = 0
    properties: list[str] = []
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "element" and parts[1] == "vertex":
            count = int(parts[2])
        elif parts[0] == "property":
            if parts[1] != "float":
                raise ValueError(f"unsupported property type: {line!r}")
            properties.append(parts[2])

    return properties, count, header_len


def read_ply(path: str | Path) -> SplatCloud:
    """Read a binary little-endian INRIA-layout PLY into a :class:`SplatCloud`.

    Requires (at minimum) the :data:`PLY_PROPERTIES` columns, in any order;
    additional columns (e.g. ``f_rest_*``) are ignored.
    """
    data = Path(path).read_bytes()
    properties, count, header_len = _parse_ply_header(data)

    stride = len(properties)
    payload = np.frombuffer(data, dtype="<f4", count=count * stride, offset=header_len)
    table = payload.reshape(count, stride)

    col = {name: idx for idx, name in enumerate(properties)}
    missing = [name for name in PLY_PROPERTIES if name not in col]
    if missing:
        raise ValueError(f"PLY missing required properties: {missing}")

    def cols(*names: str) -> np.ndarray:
        idxs = [col[name] for name in names]
        return np.ascontiguousarray(table[:, idxs], dtype=np.float32)

    return SplatCloud(
        positions=cols("x", "y", "z"),
        colors_dc=cols("f_dc_0", "f_dc_1", "f_dc_2"),
        opacity=np.ascontiguousarray(table[:, col["opacity"]], dtype=np.float32),
        log_scales=cols("scale_0", "scale_1", "scale_2"),
        quats=cols("rot_0", "rot_1", "rot_2", "rot_3"),
    )
