"""KHR_gaussian_splatting glTF / GLB exporter — RC schema (Feb 2026).

Coordinate convention
---------------------
Input ``SplatCloud`` is assumed to be in **3DGS world space** (right-handed,
+Y-up as used by gsplat/COLMAP reconstructions).  glTF 2.0 is also
right-handed +Y-up −Z-forward, so **no position transform is applied** — the
coordinate frame is identical.  Quaternions are reordered from (w,x,y,z) to
(x,y,z,w) as required by glTF.

See ``docs/architecture/coordinate-conventions.md`` for the per-engine
transforms Unity and Unreal need on top.

Extension status
----------------
KHR_gaussian_splatting was published as a release-candidate on 2026-02-03
(Khronos press release).  Ratification was expected Q2 2026.  This exporter
targets the RC attribute layout:

  Mesh primitive (mode = POINTS):
    POSITION          VEC3  FLOAT    gaussian centers
    _ROTATION         VEC4  FLOAT    (x,y,z,w) unit quaternion
    _SCALE            VEC3  FLOAT    world-space σ (exp of log_scale)
    COLOR_0           VEC4  FLOAT    (r,g,b,α) in [0,1]

  extensions.KHR_gaussian_splatting: { "sh_degree": 0 }

Re-verify the ratified schema before shipping to external parties.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from astel_splat_io.cloud import SH_C0, SplatCloud

# GLB magic / chunk types (little-endian uint32)
_GLB_MAGIC = 0x46546C67  # 'glTF'
_GLB_VERSION = 2
_CHUNK_JSON = 0x4E4F534A  # 'JSON'
_CHUNK_BIN = 0x004E4942  # 'BIN\0'

# glTF accessor component types
_CT_FLOAT = 5126  # float32

# glTF accessor types → number of components
_VEC3 = "VEC3"
_VEC4 = "VEC4"

# Extension name (RC)
_EXT = "KHR_gaussian_splatting"


def _pad4(data: bytes, pad_byte: int = 0x20) -> bytes:
    """Return ``data`` zero/space-padded to the next 4-byte boundary."""
    rem = len(data) % 4
    if rem:
        data += bytes([pad_byte] * (4 - rem))
    return data


def _build_bin(cloud: SplatCloud) -> tuple[bytes, list[dict[str, object]]]:
    """Pack all accessor arrays into one binary buffer.

    Returns ``(bin_bytes, buffer_views)`` where ``buffer_views`` is a list of
    dicts ready for the glTF JSON.  Each view covers one accessor's data.
    """
    # --- Convert to display-space / glTF conventions ---
    # positions: no change (same coordinate frame)
    positions = cloud.positions.astype("<f4")  # (N,3) float32 LE

    # quaternions: (w,x,y,z) → (x,y,z,w), no coordinate flip needed
    q = cloud.quats  # (N,4): w x y z
    rotations = np.column_stack([q[:, 1], q[:, 2], q[:, 3], q[:, 0]]).astype(
        "<f4"
    )  # (N,4): x y z w

    # scales: exp(log_scale) → world-space σ
    scales = np.exp(cloud.log_scales).astype("<f4")  # (N,3)

    # colour + opacity: DC SH → RGB [0,1], sigmoid logit → α [0,1]
    rgb = np.clip(0.5 + SH_C0 * cloud.colors_dc, 0.0, 1.0).astype("<f4")  # (N,3)
    alpha = (1.0 / (1.0 + np.exp(-cloud.opacity))).astype("<f4")  # (N,)
    color_a = np.column_stack([rgb, alpha[:, None]]).astype("<f4")  # (N,4)

    arrays = [positions, rotations, scales, color_a]
    buffer_views: list[dict[str, object]] = []
    offset = 0
    chunks = []

    for arr in arrays:
        raw = arr.tobytes(order="C")
        chunks.append(raw)
        buffer_views.append(
            {
                "buffer": 0,
                "byteOffset": offset,
                "byteLength": len(raw),
                "target": 34962,  # ARRAY_BUFFER
            }
        )
        offset += len(raw)

    return b"".join(chunks), buffer_views


def _build_gltf(
    cloud: SplatCloud,
    buf_len: int,
    buffer_views: list[dict[str, object]],
) -> dict[str, object]:
    """Build the glTF JSON dict for a single-mesh Gaussian splat asset."""
    n = cloud.count

    # Accessors: one per attribute
    def _acc(bv_idx: int, count: int, comp_type: int, acc_type: str) -> dict[str, object]:  # noqa: E501
        return {
            "bufferView": bv_idx,
            "componentType": comp_type,
            "count": count,
            "type": acc_type,
        }

    # Bounding box for POSITION (required by glTF)
    pos = cloud.positions
    acc_pos = {
        **_acc(0, n, _CT_FLOAT, _VEC3),
        "min": pos.min(axis=0).tolist(),
        "max": pos.max(axis=0).tolist(),
    }

    accessors = [
        acc_pos,  # 0  POSITION
        _acc(1, n, _CT_FLOAT, _VEC4),  # 1  _ROTATION
        _acc(2, n, _CT_FLOAT, _VEC3),  # 2  _SCALE
        _acc(3, n, _CT_FLOAT, _VEC4),  # 3  COLOR_0
    ]

    return {
        "asset": {"version": "2.0", "generator": "astel-splat-io"},
        "extensionsUsed": [_EXT],
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [
            {
                "primitives": [
                    {
                        "mode": 0,  # POINTS
                        "attributes": {
                            "POSITION": 0,
                            "_ROTATION": 1,
                            "_SCALE": 2,
                            "COLOR_0": 3,
                        },
                        "extensions": {
                            _EXT: {"sh_degree": 0},
                        },
                    }
                ]
            }
        ],
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": buf_len}],
    }


def _encode_glb(json_dict: dict[str, object], bin_bytes: bytes) -> bytes:
    """Pack JSON + binary chunks into a GLB container."""
    json_raw = json.dumps(json_dict, separators=(",", ":")).encode("utf-8")
    json_padded = _pad4(json_raw, pad_byte=0x20)  # space-pad JSON
    bin_padded = _pad4(bin_bytes, pad_byte=0x00)  # zero-pad BIN

    json_chunk = struct.pack("<II", len(json_padded), _CHUNK_JSON) + json_padded
    bin_chunk = struct.pack("<II", len(bin_padded), _CHUNK_BIN) + bin_padded

    total = 12 + len(json_chunk) + len(bin_chunk)
    header = struct.pack("<III", _GLB_MAGIC, _GLB_VERSION, total)
    return header + json_chunk + bin_chunk


def write_gltf(cloud: SplatCloud, path: str | Path) -> int:
    """Write ``cloud`` to a binary GLB file with KHR_gaussian_splatting (RC).

    Returns the number of bytes written.

    Raises ``ValueError`` if the cloud is empty (0 splats), as a glTF file
    with no geometry is degenerate.
    """
    if cloud.count == 0:
        raise ValueError("cannot export an empty SplatCloud to glTF")

    bin_bytes, buffer_views = _build_bin(cloud)
    gltf = _build_gltf(cloud, len(bin_bytes), buffer_views)
    glb = _encode_glb(gltf, bin_bytes)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(glb)
    return len(glb)


def read_gltf(path: str | Path) -> SplatCloud:
    """Read a KHR_gaussian_splatting GLB file back into a :class:`SplatCloud`.

    Only supports files written by :func:`write_gltf` (single primitive,
    sh_degree 0, no compression).  Validates the extension is present.
    """
    data = Path(path).read_bytes()
    if len(data) < 12:
        raise ValueError("file too short to be a GLB")

    magic, version, total = struct.unpack_from("<III", data, 0)
    if magic != _GLB_MAGIC:
        raise ValueError(f"not a GLB file (magic 0x{magic:08X})")
    if version != 2:
        raise ValueError(f"unsupported GLB version {version}")
    if total != len(data):
        raise ValueError("GLB length header mismatch")

    offset = 12
    json_len, json_type = struct.unpack_from("<II", data, offset)
    if json_type != _CHUNK_JSON:
        raise ValueError("first chunk is not JSON")
    json_bytes = data[offset + 8 : offset + 8 + json_len]
    gltf = json.loads(json_bytes)
    offset += 8 + json_len

    bin_len, bin_type = struct.unpack_from("<II", data, offset)
    if bin_type != _CHUNK_BIN:
        raise ValueError("second chunk is not BIN")
    bin_data = data[offset + 8 : offset + 8 + bin_len]

    if _EXT not in gltf.get("extensionsUsed", []):
        raise ValueError(f"GLB does not declare {_EXT}")

    prim = gltf["meshes"][0]["primitives"][0]
    attrs = prim["attributes"]
    accessors = gltf["accessors"]
    views = gltf["bufferViews"]

    def _load(accessor_idx: int, ncols: int) -> np.ndarray:
        acc = accessors[accessor_idx]
        bv = views[acc["bufferView"]]
        start = bv["byteOffset"]
        nbytes = bv["byteLength"]
        n = acc["count"]
        arr = np.frombuffer(bin_data[start : start + nbytes], dtype="<f4")
        return arr.reshape(n, ncols)

    positions = _load(attrs["POSITION"], 3)
    rot_xyzw = _load(attrs["_ROTATION"], 4)  # (N,4): x y z w
    scales_world = _load(attrs["_SCALE"], 3)
    color_a = _load(attrs["COLOR_0"], 4)

    # Invert display-space conversions
    quats_wxyz = np.column_stack(
        [rot_xyzw[:, 3], rot_xyzw[:, 0], rot_xyzw[:, 1], rot_xyzw[:, 2]]
    ).astype(np.float32)
    log_scales = np.log(np.maximum(scales_world, 1e-10)).astype(np.float32)
    rgb = color_a[:, :3]
    alpha = color_a[:, 3]
    colors_dc = ((rgb - 0.5) / SH_C0).astype(np.float32)
    safe_alpha = np.clip(alpha, 1e-7, 1 - 1e-7)
    opacity_logit = np.log(safe_alpha / (1.0 - safe_alpha)).astype(np.float32)

    return SplatCloud(
        positions=np.ascontiguousarray(positions, dtype=np.float32),
        colors_dc=np.ascontiguousarray(colors_dc),
        opacity=np.ascontiguousarray(opacity_logit),
        log_scales=np.ascontiguousarray(log_scales),
        quats=np.ascontiguousarray(quats_wxyz),
    )
