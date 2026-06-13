"""Niantic SPZ encoder/decoder (format version 3).

Spec source (verified 2026-06-13): Niantic Labs' open-source ``spz`` repo,
https://github.com/nianticlabs/spz — MIT License (Copyright 2025 Niantic
Labs, Copyright 2025 Adobe Inc.). Read directly from
``src/cc/load-spz.h`` and ``src/cc/load-spz.cc`` at HEAD.

As of the verification date the repo's *current default* serialised
container (``serializePackedGaussians`` / ``deserializePackedGaussians``,
used by ``saveSpz``/``loadSpz``) is still the **legacy 16-byte header +
single gzip stream** format, even though ``LATEST_SPZ_HEADER_VERSION == 4``
(a newer 32-byte-header, multi-stream ZSTD container exists in the same repo
for a different, TOC-based code path but is not what ``saveSpz``/``loadSpz``
emit). We implement the gzip container at **version 3** — the version that
introduced "smallest-three" quaternion packing and is the most recent
gzip-container version, while staying within a pure-Python + ``zlib``
implementation (no ZSTD dependency).

Header (16 bytes, little-endian)::

    uint32 magic           = 0x5053474e  ("NGSP")
    uint32 version         = 3
    uint32 num_points
    uint8  sh_degree       = 0  (SplatCloud carries no SH-rest)
    uint8  fractional_bits = 12
    uint8  flags           = 0
    uint8  reserved        = 0

Followed by gzip-compressed attribute streams, concatenated in this order
(``serializePackedGaussians``):

    positions (3 bytes/component, 24-bit fixed point, fractional_bits=12)
    alphas    (1 byte,  sigmoid(alpha) * 255)
    colors    (3 bytes, f_dc * (0.15*255) + 0.5*255)
    scales    (3 bytes, (log_scale + 10) * 16)
    rotations (4 bytes, "smallest three" quaternion packing)
    sh        (0 bytes; sh_degree == 0)

Quantization tolerances used by the round-trip test follow directly from the
step sizes above: positions to 1/4096 world units, scales to 1/16 in
log-space, colors/alpha to 1/255, rotation components to within the
smallest-three scheme's ~1/362 (sqrt(1/2)/511) resolution.
"""

from __future__ import annotations

import gzip
import struct
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from astel_splat_io.cloud import SplatCloud

NGSP_MAGIC = 0x5053474E
SPZ_VERSION = 3
FRACTIONAL_BITS = 12
COLOR_SCALE = 0.15
SQRT1_2 = 0.70710678118654752440

_HEADER_STRUCT = struct.Struct(
    "<IIIBBBB"
)  # magic, version, numPoints, shDeg, fracBits, flags, reserved


def _to_uint8(x: NDArray[np.float64]) -> NDArray[np.uint8]:
    return np.clip(np.round(x), 0.0, 255.0).astype(np.uint8)


def _sigmoid(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return 1.0 / (1.0 + np.exp(-x))


def _inv_sigmoid(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.log(x / (1.0 - x))


def _pack_positions(positions: NDArray[np.float32]) -> bytes:
    scale = float(1 << FRACTIONAL_BITS)
    fixed = np.round(positions.astype(np.float64) * scale).astype(np.int64)
    # Clamp to signed 24-bit range to avoid overflow on wrap.
    fixed = np.clip(fixed, -(1 << 23), (1 << 23) - 1)
    fixed_u32 = (fixed.astype(np.int64) & 0xFFFFFF).astype(np.uint32)
    out = np.empty((fixed_u32.shape[0], 3, 3), dtype=np.uint8)
    out[:, :, 0] = (fixed_u32 & 0xFF).astype(np.uint8)
    out[:, :, 1] = ((fixed_u32 >> 8) & 0xFF).astype(np.uint8)
    out[:, :, 2] = ((fixed_u32 >> 16) & 0xFF).astype(np.uint8)
    return out.tobytes(order="C")


def _unpack_positions(
    data: bytes, count: int, fractional_bits: int
) -> NDArray[np.float32]:
    raw = np.frombuffer(data, dtype=np.uint8, count=count * 3 * 3).reshape(count, 3, 3)
    fixed_u32 = (
        raw[:, :, 0].astype(np.uint32)
        | (raw[:, :, 1].astype(np.uint32) << 8)
        | (raw[:, :, 2].astype(np.uint32) << 16)
    )
    # Sign-extend from 24 bits to 32 bits.
    sign_bit = fixed_u32 & 0x800000
    fixed_i32 = fixed_u32.astype(np.int64)
    fixed_i32 = np.where(sign_bit != 0, fixed_i32 - (1 << 24), fixed_i32)
    scale = 1.0 / (1 << fractional_bits)
    return (fixed_i32.astype(np.float64) * scale).astype(np.float32)


def _pack_scales(log_scales: NDArray[np.float32]) -> bytes:
    packed = _to_uint8((log_scales.astype(np.float64) + 10.0) * 16.0)
    return packed.tobytes(order="C")


def _unpack_scales(data: bytes, count: int) -> NDArray[np.float32]:
    raw = np.frombuffer(data, dtype=np.uint8, count=count * 3).reshape(count, 3)
    return (raw.astype(np.float64) / 16.0 - 10.0).astype(np.float32)


def _pack_alphas(opacity: NDArray[np.float32]) -> bytes:
    return _to_uint8(_sigmoid(opacity.astype(np.float64)) * 255.0).tobytes(order="C")


def _unpack_alphas(data: bytes, count: int) -> NDArray[np.float32]:
    raw = np.frombuffer(data, dtype=np.uint8, count=count)
    alpha = raw.astype(np.float64) / 255.0
    alpha = np.clip(alpha, 1e-6, 1.0 - 1e-6)
    return _inv_sigmoid(alpha).astype(np.float32)


def _pack_colors(colors_dc: NDArray[np.float32]) -> bytes:
    packed = _to_uint8(
        colors_dc.astype(np.float64) * (COLOR_SCALE * 255.0) + (0.5 * 255.0)
    )
    return packed.tobytes(order="C")


def _unpack_colors(data: bytes, count: int) -> NDArray[np.float32]:
    raw = np.frombuffer(data, dtype=np.uint8, count=count * 3).reshape(count, 3)
    return (((raw.astype(np.float64) / 255.0) - 0.5) / COLOR_SCALE).astype(np.float32)


def _pack_rotations_smallest_three(quats_wxyz: NDArray[np.float32]) -> bytes:
    """Pack (w, x, y, z) quaternions using SPZ's "smallest three" scheme.

    SPZ stores quaternions in (x, y, z, w) order internally; we accept
    SplatCloud's (w, x, y, z) and reorder before packing.
    """
    n = quats_wxyz.shape[0]
    w = quats_wxyz[:, 0].astype(np.float64)
    xyz = quats_wxyz[:, 1:4].astype(np.float64)
    q = np.concatenate([xyz, w[:, None]], axis=1)  # (x, y, z, w)
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    norm = np.where(norm == 0.0, 1.0, norm)
    q = q / norm

    out = np.zeros((n, 4), dtype=np.uint8)
    c_mask = (1 << 9) - 1
    for i in range(n):
        row = q[i]
        i_largest = int(np.argmax(np.abs(row)))
        negate = row[i_largest] < 0.0
        comp = i_largest
        for j in range(4):
            if j == i_largest:
                continue
            negbit = int((row[j] < 0.0) ^ negate)
            mag = int(round((c_mask) * (abs(row[j]) / SQRT1_2)))
            mag = min(mag, c_mask)
            comp = (comp << 10) | (negbit << 9) | mag
        out[i, 0] = comp & 0xFF
        out[i, 1] = (comp >> 8) & 0xFF
        out[i, 2] = (comp >> 16) & 0xFF
        out[i, 3] = (comp >> 24) & 0xFF
    return out.tobytes(order="C")


def _unpack_rotations_smallest_three(data: bytes, count: int) -> NDArray[np.float32]:
    """Unpack SPZ "smallest three" quaternions to SplatCloud's (w, x, y, z)."""
    raw = np.frombuffer(data, dtype=np.uint8, count=count * 4).reshape(count, 4)
    comp = (
        raw[:, 0].astype(np.uint32)
        | (raw[:, 1].astype(np.uint32) << 8)
        | (raw[:, 2].astype(np.uint32) << 16)
        | (raw[:, 3].astype(np.uint32) << 24)
    )
    c_mask = (1 << 9) - 1

    out_xyzw = np.zeros((count, 4), dtype=np.float64)
    i_largest = (comp >> 30).astype(np.int64)
    cur = comp.copy()
    sum_squares = np.zeros(count, dtype=np.float64)
    for i in (3, 2, 1, 0):
        active = i_largest != i
        mag = (cur & c_mask).astype(np.float64)
        negbit = ((cur >> 9) & 0x1).astype(np.float64)
        val = SQRT1_2 * mag / float(c_mask)
        val = np.where(negbit == 1.0, -val, val)
        out_xyzw[:, i] = np.where(active, val, out_xyzw[:, i])
        sum_squares += np.where(active, val * val, 0.0)
        # The C++ reference only shifts `comp` right by 10 when i != i_largest
        # (the largest-component slot consumes no bits and is skipped
        # entirely in the pack/unpack loops). Shifting unconditionally here
        # would desync bit alignment whenever i_largest != 3.
        cur = np.where(active, cur >> 10, cur)

    largest_val = np.sqrt(np.clip(1.0 - sum_squares, 0.0, None))
    for i in range(4):
        mask = i_largest == i
        out_xyzw[mask, i] = largest_val[mask]

    # (x, y, z, w) -> (w, x, y, z)
    out_wxyz = np.empty((count, 4), dtype=np.float32)
    out_wxyz[:, 0] = out_xyzw[:, 3]
    out_wxyz[:, 1:4] = out_xyzw[:, 0:3]
    return out_wxyz


def write_spz(cloud: SplatCloud, path: str | Path) -> int:
    """Encode ``cloud`` as an SPZ v3 (gzip) file. Returns bytes written."""
    header = _HEADER_STRUCT.pack(
        NGSP_MAGIC, SPZ_VERSION, cloud.count, 0, FRACTIONAL_BITS, 0, 0
    )
    body = b"".join(
        [
            _pack_positions(cloud.positions),
            _pack_alphas(cloud.opacity),
            _pack_colors(cloud.colors_dc),
            _pack_scales(cloud.log_scales),
            _pack_rotations_smallest_three(cloud.quats),
            b"",  # sh: shDegree == 0
        ]
    )
    payload = header + body
    compressed = gzip.compress(payload, compresslevel=9)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(compressed)
    return len(compressed)


def read_spz(path: str | Path) -> SplatCloud:
    """Decode an SPZ v1-v3 (gzip) file into a :class:`SplatCloud`.

    SH coefficients (degree > 0) are discarded; ``SplatCloud`` carries band-0
    DC colour only.
    """
    compressed = Path(path).read_bytes()
    payload = gzip.decompress(compressed)

    magic, version, num_points, sh_degree, fractional_bits, _flags, _reserved = (
        _HEADER_STRUCT.unpack_from(payload, 0)
    )
    if magic != NGSP_MAGIC:
        raise ValueError(f"bad SPZ magic: 0x{magic:08x}")
    if version < 1 or version > 4:
        raise ValueError(f"unsupported SPZ version: {version}")

    offset = _HEADER_STRUCT.size
    n = num_points

    uses_float16 = version == 1
    if uses_float16:
        raise NotImplementedError("SPZ v1 float16 positions are not supported")

    pos_bytes = n * 3 * 3
    positions = _unpack_positions(
        payload[offset : offset + pos_bytes], n, fractional_bits
    )
    offset += pos_bytes

    alpha_bytes = n
    opacity = _unpack_alphas(payload[offset : offset + alpha_bytes], n)
    offset += alpha_bytes

    color_bytes = n * 3
    colors_dc = _unpack_colors(payload[offset : offset + color_bytes], n)
    offset += color_bytes

    scale_bytes = n * 3
    log_scales = _unpack_scales(payload[offset : offset + scale_bytes], n)
    offset += scale_bytes

    uses_smallest_three = version >= 3
    rot_bytes = n * (4 if uses_smallest_three else 3)
    if uses_smallest_three:
        quats = _unpack_rotations_smallest_three(
            payload[offset : offset + rot_bytes], n
        )
    else:
        raise NotImplementedError(
            "SPZ v2 'first three' rotation packing is not supported"
        )
    offset += rot_bytes

    # sh: discarded if present (sh_degree > 0).
    del sh_degree

    return SplatCloud(
        positions=positions,
        colors_dc=colors_dc,
        opacity=opacity,
        log_scales=log_scales,
        quats=quats,
    )
