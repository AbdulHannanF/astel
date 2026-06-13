from __future__ import annotations

import gzip
import struct
from pathlib import Path

import numpy as np

from astel_splat_io.cloud import SplatCloud
from astel_splat_io.spz import (
    _HEADER_STRUCT,
    FRACTIONAL_BITS,
    NGSP_MAGIC,
    SPZ_VERSION,
    read_spz,
    write_spz,
)

# Quantization tolerances derived from the SPZ v3 format (FORMATS.md).
POSITION_TOL = 1.0 / (1 << FRACTIONAL_BITS)  # 24-bit fixed point, 12 fractional bits
SCALE_TOL = 1.0 / 16.0  # uint8((log_scale + 10) * 16)
ALPHA_TOL = 1.0 / 255.0 * 1.5  # uint8(sigmoid(opacity) * 255), compared in alpha-space
COLOR_TOL = 1.0 / (0.15 * 255.0)  # uint8(f_dc * 0.15 * 255 + 0.5*255)
ROTATION_COMPONENT_TOL = (0.70710678118654752440 / 511.0) * 2.0  # smallest-three step


def _quat_to_rotation_matrix(q_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = q_wxyz
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def test_spz_header_magic_and_version(tmp_path: Path, small_cloud: SplatCloud) -> None:
    out = tmp_path / "cloud.spz"
    write_spz(small_cloud, out)

    payload = gzip.decompress(out.read_bytes())
    magic, version, num_points, sh_degree, fractional_bits, flags, reserved = (
        _HEADER_STRUCT.unpack_from(payload, 0)
    )

    assert magic == NGSP_MAGIC == 0x5053474E
    assert version == SPZ_VERSION == 3
    assert num_points == small_cloud.count
    assert sh_degree == 0
    assert fractional_bits == FRACTIONAL_BITS == 12
    assert flags == 0
    assert reserved == 0
    assert struct.calcsize("<IIIBBBB") == _HEADER_STRUCT.size == 16


def test_spz_round_trip_within_tolerance(
    tmp_path: Path, small_cloud: SplatCloud
) -> None:
    out = tmp_path / "cloud.spz"
    write_spz(small_cloud, out)
    loaded = read_spz(out)

    assert loaded.count == small_cloud.count

    np.testing.assert_allclose(
        loaded.positions, small_cloud.positions, atol=POSITION_TOL * 1.5
    )
    np.testing.assert_allclose(
        loaded.log_scales, small_cloud.log_scales, atol=SCALE_TOL * 1.5
    )
    # Opacity is quantized in sigmoid (alpha) space, not logit space; the
    # logit's derivative blows up near 0/1 so compare alpha directly.
    alpha_in = 1.0 / (1.0 + np.exp(-small_cloud.opacity.astype(np.float64)))
    alpha_out = 1.0 / (1.0 + np.exp(-loaded.opacity.astype(np.float64)))
    np.testing.assert_allclose(alpha_out, alpha_in, atol=ALPHA_TOL)
    np.testing.assert_allclose(
        loaded.colors_dc, small_cloud.colors_dc, atol=COLOR_TOL * 1.5
    )

    # Rotations: SPZ "smallest three" packing is sign-ambiguous (q == -q for the
    # same rotation) and stores 10-bit magnitudes per component, so compare via
    # the resulting rotation matrices rather than raw quaternion components.
    for i in range(small_cloud.count):
        r_in = _quat_to_rotation_matrix(small_cloud.quats[i].astype(np.float64))
        r_out = _quat_to_rotation_matrix(loaded.quats[i].astype(np.float64))
        np.testing.assert_allclose(r_out, r_in, atol=ROTATION_COMPONENT_TOL * 4.0)
