"""Structure + golden tests for the sample-splat PLY writer.

These guard the *contract* of the file (header text, field order, byte layout,
value ranges) rather than exact bytes — the procedural geometry may evolve, but
the INRIA field layout the viewer depends on must not silently break.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from make_sample_splat import (
    DEFAULT_SEED,
    PLY_PROPERTIES,
    SH_C0,
    SplatCloud,
    build_torus_knot,
    cloud_to_ply_bytes,
    default_output_path,
    write_ply,
)

SMALL = 2_000  # small clouds keep tests fast


@pytest.fixture
def cloud() -> SplatCloud:
    return build_torus_knot(count=SMALL, seed=DEFAULT_SEED)


def _split_header(data: bytes) -> tuple[list[str], int]:
    """Return (header lines, byte offset just past 'end_header\\n')."""
    marker = b"end_header\n"
    idx = data.index(marker) + len(marker)
    header_text = data[:idx].decode("ascii")
    return header_text.splitlines(), idx


def test_property_layout_is_inria_standard() -> None:
    assert PLY_PROPERTIES == (
        "x", "y", "z",
        "f_dc_0", "f_dc_1", "f_dc_2",
        "opacity",
        "scale_0", "scale_1", "scale_2",
        "rot_0", "rot_1", "rot_2", "rot_3",
    )
    assert len(PLY_PROPERTIES) == 14


def test_cloud_shapes_are_consistent(cloud: SplatCloud) -> None:
    assert cloud.count == SMALL
    assert cloud.positions.shape == (SMALL, 3)
    assert cloud.colors_dc.shape == (SMALL, 3)
    assert cloud.opacity.shape == (SMALL,)
    assert cloud.log_scales.shape == (SMALL, 3)
    assert cloud.quats.shape == (SMALL, 4)


def test_header_is_well_formed(cloud: SplatCloud) -> None:
    data = cloud_to_ply_bytes(cloud)
    lines, _ = _split_header(data)
    assert lines[0] == "ply"
    assert lines[1] == "format binary_little_endian 1.0"
    assert lines[2] == f"element vertex {SMALL}"
    prop_lines = [line for line in lines if line.startswith("property")]
    assert prop_lines == [f"property float {name}" for name in PLY_PROPERTIES]
    assert lines[-1] == "end_header"


def test_binary_body_size_matches_count(cloud: SplatCloud) -> None:
    data = cloud_to_ply_bytes(cloud)
    _, offset = _split_header(data)
    body = data[offset:]
    bytes_per_vertex = len(PLY_PROPERTIES) * 4  # float32
    assert len(body) == SMALL * bytes_per_vertex


def test_roundtrip_first_vertex_decodes(cloud: SplatCloud) -> None:
    data = cloud_to_ply_bytes(cloud)
    _, offset = _split_header(data)
    n = len(PLY_PROPERTIES)
    first = struct.unpack_from(f"<{n}f", data, offset)
    expected = np.concatenate(
        [
            cloud.positions[0],
            cloud.colors_dc[0],
            [cloud.opacity[0]],
            cloud.log_scales[0],
            cloud.quats[0],
        ]
    )
    np.testing.assert_allclose(first, expected, rtol=0, atol=1e-6)


def test_quaternions_are_normalised(cloud: SplatCloud) -> None:
    norms = np.linalg.norm(cloud.quats, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_scales_are_finite_log_values(cloud: SplatCloud) -> None:
    # log-scale: exp() must give sane, positive, bounded world sigmas.
    sigmas = np.exp(cloud.log_scales)
    assert np.all(np.isfinite(sigmas))
    assert np.all(sigmas > 0.0)
    assert np.all(sigmas < 1.0)


def test_opacity_logits_map_to_valid_alpha(cloud: SplatCloud) -> None:
    alpha = 1.0 / (1.0 + np.exp(-cloud.opacity))
    assert np.all(alpha > 0.0)
    assert np.all(alpha < 1.0)


def test_dc_colour_inverts_to_displayable_rgb(cloud: SplatCloud) -> None:
    rgb = 0.5 + SH_C0 * cloud.colors_dc
    # Allow a hair outside [0,1] from added noise, but it must be near-display.
    assert rgb.min() > -0.05
    assert rgb.max() < 1.05


def test_determinism_same_seed_same_bytes() -> None:
    a = cloud_to_ply_bytes(build_torus_knot(count=SMALL, seed=123))
    b = cloud_to_ply_bytes(build_torus_knot(count=SMALL, seed=123))
    assert a == b


def test_different_seed_differs() -> None:
    a = cloud_to_ply_bytes(build_torus_knot(count=SMALL, seed=1))
    b = cloud_to_ply_bytes(build_torus_knot(count=SMALL, seed=2))
    assert a != b


def test_invalid_shapes_rejected() -> None:
    with pytest.raises(ValueError):
        SplatCloud(
            positions=np.zeros((10, 3), dtype=np.float32),
            colors_dc=np.zeros((9, 3), dtype=np.float32),  # mismatched N
            opacity=np.zeros((10,), dtype=np.float32),
            log_scales=np.zeros((10, 3), dtype=np.float32),
            quats=np.zeros((10, 4), dtype=np.float32),
        )


def test_write_ply_roundtrips_to_disk(tmp_path: Path, cloud: SplatCloud) -> None:
    out = tmp_path / "sub" / "sample.ply"
    size = write_ply(cloud, out)
    assert out.exists()
    assert size == out.stat().st_size
    assert out.read_bytes()[:3] == b"ply"


def test_checked_in_sample_exists_and_is_valid() -> None:
    """The committed sample must be present and structurally valid."""
    sample = default_output_path()
    assert sample.exists(), f"missing checked-in sample at {sample}"
    data = sample.read_bytes()
    lines, offset = _split_header(data)
    assert lines[0] == "ply"
    # element vertex N
    vertex_line = next(line for line in lines if line.startswith("element vertex"))
    count = int(vertex_line.split()[-1])
    assert count > 10_000  # tasteful density
    body = data[offset:]
    assert len(body) == count * len(PLY_PROPERTIES) * 4
