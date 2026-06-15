"""CPU unit tests for the COLMAP binary-model reader.

Builds tiny ``cameras.bin`` / ``images.bin`` / ``points3D.bin`` buffers by hand
(mirroring COLMAP's documented little-endian layout) and round-trips them
through :mod:`astel_gpu.colmap_io`. No gsplat/CUDA import -> runs on any host.
"""

from __future__ import annotations

import math
import struct
from pathlib import Path

import numpy as np

from astel_gpu.colmap_io import load_colmap_model

# (image_id, qvec(wxyz), tvec, camera_id, name, n_2d_points)
_ImageSpec = tuple[
    int, tuple[float, float, float, float], tuple[float, float, float], int, str, int
]
# (xyz, rgb, track_length)
_PointSpec = tuple[tuple[float, float, float], tuple[int, int, int], int]


def _write_cameras(
    path: Path, cams: list[tuple[int, int, int, int, tuple[float, ...]]]
) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(cams)))
        for cam_id, model_id, w, h, params in cams:
            f.write(struct.pack("<iiQQ", cam_id, model_id, w, h))
            f.write(struct.pack("<" + "d" * len(params), *params))


def _write_images(path: Path, imgs: list[_ImageSpec]) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(imgs)))
        for image_id, qvec, tvec, camera_id, name, n_p2d in imgs:
            f.write(struct.pack("<idddddddi", image_id, *qvec, *tvec, camera_id))
            f.write(name.encode("utf-8") + b"\x00")
            f.write(struct.pack("<Q", n_p2d))
            for i in range(n_p2d):
                f.write(struct.pack("<ddq", float(i), float(i), -1))


def _write_points(path: Path, pts: list[_PointSpec]) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(pts)))
        for (x, y, z), (r, g, b), track in pts:
            f.write(struct.pack("<QdddBBBd", 0, x, y, z, r, g, b, 0.0))
            f.write(struct.pack("<Q", track))
            for i in range(track):
                f.write(struct.pack("<ii", 1, i))


def _build_model(tmp_path: Path) -> Path:
    _write_cameras(
        tmp_path / "cameras.bin",
        [
            (1, 1, 640, 480, (500.0, 600.0, 320.0, 240.0)),  # PINHOLE
            (2, 0, 200, 200, (400.0, 100.0, 110.0)),  # SIMPLE_PINHOLE
            (3, 2, 200, 200, (400.0, 100.0, 110.0, 0.01)),  # SIMPLE_RADIAL
        ],
    )
    # +90deg rotation about z as a quaternion (w, 0, 0, z), |q| = 1.
    s = math.sqrt(2.0) / 2.0
    _write_images(
        tmp_path / "images.bin",
        [
            # out-of-order names to exercise deterministic name sorting
            (10, (1.0, 0.0, 0.0, 0.0), (1.0, 2.0, 3.0), 1, "b_second.png", 2),
            (11, (s, 0.0, 0.0, s), (0.0, 0.0, 0.0), 3, "a_first.png", 0),
        ],
    )
    _write_points(
        tmp_path / "points3D.bin",
        [
            ((1.0, 2.0, 3.0), (10, 20, 30), 3),
            ((-4.0, -5.0, -6.0), (200, 100, 50), 1),
        ],
    )
    return tmp_path


def test_intrinsics_pinhole_and_single_focal(tmp_path: Path) -> None:
    model = load_colmap_model(_build_model(tmp_path))

    k_pinhole = model.cameras[1].k_matrix()
    assert k_pinhole[0, 0] == 500.0
    assert k_pinhole[1, 1] == 600.0
    assert k_pinhole[0, 2] == 320.0
    assert k_pinhole[1, 2] == 240.0

    k_simple = model.cameras[2].k_matrix()
    assert k_simple[0, 0] == 400.0  # fx == fy == f for a single-focal model
    assert k_simple[1, 1] == 400.0
    assert k_simple[0, 2] == 100.0
    assert k_simple[1, 2] == 110.0


def test_distortion_flag(tmp_path: Path) -> None:
    model = load_colmap_model(_build_model(tmp_path))
    assert model.cameras[1].has_distortion is False  # PINHOLE
    assert model.cameras[2].has_distortion is False  # SIMPLE_PINHOLE
    assert model.cameras[3].has_distortion is True  # SIMPLE_RADIAL
    assert model.any_distortion is True  # image 11 uses camera 3


def test_pose_identity_quaternion(tmp_path: Path) -> None:
    model = load_colmap_model(_build_model(tmp_path))
    # images sorted by name: a_first (id 11), b_second (id 10)
    assert [im.name for im in model.images] == ["a_first.png", "b_second.png"]

    b_second = model.images[1]
    vm = b_second.viewmat()
    assert np.allclose(vm[:3, :3], np.eye(3))
    assert np.allclose(vm[:3, 3], [1.0, 2.0, 3.0])
    assert np.allclose(vm[3], [0.0, 0.0, 0.0, 1.0])


def test_pose_z_rotation_quaternion(tmp_path: Path) -> None:
    model = load_colmap_model(_build_model(tmp_path))
    a_first = model.images[0]  # +90deg about z
    expected = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert np.allclose(a_first.viewmat()[:3, :3], expected, atol=1e-6)


def test_viewmats_and_k_stacks(tmp_path: Path) -> None:
    model = load_colmap_model(_build_model(tmp_path))
    assert model.viewmats().shape == (2, 4, 4)
    assert model.k_matrices().shape == (2, 3, 3)


def test_points_cloud(tmp_path: Path) -> None:
    model = load_colmap_model(_build_model(tmp_path))
    assert model.points_xyz.shape == (2, 3)
    assert model.points_rgb.shape == (2, 3)
    assert model.points_rgb.dtype == np.uint8
    assert np.allclose(model.points_xyz[0], [1.0, 2.0, 3.0])
    assert np.allclose(model.points_xyz[1], [-4.0, -5.0, -6.0])
    assert tuple(model.points_rgb[0]) == (10, 20, 30)
