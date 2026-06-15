"""CPU unit tests for the DTU loader geometry math (no torch/gsplat/PIL)."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from astel_gpu.dtu import (
    ObsMask,
    camera_center,
    decompose_projection,
    load_ply_points,
    look_at_convergence,
    points_above_plane,
    points_in_obsmask,
    read_pos_matrix,
    umeyama,
)


def _proper_rotation(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1.0
    return q


def test_decompose_projection_roundtrip() -> None:
    k = np.array([[2600.0, 2.0, 800.0], [0.0, 2620.0, 600.0], [0.0, 0.0, 1.0]])
    rot = _proper_rotation(1)
    t = np.array([-533936.0, 23434.0, 2254.0])
    p = k @ np.hstack([rot, t[:, None]])

    k_rec, rot_rec, t_rec = decompose_projection(p)
    assert np.allclose(k_rec, k, rtol=1e-6, atol=1e-4)
    assert np.allclose(rot_rec, rot, atol=1e-6)
    assert np.allclose(t_rec, t, rtol=1e-6, atol=1e-3)
    assert np.isclose(np.linalg.det(rot_rec), 1.0)
    assert k_rec[2, 2] == 1.0


def test_decompose_projection_real_dtu_matrix() -> None:
    # pos_001.txt for scan1 (verified values).
    p = np.array(
        [
            [2607.429996, -3.844898, 1498.178098, -533936.661373],
            [-192.076910, 2862.552532, 681.798177, 23434.686572],
            [-0.241605, -0.030951, 0.969881, 22.540121],
        ]
    )
    k, rot, t = decompose_projection(p)
    assert np.isclose(np.linalg.det(rot), 1.0, atol=1e-6)  # proper rotation
    assert k[0, 0] > 1000 and k[1, 1] > 1000  # plausible focal lengths (px)
    # P is already normalized (last row unit-ish), so K[R|t] reconstructs it.
    reconstructed = k @ np.hstack([rot, t[:, None]])
    assert np.allclose(reconstructed, p, atol=1e-3)


def test_read_pos_matrix(tmp_path: Path) -> None:
    f = tmp_path / "pos_001.txt"
    f.write_text("1 2 3 4\n5 6 7 8\n9 10 11 12\n")
    p = read_pos_matrix(f)
    assert p.shape == (3, 4)
    assert p[1, 2] == 7.0


def test_camera_center_and_convergence() -> None:
    target = np.array([10.0, -5.0, 3.0])
    centres = np.array(
        [[110.0, -5.0, 3.0], [10.0, 95.0, 3.0], [10.0, -5.0, 103.0], [-90.0, -5.0, 3.0]]
    )
    rots, ts = [], []
    up = np.array([0.0, 0.0, 1.0])
    for c in centres:
        forward = target - c
        forward /= np.linalg.norm(forward)
        use_up = up if abs(forward @ up) < 0.99 else np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, use_up)
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        rot = np.stack([right, down, forward])  # rows = camera axes (world->cam)
        rots.append(rot)
        ts.append(-rot @ c)

    # camera_center inverts (R, t) back to the world centre.
    assert np.allclose(camera_center(rots[0], ts[0]), centres[0], atol=1e-6)
    # convergence of optical axes recovers the common look-at target.
    recovered = look_at_convergence(np.stack(rots), np.stack(ts))
    assert np.allclose(recovered, target, atol=1e-4)


def _write_binary_ply(path: Path, xyz: list[tuple[float, float, float]]) -> None:
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(xyz)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float nx\nproperty float ny\nproperty float nz\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        for x, y, z in xyz:
            f.write(struct.pack("<6f3B", x, y, z, 0.0, 1.0, 0.0, 10, 20, 30))


def test_load_ply_points_binary(tmp_path: Path) -> None:
    pts = [(1.0, 2.0, 3.0), (-4.5, 5.5, -6.0), (100.0, 0.0, 50.0)]
    f = tmp_path / "gt.ply"
    _write_binary_ply(f, pts)
    out = load_ply_points(f)
    assert out.shape == (3, 3)
    assert np.allclose(out, np.asarray(pts))


def test_umeyama_recovers_similarity() -> None:
    rng = np.random.default_rng(3)
    src = rng.standard_normal((30, 3)) * 100.0
    rot_true = _proper_rotation(7)
    s_true = 2.5
    t_true = np.array([10.0, -5.0, 3.0])
    dst = (s_true * (rot_true @ src.T)).T + t_true

    s, rot, t, rmse = umeyama(src, dst)
    assert np.isclose(s, s_true, rtol=1e-6)
    assert np.allclose(rot, rot_true, atol=1e-6)
    assert np.allclose(t, t_true, atol=1e-4)
    assert rmse < 1e-6


def test_points_above_plane() -> None:
    plane = np.array([0.0, 0.0, 1.0, -5.0])  # z > 5 is above
    pts = np.array([[0, 0, 6], [0, 0, 4], [0, 0, 5.5]])
    assert points_above_plane(pts, plane).tolist() == [True, False, True]


def test_points_in_obsmask() -> None:
    mask = np.zeros((4, 4, 4), dtype=np.uint8)
    mask[1, 1, 1] = 1
    mask[2, 2, 2] = 1
    om = ObsMask(bb_min=np.zeros(3), res=1.0, mask=mask)
    pts = np.array(
        [[1, 1, 1], [2.1, 2.0, 2.0], [0, 0, 0], [10, 0, 0], [-1, 0, 0]]
    )
    # (1,1,1)=on; (2.1,2,2) rounds to (2,2,2)=on; (0,0,0)=off; rest out of bounds.
    assert points_in_obsmask(pts, om).tolist() == [True, True, False, False, False]


def test_load_ply_points_ascii(tmp_path: Path) -> None:
    f = tmp_path / "gt_ascii.ply"
    f.write_text(
        "ply\nformat ascii 1.0\nelement vertex 2\n"
        "property float x\nproperty float y\nproperty float z\nend_header\n"
        "1 2 3\n4 5 6\n"
    )
    out = load_ply_points(f)
    assert np.allclose(out, [[1, 2, 3], [4, 5, 6]])
