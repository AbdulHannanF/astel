"""Reader for COLMAP sparse-reconstruction binary models.

Parses ``cameras.bin``, ``images.bin`` and ``points3D.bin`` (COLMAP's native
little-endian binary format) into camera intrinsics, world-to-camera
extrinsics, and the sparse 3D point cloud (our L0 seed).

Convention: COLMAP stores, per registered image, the quaternion + translation
of the WORLD->CAMERA transform in OpenCV convention (+Z forward into the scene,
+X right, +Y down) -- exactly what ``gsplat.rasterization`` expects for
``viewmats`` and what :mod:`astel_gpu.cameras` already produces. So a COLMAP
pose drops straight into the existing refit loop with no axis juggling.

Lens distortion is dropped when building ``K`` (we use ``fx, fy, cx, cy`` only).
For the DTU smoke we run COLMAP's ``image_undistorter`` first, so images are
PINHOLE and there is nothing to drop; the lossy case (a distorted model used
directly) is surfaced via :attr:`ColmapModel.any_distortion` so callers can
refuse to report geometry numbers off uncorrected images.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

#: COLMAP ``model_id`` -> (model name, number of intrinsic params).
_CAMERA_MODELS: dict[int, tuple[str, int]] = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}

#: Models whose first param is a single shared focal length (fx == fy == f),
#: followed by (cx, cy). Everything else stores (fx, fy, cx, cy) up front.
_SINGLE_FOCAL = frozenset(
    {
        "SIMPLE_PINHOLE",
        "SIMPLE_RADIAL",
        "RADIAL",
        "FOV",
        "SIMPLE_RADIAL_FISHEYE",
        "RADIAL_FISHEYE",
    }
)

#: Models that carry no distortion params (K is exact, not a truncation).
_PINHOLE_EXACT = frozenset({"SIMPLE_PINHOLE", "PINHOLE"})


def _read(fid: BinaryIO, fmt: str) -> tuple[Any, ...]:
    """Read and unpack ``fmt`` (little-endian) from ``fid``; raise on short read."""
    size = struct.calcsize(fmt)
    data = fid.read(size)
    if len(data) != size:
        raise EOFError(f"expected {size} bytes for '{fmt}', got {len(data)}")
    return struct.unpack(fmt, data)


def _qvec_to_rotmat(qvec: tuple[float, float, float, float]) -> np.ndarray:
    """COLMAP quaternion ``(qw, qx, qy, qz)`` -> 3x3 world->camera rotation."""
    q = np.asarray(qvec, dtype=np.float64)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class Camera:
    """A COLMAP camera intrinsic (one per physical camera, shared across images)."""

    id: int
    model: str
    width: int
    height: int
    params: tuple[float, ...]

    @property
    def has_distortion(self) -> bool:
        """True if the model carries distortion params dropped from ``K``."""
        return self.model not in _PINHOLE_EXACT

    def k_matrix(self) -> np.ndarray:
        """Pinhole intrinsics ``K`` (3x3), distortion ignored."""
        if self.model in _SINGLE_FOCAL:
            f, cx, cy = self.params[0], self.params[1], self.params[2]
            fx = fy = f
        else:
            fx, fy, cx, cy = self.params[:4]
        return np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
        )


@dataclass(frozen=True)
class ImagePose:
    """A registered image's world->camera pose (OpenCV convention)."""

    id: int
    qvec: tuple[float, float, float, float]
    tvec: tuple[float, float, float]
    camera_id: int
    name: str

    def viewmat(self) -> np.ndarray:
        """4x4 world->camera matrix, gsplat-ready."""
        m = np.eye(4, dtype=np.float64)
        m[:3, :3] = _qvec_to_rotmat(self.qvec)
        m[:3, 3] = self.tvec
        return m


def read_cameras_binary(path: Path) -> dict[int, Camera]:
    """Parse ``cameras.bin`` -> {camera_id: Camera}."""
    cameras: dict[int, Camera] = {}
    with open(path, "rb") as f:
        (num,) = _read(f, "<Q")
        for _ in range(num):
            cam_id, model_id, width, height = _read(f, "<iiQQ")
            model_name, nparams = _CAMERA_MODELS[model_id]
            params = _read(f, "<" + "d" * nparams)
            cameras[cam_id] = Camera(
                id=cam_id,
                model=model_name,
                width=width,
                height=height,
                params=tuple(float(p) for p in params),
            )
    return cameras


def read_images_binary(path: Path) -> dict[int, ImagePose]:
    """Parse ``images.bin`` -> {image_id: ImagePose} (2D-point tracks skipped)."""
    images: dict[int, ImagePose] = {}
    with open(path, "rb") as f:
        (num,) = _read(f, "<Q")
        for _ in range(num):
            image_id, qw, qx, qy, qz, tx, ty, tz, camera_id = _read(f, "<idddddddi")
            name_bytes = bytearray()
            while True:
                c = f.read(1)
                if c in (b"\x00", b""):
                    break
                name_bytes += c
            (num_p2d,) = _read(f, "<Q")
            f.seek(24 * num_p2d, 1)  # each 2D point is (x, y, point3D_id) = 24 B
            images[image_id] = ImagePose(
                id=image_id,
                qvec=(float(qw), float(qx), float(qy), float(qz)),
                tvec=(float(tx), float(ty), float(tz)),
                camera_id=camera_id,
                name=name_bytes.decode("utf-8"),
            )
    return images


def read_points3d_binary(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse ``points3D.bin`` -> (xyz ``(P, 3)`` float64, rgb ``(P, 3)`` uint8)."""
    xyz: list[tuple[float, float, float]] = []
    rgb: list[tuple[int, int, int]] = []
    with open(path, "rb") as f:
        (num,) = _read(f, "<Q")
        for _ in range(num):
            _pid, x, y, z, r, g, b, _err = _read(f, "<QdddBBBd")
            (track_len,) = _read(f, "<Q")
            f.seek(8 * track_len, 1)  # each track elem is (image_id, p2d_idx) = 8 B
            xyz.append((float(x), float(y), float(z)))
            rgb.append((int(r), int(g), int(b)))
    if not xyz:
        return (
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, 3), dtype=np.uint8),
        )
    return np.asarray(xyz, dtype=np.float64), np.asarray(rgb, dtype=np.uint8)


@dataclass
class ColmapModel:
    """A loaded COLMAP sparse model: poses, intrinsics, and the L0 point cloud."""

    images: list[ImagePose]
    cameras: dict[int, Camera]
    points_xyz: np.ndarray
    points_rgb: np.ndarray

    def viewmats(self) -> np.ndarray:
        """``(N, 4, 4)`` world->camera matrices, ordered as :attr:`images`."""
        if not self.images:
            return np.zeros((0, 4, 4), dtype=np.float64)
        return np.stack([im.viewmat() for im in self.images])

    def k_matrices(self) -> np.ndarray:
        """``(N, 3, 3)`` intrinsics, ordered as :attr:`images`."""
        if not self.images:
            return np.zeros((0, 3, 3), dtype=np.float64)
        return np.stack([self.cameras[im.camera_id].k_matrix() for im in self.images])

    @property
    def any_distortion(self) -> bool:
        """True if any image's camera model carried distortion dropped from ``K``."""
        return any(self.cameras[im.camera_id].has_distortion for im in self.images)


def load_colmap_model(model_dir: Path) -> ColmapModel:
    """Load a COLMAP binary model directory (``cameras/images/points3D.bin``).

    Images are returned sorted by filename for deterministic ordering across
    runs (COLMAP's on-disk order follows registration, which is not stable).
    """
    model_dir = Path(model_dir)
    cameras = read_cameras_binary(model_dir / "cameras.bin")
    images_map = read_images_binary(model_dir / "images.bin")
    xyz, rgb = read_points3d_binary(model_dir / "points3D.bin")
    images = sorted(images_map.values(), key=lambda im: im.name)
    return ColmapModel(images=images, cameras=cameras, points_xyz=xyz, points_rgb=rgb)
