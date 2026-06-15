"""Loader for the DTU MVS dataset (the M2 real-world geometry benchmark).

DTU ships, per view, a 3x4 camera projection matrix ``P = K[R|t]`` (``pos_NNN.txt``)
in MILLIMETRES, expressed in the SAME frame of reference as the structured-light
ground-truth scan (``Points/stl/stlNNN_total.ply``). So a gaussian cloud fit
with these poses lands directly in the GT metric frame -- the first real-world
Chamfer needs NO registration. See ``docs/research/DECISIONS.md`` (session 9).

This module is pure-numpy for the geometry math (pose parse, RQ decomposition,
GT-free object-centre estimation, PLY read) so it unit-tests on CPU; only
:func:`load_dtu_scan` pulls in torch + PIL to build render tensors. It imports
no gsplat, so it needs no MSVC launcher to test.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_FLIP = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])

_IMAGE_RE = re.compile(r"rect_(\d+)_")


def read_pos_matrix(path: Path) -> np.ndarray:
    """Read a DTU ``pos_NNN.txt`` -> ``(3, 4)`` projection matrix ``P``."""
    values = [float(x) for x in Path(path).read_text().split()]
    if len(values) != 12:
        raise ValueError(
            f"{path}: expected 12 numbers for a 3x4 matrix, got {len(values)}"
        )
    return np.asarray(values, dtype=np.float64).reshape(3, 4)


def decompose_projection(p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decompose ``P = K[R|t]`` -> ``(K, R, t)``.

    ``K`` is upper-triangular with positive diagonal and ``K[2, 2] == 1``; ``R``
    is a proper rotation (``det == +1``) giving world->camera; ``t`` is the
    world->camera translation (millimetres, DTU frame). OpenCV/COLMAP
    convention (+Z forward), matching what gsplat expects for ``viewmats``.
    """
    m = p[:, :3]
    # RQ decomposition of m via a flipped QR.
    a = _FLIP @ m
    q_, r_ = np.linalg.qr(a.T)
    k = _FLIP @ r_.T @ _FLIP
    rot = _FLIP @ q_.T

    # Force a positive diagonal on K (absorb sign flips into R).
    sign = np.diag(np.sign(np.diag(k)))
    k = k @ sign
    rot = sign @ rot

    # Force R to be a proper rotation (det +1); a global flip keeps K's
    # diagonal positive because it negates whole rows/columns consistently.
    if np.linalg.det(rot) < 0:
        rot = -rot
        k = -k
        sign2 = np.diag(np.sign(np.diag(k)))
        k = k @ sign2
        rot = sign2 @ rot

    k = k / k[2, 2]
    t = np.linalg.solve(k, p[:, 3])
    return k, rot, t


def camera_center(rot: np.ndarray, t: np.ndarray) -> np.ndarray:
    """World-space camera centre ``C = -R^T t`` for a world->camera ``(R, t)``."""
    centre: np.ndarray = -rot.T @ t
    return centre


def look_at_convergence(rots: np.ndarray, ts: np.ndarray) -> np.ndarray:
    """Estimate the object centre as the least-squares meet of camera axes.

    GT-free: uses only the camera poses. Each camera contributes its optical
    axis (a line through its centre along +Z-in-world); the returned point
    minimises the sum of squared distances to those lines -- a robust object
    centre for initialising gaussians without touching the ground truth.
    """
    eye = np.eye(3)
    a_acc = np.zeros((3, 3))
    b_acc = np.zeros(3)
    for rot, t in zip(rots, ts, strict=True):
        centre = camera_center(rot, t)
        axis = rot.T @ np.array([0.0, 0.0, 1.0])
        axis = axis / np.linalg.norm(axis)
        proj = eye - np.outer(axis, axis)
        a_acc += proj
        b_acc += proj @ centre
    return np.linalg.solve(a_acc, b_acc)


def load_ply_points(path: Path) -> np.ndarray:
    """Read XYZ vertices from a PLY (binary-LE or ASCII) -> ``(P, 3)`` float64.

    Parses the property list to locate x/y/z within each vertex record, so it
    handles DTU's ``float x,y,z, nx,ny,nz, uchar r,g,b`` layout (and others).
    """
    path = Path(path)
    _DTYPES = {
        "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
        "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
        "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
        "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
    }
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError(f"{path}: not a PLY file")
        fmt = ""
        n_vertices = 0
        props: list[tuple[str, str]] = []
        in_vertex = False
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: unexpected EOF in header")
            tokens = line.split()
            tag = tokens[0]
            if tag == b"format":
                fmt = tokens[1].decode()
            elif tag == b"element":
                in_vertex = tokens[1] == b"vertex"
                if in_vertex:
                    n_vertices = int(tokens[2])
            elif tag == b"property" and in_vertex:
                props.append((tokens[1].decode(), tokens[2].decode()))
            elif tag == b"end_header":
                break

        names = [name for _t, name in props]
        for axis in ("x", "y", "z"):
            if axis not in names:
                raise ValueError(f"{path}: vertex has no '{axis}' property")

        if fmt == "ascii":
            rows = []
            for _ in range(n_vertices):
                vals = f.readline().split()
                rows.append([float(vals[names.index(a)]) for a in ("x", "y", "z")])
            return np.asarray(rows, dtype=np.float64)

        if fmt != "binary_little_endian":
            raise ValueError(f"{path}: unsupported PLY format '{fmt}'")
        dtype = np.dtype([(name, _DTYPES[t]) for t, name in props])
        data = np.frombuffer(f.read(n_vertices * dtype.itemsize), dtype=dtype)
        xyz: np.ndarray = np.stack(
            [data["x"], data["y"], data["z"]], axis=1
        ).astype(np.float64)
        return xyz


def umeyama(
    src: np.ndarray, dst: np.ndarray, *, with_scale: bool = True
) -> tuple[float, np.ndarray, np.ndarray, float]:
    """Least-squares similarity ``dst ≈ s·R·src + t`` (Umeyama 1991).

    Returns ``(s, R, t, rmse)``. Used to align a scale-free COLMAP
    reconstruction's camera centres to DTU's metric GT centres, giving a pose
    accuracy (RMSE, mm) and the recovered metric scale of the SfM solution.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = src.shape[0]
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    xs, xd = src - mu_s, dst - mu_d
    cov = (xd.T @ xs) / n
    u, d, vt = np.linalg.svd(cov)
    s_corr = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s_corr[2, 2] = -1.0
    rot: np.ndarray = u @ s_corr @ vt
    if with_scale:
        var_s = float((xs**2).sum() / n)
        scale = float((d * np.diag(s_corr)).sum() / var_s)
    else:
        scale = 1.0
    trans: np.ndarray = mu_d - scale * (rot @ mu_s)
    aligned = (scale * (rot @ src.T)).T + trans
    rmse = float(np.sqrt(((aligned - dst) ** 2).sum(axis=1).mean()))
    return scale, rot, trans, rmse


@dataclass
class ObsMask:
    """DTU observable-region voxel mask: ``mask[idx]`` true => point is evaluated."""

    bb_min: np.ndarray  # (3,) millimetres
    res: float  # voxel size (mm)
    mask: np.ndarray  # (X, Y, Z) occupancy


def load_obsmask(path: Path) -> ObsMask:
    """Load ``ObsMask{scan}_10.mat`` (BB, Res, ObsMask)."""
    from scipy.io import loadmat

    m = loadmat(str(path))
    bb = np.asarray(m["BB"], dtype=np.float64)
    return ObsMask(
        bb_min=bb[0],
        res=float(np.asarray(m["Res"]).ravel()[0]),
        mask=np.asarray(m["ObsMask"]),
    )


def load_plane(path: Path) -> np.ndarray:
    """Load DTU ``Plane{scan}.mat`` -> ``(4,)`` plane coeffs ``[a, b, c, d]``."""
    from scipy.io import loadmat

    return np.asarray(loadmat(str(path))["P"], dtype=np.float64).ravel()


def points_in_obsmask(points: np.ndarray, obsmask: ObsMask) -> np.ndarray:
    """Boolean ``(N,)``: which ``points`` (mm) fall in an observable voxel.

    Mirrors the DTU eval: ``idx = round((p - BB_min) / Res)`` (their 1-based
    ``+1`` then ``[1, size]`` test is our 0-based ``[0, size)``), then look up
    ``ObsMask``; out-of-bounds points are not observable.
    """
    idx = np.round((points - obsmask.bb_min) / obsmask.res).astype(np.int64)
    dims = np.asarray(obsmask.mask.shape)
    valid = np.all((idx >= 0) & (idx < dims), axis=1)
    out = np.zeros(points.shape[0], dtype=bool)
    vi = idx[valid]
    out[valid] = obsmask.mask[vi[:, 0], vi[:, 1], vi[:, 2]] > 0
    return out


def points_above_plane(points: np.ndarray, plane: np.ndarray) -> np.ndarray:
    """Boolean ``(N,)``: which ``points`` are above the DTU ground plane."""
    above: np.ndarray = (points @ plane[:3] + plane[3]) > 0
    return above


@dataclass
class DtuScan:
    """A loaded DTU scan: render tensors + per-view poses (millimetre frame)."""

    viewmats: np.ndarray  # (V, 4, 4) world->camera
    ks: np.ndarray  # (V, 3, 3) intrinsics for the (downscaled) images
    images: np.ndarray  # (V, H, W, 3) float32 in [0, 1]
    width: int
    height: int
    object_center: np.ndarray  # (3,) GT-free estimate, millimetres


def load_dtu_scan(
    image_dir: Path, pos_dir: Path, *, downscale: int = 4
) -> DtuScan:
    """Load a DTU scan's light-3 images + matched poses, downscaled by ``downscale``.

    Image ``rect_NNN_*`` is matched to pose ``pos_NNN.txt``. Intrinsics are
    scaled to the downsampled resolution. Returns numpy arrays; the caller moves
    them to torch/CUDA (keeps this module gsplat-free and CPU-importable).
    """
    from PIL import Image

    image_dir = Path(image_dir)
    pos_dir = Path(pos_dir)
    image_paths = sorted(image_dir.glob("rect_*.png"))
    if not image_paths:
        raise ValueError(f"no rect_*.png images under {image_dir}")

    viewmats: list[np.ndarray] = []
    ks: list[np.ndarray] = []
    images: list[np.ndarray] = []
    rots: list[np.ndarray] = []
    ts: list[np.ndarray] = []
    width = height = 0

    for img_path in image_paths:
        match = _IMAGE_RE.search(img_path.name)
        if match is None:
            raise ValueError(f"cannot parse view index from {img_path.name}")
        view = match.group(1)
        p = read_pos_matrix(pos_dir / f"pos_{view}.txt")
        k, rot, t = decompose_projection(p)

        with Image.open(img_path) as im:
            rgb = im.convert("RGB")
        if downscale != 1:
            rgb = rgb.resize(
                (rgb.width // downscale, rgb.height // downscale),
                Image.Resampling.BILINEAR,
            )
        arr = np.asarray(rgb, dtype=np.float32) / 255.0
        height, width = arr.shape[0], arr.shape[1]

        k_scaled = k.copy()
        k_scaled[:2, :] /= downscale

        vm = np.eye(4)
        vm[:3, :3] = rot
        vm[:3, 3] = t
        viewmats.append(vm)
        ks.append(k_scaled)
        images.append(arr)
        rots.append(rot)
        ts.append(t)

    return DtuScan(
        viewmats=np.stack(viewmats),
        ks=np.stack(ks),
        images=np.stack(images),
        width=width,
        height=height,
        object_center=look_at_convergence(np.stack(rots), np.stack(ts)),
    )
