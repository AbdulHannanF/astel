"""Test helpers: synthetic motion generators for astel_dynamics tests."""

from __future__ import annotations

from typing import cast

import numpy as np


def static_cloud(n: int, seed: int = 0) -> np.ndarray:
    """Return *n* random Gaussian means in the unit cube ``[0,1)^3``.

    Parameters
    ----------
    n:
        Number of points.
    seed:
        NumPy random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Shape ``(N, 3)`` float32.
    """
    rng = np.random.default_rng(seed)
    return rng.random((n, 3)).astype(np.float32)


def rigid_rotation_motion(
    base: np.ndarray,
    n_frames: int,
    axis: np.ndarray | list[float],
    total_angle: float,
) -> np.ndarray:
    """Apply a global rotation that increases linearly per frame.

    Frame *f* applies a rotation of ``total_angle * (f+1) / n_frames`` radians
    around *axis* (normalised internally).

    Parameters
    ----------
    base:
        Rest-pose positions, shape ``(N, 3)``.
    n_frames:
        Number of output frames.
    axis:
        Rotation axis (need not be unit-length).
    total_angle:
        Total rotation in radians after the last frame.

    Returns
    -------
    np.ndarray
        Shape ``(F, N, 3)`` float32.
    """
    ax = np.asarray(axis, dtype=np.float64)
    ax = ax / np.linalg.norm(ax)

    frames = np.empty((n_frames, base.shape[0], 3), dtype=np.float32)
    for f in range(n_frames):
        angle = total_angle * (f + 1) / n_frames
        # Rodrigues rotation matrix
        c, s = np.cos(angle), np.sin(angle)
        K_mat = np.array(
            [
                [0, -ax[2], ax[1]],
                [ax[2], 0, -ax[0]],
                [-ax[1], ax[0], 0],
            ],
            dtype=np.float64,
        )
        R = c * np.eye(3) + s * K_mat + (1 - c) * np.outer(ax, ax)
        frames[f] = (base.astype(np.float64) @ R.T).astype(np.float32)
    return frames


def bend_motion(
    base: np.ndarray,
    n_frames: int,
    max_angle: float,
) -> np.ndarray:
    """Low-rank bend: rotation angle proportional to the x-coordinate.

    Each point is rotated around the Y-axis by
    ``max_angle * x * (f+1) / n_frames`` where x is normalised to ``[0, 1]``.
    This produces a bending deformation that is well-approximated by LBS with
    enough nodes.

    Parameters
    ----------
    base:
        Rest-pose positions, shape ``(N, 3)``.
    n_frames:
        Number of output frames.
    max_angle:
        Maximum rotation (radians) at x=1 on the last frame.

    Returns
    -------
    np.ndarray
        Shape ``(F, N, 3)`` float32.
    """
    x = base[:, 0].astype(np.float64)
    x_norm = (x - x.min()) / max(x.max() - x.min(), 1e-12)  # [0, 1]

    frames = np.empty((n_frames, base.shape[0], 3), dtype=np.float32)
    base_f64 = base.astype(np.float64)

    for f in range(n_frames):
        t_frac = (f + 1) / n_frames
        angles = max_angle * x_norm * t_frac  # (N,)

        # Per-point rotation around Y axis
        c = np.cos(angles)  # (N,)
        s = np.sin(angles)  # (N,)
        bx = base_f64[:, 0]
        by = base_f64[:, 1]
        bz = base_f64[:, 2]
        # Y-axis rotation: x' = c*x + s*z, y' = y, z' = -s*x + c*z
        rx = c * bx + s * bz
        ry = by
        rz = -s * bx + c * bz
        frames[f] = np.stack([rx, ry, rz], axis=1).astype(np.float32)

    return frames


def random_motion(
    base: np.ndarray,
    n_frames: int,
    seed: int = 42,
) -> np.ndarray:
    """High-rank motion: independent random displacement per point per frame.

    NOT LBS-compatible — each point moves independently.  Used to prove that
    the fitter reports honestly large errors on incompressible motion.

    Parameters
    ----------
    base:
        Rest-pose positions, shape ``(N, 3)``.
    n_frames:
        Number of output frames.
    seed:
        NumPy random seed.

    Returns
    -------
    np.ndarray
        Shape ``(F, N, 3)`` float32.
    """
    rng = np.random.default_rng(seed)
    # Displacement magnitude ~10% of unit cube diagonal
    displacement = cast(
        np.ndarray,
        rng.standard_normal((n_frames, base.shape[0], 3)).astype(np.float32) * 0.1,
    )
    return cast(np.ndarray, base[np.newaxis, :, :] + displacement)
