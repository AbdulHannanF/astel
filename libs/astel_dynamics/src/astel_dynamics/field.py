"""DeformationField: LBS deformation over a Gaussian splat cloud.

Stores K control nodes, per-gaussian blend weights (rows sum to 1), and
per-frame affine transforms [R|t] for each node.  The LBS deformation for
gaussian *n* at frame *f* is::

    deformed[n] = Σ_k  weights[n, k] * (R[f, k] @ base[n] + t[f, k])

Vectorised via np.einsum — no Python loops over N.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np


@dataclass
class DeformationField:
    """LBS deformation field binding a splat cloud to K control nodes.

    Parameters
    ----------
    node_positions:
        Rest positions of control nodes, shape ``(K, 3)`` float32.
    weights:
        Per-gaussian blend weights, shape ``(N, K)`` float32.  Rows must sum
        to 1 (enforced in ``__post_init__``).
    node_transforms:
        Per-frame affine transforms ``[R | t]``, shape ``(F, K, 3, 4)``
        float32.  The last column is the translation; the first three columns
        are the (unconstrained affine) rotation matrix.
    """

    node_positions: np.ndarray  # (K, 3) float32
    weights: np.ndarray  # (N, K) float32
    node_transforms: np.ndarray  # (F, K, 3, 4) float32

    def __post_init__(self) -> None:
        np_pos = np.asarray(self.node_positions, dtype=np.float32)
        np_w = np.asarray(self.weights, dtype=np.float32)
        np_tf = np.asarray(self.node_transforms, dtype=np.float32)

        if np_pos.ndim != 2 or np_pos.shape[1] != 3:
            raise ValueError(f"node_positions must be (K, 3), got {np_pos.shape}")
        K = np_pos.shape[0]

        if np_w.ndim != 2 or np_w.shape[1] != K:
            raise ValueError(f"weights must be (N, {K}), got {np_w.shape}")

        if np_tf.ndim != 4 or np_tf.shape[1] != K or np_tf.shape[2:] != (3, 4):
            raise ValueError(
                f"node_transforms must be (F, {K}, 3, 4), got {np_tf.shape}"
            )

        # Store as float32 C-contiguous arrays.
        object.__setattr__(self, "node_positions", np.ascontiguousarray(np_pos))
        object.__setattr__(self, "weights", np.ascontiguousarray(np_w))
        object.__setattr__(self, "node_transforms", np.ascontiguousarray(np_tf))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_gaussians(self) -> int:
        """Number of Gaussians (N)."""
        return int(self.weights.shape[0])

    @property
    def n_nodes(self) -> int:
        """Number of control nodes (K)."""
        return int(self.node_positions.shape[0])

    @property
    def n_frames(self) -> int:
        """Number of frames (F)."""
        return int(self.node_transforms.shape[0])

    # ------------------------------------------------------------------
    # Deformation
    # ------------------------------------------------------------------

    def apply(self, base_positions: np.ndarray, frame: int) -> np.ndarray:
        """Return deformed Gaussian means for *frame*.

        Parameters
        ----------
        base_positions:
            Rest-pose Gaussian means, shape ``(N, 3)`` float32.
        frame:
            Zero-based frame index in ``[0, F)``.

        Returns
        -------
        np.ndarray
            Deformed positions, shape ``(N, 3)`` float32.
        """
        base = np.asarray(base_positions, dtype=np.float32)
        if base.shape != (self.n_gaussians, 3):
            raise ValueError(
                f"base_positions must be ({self.n_gaussians}, 3), got {base.shape}"
            )
        if not (0 <= frame < self.n_frames):
            raise ValueError(f"frame must be in [0, {self.n_frames}), got {frame}")

        # node_transforms[frame]: shape (K, 3, 4)
        tf = self.node_transforms[frame]  # (K, 3, 4)
        R = tf[:, :, :3]  # (K, 3, 3)
        t = tf[:, :, 3]  # (K, 3)

        # Rotate each node's contribution: R[k] @ base[n]
        # base: (N, 3) → rotated: (K, N, 3)
        # einsum 'kij, nj -> kni': for each node k and gaussian n, R[k] @ base[n]
        rotated = np.einsum("kij,nj->kni", R, base)  # (K, N, 3)

        # Add translation: rotated[k, n] + t[k]  → (K, N, 3)
        translated = rotated + t[:, np.newaxis, :]  # (K, N, 3)

        # Blend: weights (N, K), translated (K, N, 3)
        # deformed[n] = sum_k weights[n, k] * translated[k, n]
        # einsum 'nk, kni -> ni'
        deformed = np.einsum("nk,kni->ni", self.weights, translated)  # (N, 3)

        return cast(np.ndarray, deformed.astype(np.float32))
