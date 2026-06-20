"""Binary serialisation for :class:`DeformationField`.

File layout (little-endian)
---------------------------
::

    [8 bytes]  magic  b"ASTLDYN0"
    [4 bytes]  uint32 N  (n_gaussians)
    [4 bytes]  uint32 K  (n_nodes)
    [4 bytes]  uint32 F  (n_frames)
    [K*3 * 4 bytes]      node_positions  float32 C-contiguous
    [N*K * 4 bytes]      weights         float32 C-contiguous
    [F*K*3*4 * 4 bytes]  node_transforms float32 C-contiguous

Round-trip is lossless to float32 precision.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from .field import DeformationField

_MAGIC = b"ASTLDYN0"
_HEADER_FMT = "<III"  # N, K, F as uint32 little-endian


def write_deformation_bin(field: DeformationField, path: str | Path) -> None:
    """Write *field* to a binary ``.bin`` file.

    Parameters
    ----------
    field:
        The :class:`DeformationField` to serialise.
    path:
        Output file path.
    """
    out = Path(path)
    N = field.n_gaussians
    K = field.n_nodes
    F = field.n_frames

    with out.open("wb") as fp:
        fp.write(_MAGIC)
        fp.write(struct.pack(_HEADER_FMT, N, K, F))
        fp.write(np.ascontiguousarray(field.node_positions, dtype=np.float32).tobytes())
        fp.write(np.ascontiguousarray(field.weights, dtype=np.float32).tobytes())
        fp.write(
            np.ascontiguousarray(field.node_transforms, dtype=np.float32).tobytes()
        )


def read_deformation_bin(path: str | Path) -> DeformationField:
    """Read a :class:`DeformationField` from a binary ``.bin`` file.

    Parameters
    ----------
    path:
        Input file path written by :func:`write_deformation_bin`.

    Returns
    -------
    DeformationField
        The deserialised field.

    Raises
    ------
    ValueError
        If the magic bytes are wrong or the file is truncated.
    """
    data = Path(path).read_bytes()
    offset = 0

    # Magic
    magic = data[offset : offset + 8]
    if magic != _MAGIC:
        raise ValueError(f"Bad magic: expected {_MAGIC!r}, got {magic!r}")
    offset += 8

    # Header
    header_size = struct.calcsize(_HEADER_FMT)
    N, K, F = struct.unpack_from(_HEADER_FMT, data, offset)
    offset += header_size

    # Validate the declared sizes against the ACTUAL file length BEFORE slicing
    # or allocating. `.astel` packages may be untrusted input; without this a
    # tiny file declaring an enormous N/K/F would otherwise fail with a confusing
    # reshape error. The exact-size check closes the amplification vector (a
    # small file can never pass while claiming huge arrays) and also rejects
    # truncated or trailing-junk files with a clear, actionable message. Python
    # slicing is memory-safe regardless, so this is hardening, not a buffer fix.
    n_floats = K * 3 + N * K + F * K * 3 * 4
    expected = 8 + header_size + n_floats * 4
    if len(data) != expected:
        raise ValueError(
            f"deformation .bin size mismatch: header declares N={N}, K={K}, "
            f"F={F} (expected {expected} bytes total) but file is "
            f"{len(data)} bytes — truncated, corrupt, or maliciously crafted."
        )

    def _read_array(shape: tuple[int, ...]) -> np.ndarray:
        nonlocal offset
        count = 1
        for s in shape:
            count *= s
        nbytes = count * 4  # float32
        arr = np.frombuffer(data[offset : offset + nbytes], dtype=np.float32).copy()
        arr = arr.reshape(shape)
        offset += nbytes
        return arr

    node_positions = _read_array((K, 3))
    weights = _read_array((N, K))
    node_transforms = _read_array((F, K, 3, 4))

    return DeformationField(
        node_positions=node_positions,
        weights=weights,
        node_transforms=node_transforms,
    )
