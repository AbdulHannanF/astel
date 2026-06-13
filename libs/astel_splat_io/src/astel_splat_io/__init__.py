"""Splat export/import writers for Astel.

Public surface:

- :class:`SplatCloud` — shared in-memory splat representation (3DGS raw
  parameterisation: positions, SH band-0 DC colour, opacity logit, log-scale,
  wxyz quaternion). Matches ``pipelines/stub/make_sample_splat.py``.
- :func:`write_ply` / :func:`read_ply` — binary little-endian INRIA-layout PLY
  (the archival master format).
- :func:`write_spz` / :func:`read_spz` — Niantic SPZ v3 (gzip, smallest-three
  quaternions). See ``FORMATS.md`` for the verified spec and sources.
- :func:`write_sog` / :func:`read_sog` — PlayCanvas SOG/SOGS bundle. Partial:
  see ``FORMATS.md`` for what is implemented.
- :func:`write_provenance_sidecar` — ``*.astl.json`` + companion ``.bin``
  per manifest-v0 section 11.3.
"""

from astel_splat_io.cloud import SplatCloud
from astel_splat_io.ply import read_ply, write_ply
from astel_splat_io.provenance import write_provenance_sidecar
from astel_splat_io.sog import read_sog, write_sog
from astel_splat_io.spz import read_spz, write_spz

__all__ = [
    "SplatCloud",
    "read_ply",
    "read_sog",
    "read_spz",
    "write_ply",
    "write_provenance_sidecar",
    "write_sog",
    "write_spz",
]
