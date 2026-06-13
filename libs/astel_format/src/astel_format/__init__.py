"""Reader/writer for the `.astel` package format (manifest-v0).

See ``docs/specs/manifest-v0.md`` (repo root) for the authoritative prose
spec and ``astel_format.schemas`` for the bundled JSON Schemas (draft
2020-12) that this package's pydantic models mirror. Where the prose and a
schema disagree, the schema wins -- noted inline where relevant.
"""

from __future__ import annotations

from astel_format.builder import build_minimal_package
from astel_format.errors import (
    AstelFormatError,
    AstelValidationError,
    PathSecurityError,
)
from astel_format.models import (
    Accessor,
    AssetIdentity,
    BufferEntry,
    BufferTable,
    BufferView,
    CoordinateSystem,
    ExportRecord,
    FileRef,
    GeometricError,
    HallucinationReport,
    LayerEntry,
    Manifest,
    ProvenanceChannel,
    ProvenanceDescriptor,
    QualityReport,
    Scale,
    ScaleConfidence,
)
from astel_format.package import MIMETYPE, MIMETYPE_BYTES, AstelPackage

__all__ = [
    "Accessor",
    "AssetIdentity",
    "AstelFormatError",
    "AstelPackage",
    "AstelValidationError",
    "BufferEntry",
    "BufferTable",
    "BufferView",
    "CoordinateSystem",
    "ExportRecord",
    "FileRef",
    "GeometricError",
    "HallucinationReport",
    "LayerEntry",
    "MIMETYPE",
    "MIMETYPE_BYTES",
    "Manifest",
    "PathSecurityError",
    "ProvenanceChannel",
    "ProvenanceDescriptor",
    "QualityReport",
    "Scale",
    "ScaleConfidence",
    "build_minimal_package",
]
