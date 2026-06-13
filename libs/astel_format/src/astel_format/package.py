"""`.astel` package reader/writer (manifest-v0.md section 1).

A `.astel` file is a ZIP archive whose first entry is an uncompressed
``mimetype`` member (OPC/ODF convention) containing the literal ASCII bytes
``application/vnd.astel.package+zip``, followed by ``manifest.json`` and the
files it references under ``layers/``, ``quality/``, and ``exports/``.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

from astel_format.errors import AstelValidationError
from astel_format.models import LayerEntry, Manifest
from astel_format.paths import validate_member_path
from astel_format.schema_validation import validate_manifest_dict

MIMETYPE = "application/vnd.astel.package+zip"
MIMETYPE_BYTES = MIMETYPE.encode("ascii")

_MIMETYPE_MEMBER = "mimetype"
_MANIFEST_MEMBER = "manifest.json"


def _manifest_referenced_paths(manifest: Manifest) -> set[str]:
    """Collect every package-relative file path the manifest references.

    Covers layer ``files[].path``, layer kind-specific paths (appearance
    env_map/baked_pbr, collision sdf/convex_set/isosurface/mass_props,
    physics_material regions, dynamics deformation/timeline), buffer URIs,
    and export records' ``path``/``sidecar_path``.
    """
    paths: set[str] = set()

    def layer_paths(layer: LayerEntry) -> None:
        for f in layer.files or []:
            paths.add(f.path)
        if layer.appearance is not None:
            a = layer.appearance
            for p in (a.env_map_path, a.baked_pbr_path):
                if p:
                    paths.add(p)
        if layer.collision is not None:
            c = layer.collision
            for p in (c.sdf_path, c.convex_set_path, c.mass_props_path):
                if p:
                    paths.add(p)
            if c.isosurface is not None:
                paths.add(c.isosurface.path)
        if layer.physics_material is not None:
            pm = layer.physics_material
            if pm.regions_path:
                paths.add(pm.regions_path)
        if layer.dynamics is not None:
            d = layer.dynamics
            for p in (d.deformation_path, d.timeline_path):
                if p:
                    paths.add(p)

    for layer in (
        manifest.layers.l0,
        manifest.layers.l1,
        manifest.layers.l2,
        manifest.layers.l3,
        manifest.layers.l4,
        manifest.layers.l5,
        manifest.layers.l6,
        manifest.layers.l7,
    ):
        if layer is not None:
            layer_paths(layer)

    for buf in manifest.buffers.buffers:
        paths.add(buf.uri)

    for export in manifest.exports or []:
        paths.add(export.path)
        if export.sidecar_path:
            paths.add(export.sidecar_path)

    return paths


class AstelPackage:
    """An in-memory `.astel` package: a :class:`Manifest` plus member bytes.

    ``files`` maps POSIX-relative package paths (e.g.
    ``"layers/l3_refined/splats.ply"``, ``"quality/report.json"``) to their
    raw bytes. ``manifest.json`` and ``mimetype`` are NOT stored in
    ``files``; they are derived from :attr:`manifest` on write.
    """

    def __init__(self, manifest: Manifest, files: dict[str, bytes]) -> None:
        self.manifest = manifest
        self.files: dict[str, bytes] = dict(files)

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def to_manifest_dict(self) -> dict[str, Any]:
        """Serialise :attr:`manifest` to a plain dict (JSON-ready).

        Uses ``exclude_unset=True`` so fields never assigned by the caller
        (and therefore not part of the original/intended manifest) are
        omitted -- required for ``additionalProperties: false`` schemas --
        while explicit ``null`` values (the honesty contract, section 6)
        and unknown/vendor keys (``extra="allow"``, section 10) are
        preserved.
        """
        return self.manifest.model_dump(mode="json", exclude_unset=True, by_alias=True)

    def validate(self) -> None:
        """Validate the manifest against the schema and cross-reference files.

        Raises :class:`AstelValidationError` if the manifest fails
        ``manifest.schema.json``, references a path outside the package
        root, or references a file not present in :attr:`files`.
        """
        # Path-safety checks run first (and raise PathSecurityError) so a
        # traversal/absolute-path reference is reported with a precise
        # error type even though the schema's fileRef.path/buffer.uri regex
        # would also reject it (manifest-v0.md section 1).
        referenced = _manifest_referenced_paths(self.manifest)
        for path in referenced:
            validate_member_path(path)

        manifest_dict = self.to_manifest_dict()
        validate_manifest_dict(manifest_dict)

        for path in referenced:
            if path not in self.files:
                raise AstelValidationError(
                    f"manifest references missing file: {path!r}"
                )

    def write(self, path: str | Path) -> None:
        """Write this package to ``path`` as a `.astel` ZIP.

        Validates the manifest first (raises on failure; no partial file is
        left behind on validation error -- the zip is only opened after
        :meth:`validate` succeeds).

        Layout: ``mimetype`` (STORED, first entry, exact MIME bytes), then
        ``manifest.json``, then every file in :attr:`files` referenced by
        the manifest. Unreferenced entries in :attr:`files` are written too
        (manifest-v0.md: "a file present in the zip but unreferenced by the
        manifest is IGNORED" -- writers may include scratch files, but this
        writer only ever writes what was given to it).
        """
        self.validate()

        manifest_bytes = json.dumps(
            self.to_manifest_dict(), indent=2, sort_keys=False
        ).encode("utf-8")

        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(
                zipfile.ZipInfo(_MIMETYPE_MEMBER), MIMETYPE_BYTES, zipfile.ZIP_STORED
            )
            zf.writestr(_MANIFEST_MEMBER, manifest_bytes)
            for member_path, data in self.files.items():
                validate_member_path(member_path)
                zf.writestr(member_path, data)

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    @classmethod
    def read(cls, path: str | Path) -> AstelPackage:
        """Read and validate a `.astel` package from ``path``.

        Enforces (manifest-v0.md section 1):
          - ``mimetype`` is present, first, STORED, and contains the exact
            ``application/vnd.astel.package+zip`` bytes.
          - ``manifest.json`` is present and validates against
            ``manifest.schema.json``.
          - every manifest file reference exists in the archive and is a
            safe POSIX-relative path (no absolute paths, no ``..``).
          - a manifest reference to a missing file is an ERROR.
          - a file present but unreferenced is IGNORED (not loaded into
            :attr:`files`).
        """
        with zipfile.ZipFile(path, "r") as zf:
            infos = zf.infolist()
            if not infos:
                raise AstelValidationError("empty zip archive")

            first = infos[0]
            if first.filename != _MIMETYPE_MEMBER:
                raise AstelValidationError(
                    f"first zip entry must be {_MIMETYPE_MEMBER!r}, "
                    f"got {first.filename!r}"
                )
            if first.compress_type != zipfile.ZIP_STORED:
                raise AstelValidationError(
                    "mimetype entry must be STORED (uncompressed)"
                )
            mimetype_bytes = zf.read(_MIMETYPE_MEMBER)
            if mimetype_bytes != MIMETYPE_BYTES:
                raise AstelValidationError(
                    f"mimetype entry must be {MIMETYPE_BYTES!r}, got {mimetype_bytes!r}"
                )

            names = set(zf.namelist())
            if _MANIFEST_MEMBER not in names:
                raise AstelValidationError(f"missing {_MANIFEST_MEMBER}")

            manifest_dict = json.loads(zf.read(_MANIFEST_MEMBER).decode("utf-8"))
            validate_manifest_dict(manifest_dict)
            manifest = Manifest.model_validate(manifest_dict)

            referenced = _manifest_referenced_paths(manifest)
            files: dict[str, bytes] = {}
            for ref_path in referenced:
                validate_member_path(ref_path)
                if ref_path not in names:
                    raise AstelValidationError(
                        f"manifest references missing file: {ref_path!r}"
                    )
                files[ref_path] = zf.read(ref_path)

            # Unreferenced members (besides mimetype/manifest.json) are
            # ignored per spec -- not loaded into `files`.

        return cls(manifest=manifest, files=files)
