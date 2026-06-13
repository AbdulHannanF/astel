"""Path-safety helpers for `.astel` package members.

manifest-v0.md section 1: "All layer/quality/export files are referenced
from the manifest by POSIX relative path from the package root. No absolute
paths, no `..` traversal. Readers MUST reject paths that escape the root."
This mirrors ``layer.schema.json``'s ``fileRef.path`` pattern
(``^(?!/)(?!.*\\.\\.).+$``) and ``buffers.schema.json``'s ``buffers[].uri``
pattern, applied uniformly to every path-shaped manifest field.
"""

from __future__ import annotations

import posixpath

from astel_format.errors import PathSecurityError


def validate_member_path(path: str) -> str:
    """Validate ``path`` is a safe POSIX-relative package-member path.

    Rejects:
      - empty paths
      - absolute paths (leading ``/`` or a drive letter / backslash, which
        would indicate a Windows-style absolute path slipped into the
        manifest)
      - any path containing a ``..`` segment (parent traversal)
      - backslashes (the manifest is POSIX-relative; mixing separators is a
        portability bug we refuse outright)

    Returns the path unchanged if valid. Raises :class:`PathSecurityError`
    otherwise.
    """
    if not path:
        raise PathSecurityError("empty path")
    if "\\" in path:
        raise PathSecurityError(f"path must use POSIX separators: {path!r}")
    if path.startswith("/"):
        raise PathSecurityError(f"absolute path not allowed: {path!r}")
    if posixpath.splitdrive(path)[0]:
        raise PathSecurityError(
            f"absolute (drive-qualified) path not allowed: {path!r}"
        )
    parts = path.split("/")
    if any(part == ".." for part in parts):
        raise PathSecurityError(f"path traversal not allowed: {path!r}")
    if any(part == "" for part in parts):
        raise PathSecurityError(f"malformed path (empty segment): {path!r}")
    return path
