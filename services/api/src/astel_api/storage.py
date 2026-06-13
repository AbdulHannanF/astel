"""Local-filesystem artifact store with an S3-swappable seam.

M1 stores per-task generated artifacts (``l3.ply``, ``quality-report.json``,
...) on the local filesystem under ``settings.artifact_dir``. The
:class:`ArtifactStore` protocol is the seam: a future S3/MinIO-backed
implementation can satisfy the same interface without touching callers.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Protocol

from .config import Settings

_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")


class ArtifactStore(Protocol):
    """Per-task key/value-ish blob storage for generated artifacts."""

    def put(self, task_id: str, name: str, data: bytes) -> None:
        """Write ``data`` as artifact ``name`` for ``task_id``, creating dirs."""
        ...

    def path_for(self, task_id: str, name: str) -> Path | None:
        """Return the filesystem path for ``name``, or ``None`` if missing."""
        ...

    def list_names(self, task_id: str) -> list[str]:
        """List artifact names stored for ``task_id``."""
        ...


def _validate_name(name: str) -> None:
    if not _NAME_RE.match(name):
        raise ValueError(f"invalid artifact name: {name!r}")


class LocalArtifactStore:
    """Artifacts laid out as ``{root}/{task_id}/{name}`` on local disk."""

    def __init__(self, root: Path) -> None:
        self._root = root

    def _task_dir(self, task_id: str) -> Path:
        return self._root / task_id

    def put(self, task_id: str, name: str, data: bytes) -> None:
        _validate_name(name)
        task_dir = self._task_dir(task_id)
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / name).write_bytes(data)

    def path_for(self, task_id: str, name: str) -> Path | None:
        _validate_name(name)
        path = self._task_dir(task_id) / name
        if not path.is_file():
            return None
        return path

    def list_names(self, task_id: str) -> list[str]:
        task_dir = self._task_dir(task_id)
        if not task_dir.is_dir():
            return []
        return sorted(p.name for p in task_dir.iterdir() if p.is_file())


@lru_cache
def _store_for_root(root: Path) -> LocalArtifactStore:
    return LocalArtifactStore(root)


def get_artifact_store(settings: Settings) -> LocalArtifactStore:
    """Return a cached :class:`LocalArtifactStore` rooted at the artifact dir."""
    return _store_for_root(settings.artifact_dir)
