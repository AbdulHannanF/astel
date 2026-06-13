"""Adapter protocol and stub adapters for the blind-eval systems.

Per CORPUS.md §4.1, every case is run through each of: **AURIGA** (Astel),
**raw TRELLIS.2**, **Meshy free tier**, and **Tripo free tier** (secondary
baseline). This module defines the common ``Adapter`` protocol all systems
implement, plus stub implementations that do no real generation work (no
network, no GPU) -- they exist so the runner and scoring scaffold can be
exercised end-to-end before real backends land.

Honesty (CLAUDE.md §1.3): every stub artifact carries ``available=False`` and
a human-readable ``unavailable_reason`` so results can never be mistaken for
real generations. The runner and CLI must surface this prominently.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from astel_eval.corpus import Case


@dataclass(frozen=True, slots=True)
class GenerationArtifact:
    """The output of running one system on one case.

    Real adapters will populate ``output_paths`` with paths to generated
    splat/render files. Stub adapters leave it empty and set
    ``available=False``.
    """

    case_id: str
    system: str
    available: bool
    """``False`` for stub adapters (or any adapter that could not produce a
    real generation). Downstream scoring/reporting MUST treat
    ``available=False`` artifacts as non-results, never as a real score of 1
    or any other value."""

    unavailable_reason: str = ""
    """Human-readable explanation, e.g. 'STUB adapter: no GPU/network access
    configured; real backend lands post-M3'. Empty when ``available=True``."""

    output_paths: tuple[Path, ...] = field(default_factory=tuple)
    """Paths to generated artifacts (splat files, renders, etc.). Empty for
    stub adapters."""

    metadata: dict[str, str] = field(default_factory=dict)
    """Free-form key/value metadata (model version, seed, timing notes the
    adapter itself wants to record, etc.)."""


class Adapter(Protocol):
    """Protocol every per-system adapter implements.

    ``name`` is the human-readable system label used in results files and
    reports (e.g. ``"astel"``, ``"trellis2"``, ``"meshy_free"``, ``"tripo_free"``).
    """

    name: str

    def generate(self, case: Case) -> GenerationArtifact:
        """Run (or stub-run) generation for ``case`` and return the artifact."""
        ...


_STUB_REASON = (
    "STUB adapter ({name}): no real generation performed (no network, no "
    "GPU). This is a placeholder for harness wiring only -- do not use this "
    "result for any scoring or comparison. Real backend lands post-M3 per "
    "CLAUDE.md M3 gate ordering."
)


@dataclass
class _StubAdapter:
    """Shared stub behavior: every stub adapter just records its own name and
    returns an ``available=False`` artifact with no output paths.
    """

    name: str

    def generate(self, case: Case) -> GenerationArtifact:
        return GenerationArtifact(
            case_id=case.id,
            system=self.name,
            available=False,
            unavailable_reason=_STUB_REASON.format(name=self.name),
            output_paths=(),
            metadata={"modality": case.modality},
        )


def AstelAdapter() -> Adapter:  # noqa: N802 - factory named like a class per spec
    """Stub adapter for AURIGA/Astel itself.

    Real implementation will invoke the Astel generation pipeline (text/image/
    capture -> L0..L3+ -> exported render/splat) once it exists.
    """
    return _StubAdapter(name="astel")


def Trellis2Adapter() -> Adapter:  # noqa: N802
    """Stub adapter for raw TRELLIS.2 (unmodified checkpoint, no Astel
    finishing pipeline), per CORPUS.md §4.1.
    """
    return _StubAdapter(name="trellis2")


def MeshyAdapter() -> Adapter:  # noqa: N802
    """Stub adapter for Meshy free tier, per CORPUS.md §4.1."""
    return _StubAdapter(name="meshy_free")


def TripoAdapter() -> Adapter:  # noqa: N802
    """Stub adapter for Tripo free tier (secondary baseline, not required for
    the M3 gate but tracked), per CORPUS.md §4.1.
    """
    return _StubAdapter(name="tripo_free")


#: Canonical ordered list of all system adapter factories in the blind eval,
#: per CORPUS.md §4.1.
ALL_SYSTEM_FACTORIES: tuple[Callable[[], Adapter], ...] = (
    AstelAdapter,
    Trellis2Adapter,
    MeshyAdapter,
    TripoAdapter,
)


def all_adapters() -> tuple[Adapter, ...]:
    """Construct one fresh instance of every system adapter, in canonical order."""
    return (AstelAdapter(), Trellis2Adapter(), MeshyAdapter(), TripoAdapter())
