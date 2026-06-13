"""Run plan + runner: enumerate (case x system) jobs and execute adapters.

CPU-only, resumable-friendly: each job result is written incrementally to the
results directory as one JSON file per job, keyed by ``<case_id>__<system>.json``.
Re-running with the same results directory skips jobs that already have a
result file unless ``overwrite=True`` is passed.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from astel_eval.adapters import Adapter, GenerationArtifact, all_adapters
from astel_eval.corpus import Case, load_corpus


@dataclass(frozen=True, slots=True)
class JobResult:
    """Telemetry + outcome for one (case, system) job."""

    case_id: str
    system: str
    status: str
    """``"ok"`` (adapter ran without raising) or ``"error"`` (adapter raised)."""

    wall_seconds: float
    available: bool
    unavailable_reason: str
    output_paths: tuple[str, ...]
    metadata: dict[str, str]
    error: str = ""
    """Exception message, set only when ``status == "error"``."""


@dataclass(frozen=True, slots=True)
class RunPlan:
    """A fully-enumerated set of (case, system) jobs."""

    cases: tuple[Case, ...]
    systems: tuple[Adapter, ...] = field(default_factory=all_adapters)

    @property
    def job_count(self) -> int:
        return len(self.cases) * len(self.systems)

    def jobs(self) -> tuple[tuple[Case, Adapter], ...]:
        """All (case, adapter) pairs, cases-major then systems."""
        return tuple((case, system) for case in self.cases for system in self.systems)


def default_plan() -> RunPlan:
    """Build the full 50-case x N-system plan from the frozen corpus."""
    return RunPlan(cases=load_corpus())


def _result_path(results_dir: Path, case_id: str, system: str) -> Path:
    return results_dir / f"{case_id}__{system}.json"


def _job_result_to_json(result: JobResult) -> dict[str, object]:
    return asdict(result)


def _run_one_job(case: Case, adapter: Adapter) -> JobResult:
    start = time.perf_counter()
    try:
        artifact: GenerationArtifact = adapter.generate(case)
    except Exception as exc:  # noqa: BLE001 - record any adapter failure
        elapsed = time.perf_counter() - start
        return JobResult(
            case_id=case.id,
            system=adapter.name,
            status="error",
            wall_seconds=elapsed,
            available=False,
            unavailable_reason="",
            output_paths=(),
            metadata={},
            error=str(exc),
        )
    elapsed = time.perf_counter() - start
    return JobResult(
        case_id=artifact.case_id,
        system=artifact.system,
        status="ok",
        wall_seconds=elapsed,
        available=artifact.available,
        unavailable_reason=artifact.unavailable_reason,
        output_paths=tuple(str(p) for p in artifact.output_paths),
        metadata=dict(artifact.metadata),
    )


def run_plan(
    plan: RunPlan,
    results_dir: Path,
    *,
    overwrite: bool = False,
) -> list[JobResult]:
    """Execute every job in ``plan``, writing incremental JSON results.

    Returns the list of ``JobResult`` for this invocation in job order
    (skipped/pre-existing jobs are still loaded and included so the return
    value always reflects the full plan).
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    results: list[JobResult] = []
    for case, adapter in plan.jobs():
        path = _result_path(results_dir, case.id, adapter.name)
        if path.exists() and not overwrite:
            existing = json.loads(path.read_text(encoding="utf-8"))
            results.append(
                JobResult(
                    case_id=existing["case_id"],
                    system=existing["system"],
                    status=existing["status"],
                    wall_seconds=existing["wall_seconds"],
                    available=existing["available"],
                    unavailable_reason=existing["unavailable_reason"],
                    output_paths=tuple(existing["output_paths"]),
                    metadata=dict(existing["metadata"]),
                    error=existing.get("error", ""),
                )
            )
            continue
        result = _run_one_job(case, adapter)
        path.write_text(
            json.dumps(_job_result_to_json(result), indent=2), encoding="utf-8"
        )
        results.append(result)
    return results


def load_results(results_dir: Path) -> list[JobResult]:
    """Load all previously-written job result files from ``results_dir``."""
    results: list[JobResult] = []
    for path in sorted(results_dir.glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        results.append(
            JobResult(
                case_id=raw["case_id"],
                system=raw["system"],
                status=raw["status"],
                wall_seconds=raw["wall_seconds"],
                available=raw["available"],
                unavailable_reason=raw["unavailable_reason"],
                output_paths=tuple(raw["output_paths"]),
                metadata=dict(raw["metadata"]),
                error=raw.get("error", ""),
            )
        )
    return results
