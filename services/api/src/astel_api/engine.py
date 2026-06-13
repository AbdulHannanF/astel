"""Task engine abstraction.

``TaskEngine`` is the seam between the API and whatever actually runs the
pipeline. M1 ships :class:`InProcessStubEngine`, which simulates the L0->L3
stages with realistic per-stage durations and shaped metrics, streaming
:class:`ProgressEvent`s. Next session a ``TemporalTaskEngine`` implements the
same protocol — routes never change.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from .schemas import (
    PIPELINE,
    LayerStage,
    ProgressEvent,
    StageMetrics,
    StageSpec,
    TaskStatus,
)
from .temporal.shared import STAGE_IDS, WorkflowProgress

# Per-stage terminal metrics the stub "achieves". Shaped to look like a real
# refine run: splat count climbs, PSNR improves, Chamfer error shrinks.
_STAGE_TARGETS: dict[LayerStage, StageMetrics] = {
    LayerStage.L0_SEED: StageMetrics(splats=4_800, vram_gb=1.2),
    LayerStage.L1_DENSE: StageMetrics(splats=22_000, chamfer_mm=4.1, vram_gb=2.8),
    LayerStage.L2_COARSE: StageMetrics(
        splats=48_000, psnr_db=24.6, chamfer_mm=2.7, vram_gb=5.1
    ),
    LayerStage.L3_REFINED: StageMetrics(
        splats=48_000, psnr_db=31.2, chamfer_mm=0.9, vram_gb=7.4
    ),
}

# Ticks per stage for a smooth progress rail without flooding the client.
_TICKS_PER_STAGE = 6


@runtime_checkable
class TaskEngine(Protocol):
    """Runs a generation and yields progress events until terminal state."""

    def run(self, task_id: str) -> AsyncIterator[ProgressEvent]:
        """Async-iterate progress events for ``task_id``."""
        ...


class InProcessStubEngine:
    """Simulates the layered pipeline in-process (no GPU, no external deps)."""

    def __init__(self, sim_speed: float = 1.0, seed: int | None = None) -> None:
        # sim_speed > 1 compresses wall time (tests run fast).
        self._sim_speed = max(sim_speed, 0.001)
        self._rng = random.Random(seed)

    async def run(self, task_id: str) -> AsyncIterator[ProgressEvent]:
        count = len(PIPELINE)
        yield ProgressEvent(
            task_id=task_id,
            status=TaskStatus.RUNNING,
            stage=None,
            stage_label="Accepted",
            stage_index=0,
            stage_count=count,
            progress=0.0,
            message="Generation accepted; starting pipeline",
        )

        for index, spec in enumerate(PIPELINE):
            async for event in self._run_stage(task_id, spec, index, count):
                yield event

        yield ProgressEvent(
            task_id=task_id,
            status=TaskStatus.SUCCEEDED,
            stage=LayerStage.L3_REFINED,
            stage_label="Complete",
            stage_index=count,
            stage_count=count,
            progress=1.0,
            message="Asset ready",
            metrics=_STAGE_TARGETS[LayerStage.L3_REFINED],
        )

    async def _run_stage(
        self, task_id: str, spec: StageSpec, index: int, count: int
    ) -> AsyncIterator[ProgressEvent]:
        target = _STAGE_TARGETS[spec.stage]
        # Per-stage wall time, jittered ±15% then scaled by sim_speed.
        jitter = 1.0 + self._rng.uniform(-0.15, 0.15)
        stage_seconds = spec.nominal_seconds * jitter / self._sim_speed
        tick_seconds = stage_seconds / _TICKS_PER_STAGE

        for tick in range(1, _TICKS_PER_STAGE + 1):
            await asyncio.sleep(tick_seconds)
            frac = tick / _TICKS_PER_STAGE
            overall = (index + frac) / count
            final_tick = tick == _TICKS_PER_STAGE
            yield ProgressEvent(
                task_id=task_id,
                status=TaskStatus.RUNNING,
                stage=spec.stage,
                stage_label=spec.label,
                stage_index=index,
                stage_count=count,
                progress=round(overall, 4),
                message=spec.description,
                metrics=self._interp_metrics(target, frac) if final_tick else None,
            )

    @staticmethod
    def _interp_metrics(target: StageMetrics, frac: float) -> StageMetrics:
        """Report the stage's terminal metrics once it completes."""
        return target if frac >= 1.0 else StageMetrics()


# Stage id -> (label, index) lookup, derived from PIPELINE for the Temporal
# translation below (keeps it in lockstep with the stub's stage ordering).
_STAGE_LABELS: dict[str, str] = {spec.stage.value: spec.label for spec in PIPELINE}
_STAGE_DESCRIPTIONS: dict[str, str] = {
    spec.stage.value: spec.description for spec in PIPELINE
}


def workflow_progress_to_event(
    task_id: str, progress: WorkflowProgress
) -> ProgressEvent:
    """Translate a :class:`WorkflowProgress` snapshot into a :class:`ProgressEvent`.

    Pure function — unit-testable without a Temporal server. Mirrors the
    stub engine's event shape: ``stage``/``stage_label``/``stage_index`` track
    the in-progress (or last-completed) stage, ``progress`` is the overall
    fraction across all stages, and terminal :class:`StageMetrics` are only
    attached on the final SUCCEEDED event.
    """
    count = progress.total

    if progress.failed:
        stage_id = progress.current_stage or (
            progress.completed_stages[-1] if progress.completed_stages else None
        )
        stage = LayerStage(stage_id) if stage_id else None
        return ProgressEvent(
            task_id=task_id,
            status=TaskStatus.FAILED,
            stage=stage,
            stage_label=_STAGE_LABELS.get(stage_id) if stage_id else None,
            stage_index=len(progress.completed_stages),
            stage_count=count,
            progress=round(len(progress.completed_stages) / count, 4) if count else 0.0,
            message="Generation failed",
        )

    if progress.done:
        final_stage_id = STAGE_IDS[-1]
        return ProgressEvent(
            task_id=task_id,
            status=TaskStatus.SUCCEEDED,
            stage=LayerStage(final_stage_id),
            stage_label="Complete",
            stage_index=count,
            stage_count=count,
            progress=1.0,
            message="Asset ready",
            metrics=progress.metrics,
        )

    if progress.current_stage is None:
        # No stage started yet: mirrors the stub's initial "Accepted" event.
        return ProgressEvent(
            task_id=task_id,
            status=TaskStatus.RUNNING,
            stage=None,
            stage_label="Accepted",
            stage_index=0,
            stage_count=count,
            progress=0.0,
            message="Generation accepted; starting pipeline",
        )

    stage_id = progress.current_stage
    index = progress.current_index
    return ProgressEvent(
        task_id=task_id,
        status=TaskStatus.RUNNING,
        stage=LayerStage(stage_id),
        stage_label=_STAGE_LABELS[stage_id],
        stage_index=index,
        stage_count=count,
        progress=round(index / count, 4),
        message=_STAGE_DESCRIPTIONS[stage_id],
    )


class TemporalTaskEngine:
    """Runs the pipeline as a durable Temporal workflow.

    ``run(task_id)`` starts (or attaches to) ``AstelPipelineWorkflow`` with
    workflow id ``astel-gen-{task_id}`` on the configured task queue, then
    polls the ``progress`` query, translating each snapshot into a
    :class:`ProgressEvent` via :func:`workflow_progress_to_event`.
    """

    def __init__(
        self,
        address: str = "localhost:7233",
        namespace: str = "default",
        task_queue: str = "astel-pipeline",
        sim_speed: float = 1.0,
        poll_interval: float = 0.25,
    ) -> None:
        self._address = address
        self._namespace = namespace
        self._task_queue = task_queue
        self._sim_speed = max(sim_speed, 0.001)
        self._poll_interval = poll_interval

    async def run(self, task_id: str) -> AsyncIterator[ProgressEvent]:
        # Imported lazily so the default (stub) path never needs temporalio
        # at import time, and so `engine.py` stays importable offline.
        from temporalio.client import Client
        from temporalio.common import WorkflowIDConflictPolicy, WorkflowIDReusePolicy

        from .temporal.workflows import AstelPipelineWorkflow

        client = await Client.connect(self._address, namespace=self._namespace)

        handle = await client.start_workflow(
            AstelPipelineWorkflow.run,
            args=[task_id, self._sim_speed],
            id=f"astel-gen-{task_id}",
            task_queue=self._task_queue,
            id_reuse_policy=WorkflowIDReusePolicy.ALLOW_DUPLICATE_FAILED_ONLY,
            id_conflict_policy=WorkflowIDConflictPolicy.USE_EXISTING,
        )

        yield ProgressEvent(
            task_id=task_id,
            status=TaskStatus.RUNNING,
            stage=None,
            stage_label="Accepted",
            stage_index=0,
            stage_count=len(STAGE_IDS),
            progress=0.0,
            message="Generation accepted; starting pipeline",
        )

        failed = False
        while True:
            progress = await handle.query(AstelPipelineWorkflow.progress)

            if progress.done:
                yield workflow_progress_to_event(task_id, progress)
                return

            if not failed:
                yield workflow_progress_to_event(task_id, progress)

            # Check whether the workflow has terminally failed between polls.
            desc = await handle.describe()
            if desc.status is not None and desc.status.name not in (
                "RUNNING",
                "COMPLETED",
            ):
                failed = True
                yield ProgressEvent(
                    task_id=task_id,
                    status=TaskStatus.FAILED,
                    stage=(
                        LayerStage(progress.current_stage)
                        if progress.current_stage
                        else None
                    ),
                    stage_label=(
                        _STAGE_LABELS.get(progress.current_stage)
                        if progress.current_stage
                        else None
                    ),
                    stage_index=len(progress.completed_stages),
                    stage_count=progress.total,
                    progress=round(len(progress.completed_stages) / progress.total, 4)
                    if progress.total
                    else 0.0,
                    message=f"Generation failed: workflow status {desc.status.name}",
                )
                return

            await asyncio.sleep(self._poll_interval / self._sim_speed)
