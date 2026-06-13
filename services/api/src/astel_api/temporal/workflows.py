"""``AstelPipelineWorkflow``: runs each PIPELINE stage as a ``run_stage`` activity.

Exposes a ``progress`` query returning :class:`WorkflowProgress`, which the
gateway translates into :class:`astel_api.schemas.ProgressEvent`s for SSE.
"""

from __future__ import annotations

from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from .shared import STAGE_IDS, StageInput, StageResult, WorkflowProgress


@workflow.defn
class AstelPipelineWorkflow:
    """Runs L0->L1->L2->L3 (per ``schemas.PIPELINE``) as ordered activities."""

    def __init__(self) -> None:
        self._completed: list[str] = []
        self._current: str | None = None
        self._current_index: int = 0
        self._done = False
        self._failed = False
        self._last_metrics: StageResult | None = None

    @workflow.run
    async def run(self, task_id: str, sim_speed: float = 1.0) -> list[str]:
        from .activities import run_stage

        for index, stage in enumerate(STAGE_IDS):
            self._current = stage
            self._current_index = index
            result = await workflow.execute_activity(
                run_stage,
                StageInput(stage=stage, task_id=task_id, sim_speed=sim_speed),
                start_to_close_timeout=timedelta(minutes=30),
                heartbeat_timeout=timedelta(seconds=10),
                retry_policy=RetryPolicy(maximum_attempts=10),
            )
            self._last_metrics = result
            self._completed.append(stage)
            self._current = None

        self._done = True
        return self._completed

    @workflow.query
    def progress(self) -> WorkflowProgress:
        return WorkflowProgress(
            completed_stages=list(self._completed),
            current_stage=self._current,
            current_index=self._current_index,
            total=len(STAGE_IDS),
            done=self._done,
            failed=self._failed,
            metrics=self._last_metrics.metrics if self._last_metrics else None,
        )
