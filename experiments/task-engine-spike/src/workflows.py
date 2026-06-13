"""3-stage toy pipeline workflow: l0_seed -> l1_dense -> l2_coarse.

Exposes a query for progress so a UI/SSE layer could poll it, mirroring
the per-layer preview/refine progress events Astel's orchestrator needs.
"""

from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from shared import STAGES, PipelineProgress, StageInput
    from activities import run_stage


@workflow.defn
class AssetPipelineWorkflow:
    def __init__(self) -> None:
        self._completed: list[str] = []
        self._current: str | None = None
        self._done = False

    @workflow.run
    async def run(self, asset_id: str, seconds_per_stage: float = 5.0) -> list[str]:
        for stage in STAGES:
            self._current = stage
            await workflow.execute_activity(
                run_stage,
                StageInput(stage=stage, asset_id=asset_id, seconds=seconds_per_stage),
                start_to_close_timeout=timedelta(minutes=5),
                heartbeat_timeout=timedelta(seconds=10),
                retry_policy=RetryPolicy(maximum_attempts=10),
            )
            self._completed.append(stage)
            self._current = None

        self._done = True
        return self._completed

    @workflow.query
    def progress(self) -> PipelineProgress:
        return PipelineProgress(
            asset_id="(see workflow id)",
            completed_stages=list(self._completed),
            current_stage=self._current,
            done=self._done,
        )
