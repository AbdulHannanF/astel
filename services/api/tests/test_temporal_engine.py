"""Unit tests for the pure WorkflowProgress -> ProgressEvent translation.

These run fully offline: no Temporal server, no temporalio runtime, just the
dataclass/translation logic in :mod:`astel_api.engine` and
:mod:`astel_api.temporal.shared`.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("ASTEL_DATABASE_URL", "sqlite+aiosqlite:///./astel_test.db")

from astel_api.engine import workflow_progress_to_event  # noqa: E402
from astel_api.schemas import LayerStage, TaskStatus  # noqa: E402
from astel_api.temporal.shared import (  # noqa: E402
    STAGE_IDS,
    STAGE_TARGETS,
    WorkflowProgress,
)

TASK_ID = "task-123"


def test_initial_progress_is_accepted() -> None:
    progress = WorkflowProgress(
        completed_stages=[],
        current_stage=None,
        current_index=0,
        total=len(STAGE_IDS),
        done=False,
    )
    event = workflow_progress_to_event(TASK_ID, progress)

    assert event.status == TaskStatus.RUNNING
    assert event.stage is None
    assert event.stage_label == "Accepted"
    assert event.stage_index == 0
    assert event.progress == 0.0
    assert event.metrics is None


def test_l0_to_l3_ordering_and_fractions() -> None:
    """Stepping current_stage through STAGE_IDS yields L0->L3 in order with
    monotonically increasing overall fractions."""
    events = []
    for index, stage_id in enumerate(STAGE_IDS):
        progress = WorkflowProgress(
            completed_stages=list(STAGE_IDS[:index]),
            current_stage=stage_id,
            current_index=index,
            total=len(STAGE_IDS),
            done=False,
        )
        events.append(workflow_progress_to_event(TASK_ID, progress))

    assert [e.stage for e in events] == [
        LayerStage.L0_SEED,
        LayerStage.L1_DENSE,
        LayerStage.L2_COARSE,
        LayerStage.L3_REFINED,
    ]
    assert [e.stage_index for e in events] == [0, 1, 2, 3]

    fractions = [e.progress for e in events]
    assert fractions == sorted(fractions)
    for index, frac in enumerate(fractions):
        assert frac == pytest.approx(index / len(STAGE_IDS))

    # No metrics on in-progress events.
    assert all(e.metrics is None for e in events)
    assert all(e.status == TaskStatus.RUNNING for e in events)


def test_terminal_progress_matches_l3_target_metrics() -> None:
    terminal_metrics = STAGE_TARGETS[LayerStage.L3_REFINED.value]
    progress = WorkflowProgress(
        completed_stages=list(STAGE_IDS),
        current_stage=None,
        current_index=len(STAGE_IDS),
        total=len(STAGE_IDS),
        done=True,
        metrics=terminal_metrics,
    )
    event = workflow_progress_to_event(TASK_ID, progress)

    assert event.status == TaskStatus.SUCCEEDED
    assert event.stage == LayerStage.L3_REFINED
    assert event.stage_label == "Complete"
    assert event.stage_index == len(STAGE_IDS)
    assert event.stage_count == len(STAGE_IDS)
    assert event.progress == 1.0
    assert event.metrics is not None
    assert event.metrics.splats == 48_000
    assert event.metrics == terminal_metrics


def test_failed_progress_mid_stage() -> None:
    progress = WorkflowProgress(
        completed_stages=[STAGE_IDS[0]],
        current_stage=STAGE_IDS[1],
        current_index=1,
        total=len(STAGE_IDS),
        done=False,
        failed=True,
    )
    event = workflow_progress_to_event(TASK_ID, progress)

    assert event.status == TaskStatus.FAILED
    assert event.stage == LayerStage(STAGE_IDS[1])
    assert event.stage_index == 1
    assert event.metrics is None


@pytest.mark.skipif(
    os.environ.get("ASTEL_TEMPORAL_TESTS") != "1",
    reason="full Temporal time-skipping integration test; set ASTEL_TEMPORAL_TESTS=1",
)
async def test_full_pipeline_via_temporal_test_environment() -> None:
    """End-to-end: run the real workflow + activities against Temporal's
    time-skipping test server and assert a full L0->L3->SUCCEEDED stream.

    Requires network access on first run (downloads the test server binary).
    """
    from temporalio.testing import WorkflowEnvironment
    from temporalio.worker import Worker

    from astel_api.temporal.activities import run_stage
    from astel_api.temporal.shared import TASK_QUEUE
    from astel_api.temporal.workflows import AstelPipelineWorkflow

    async with (
        await WorkflowEnvironment.start_time_skipping() as env,
        Worker(
            env.client,
            task_queue=TASK_QUEUE,
            workflows=[AstelPipelineWorkflow],
            activities=[run_stage],
        ),
    ):
        handle = await env.client.start_workflow(
            AstelPipelineWorkflow.run,
            args=["integration-task", 1000.0],
            id="astel-gen-integration-task",
            task_queue=TASK_QUEUE,
        )

        seen_stages: set[str] = set()
        terminal = None
        for _ in range(200):
            progress = await handle.query(AstelPipelineWorkflow.progress)
            seen_stages.update(progress.completed_stages)
            if progress.done:
                terminal = workflow_progress_to_event("integration-task", progress)
                break

        assert terminal is not None
        assert terminal.status == TaskStatus.SUCCEEDED
        assert terminal.progress == 1.0
        assert set(STAGE_IDS) <= seen_stages
        assert terminal.metrics is not None
        assert terminal.metrics.splats == 48_000

        await handle.result()
