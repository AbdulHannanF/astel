"""Activities for the Astel pipeline workflow.

``run_stage`` simulates the GPU work for one layer-stack stage: it sleeps for
the stage's nominal duration (scaled by ``sim_speed``), heartbeating once per
tick so a worker crash mid-stage can resume from the last recorded tick
(mirrors ``experiments/task-engine-spike/src/activities.py``).
"""

from __future__ import annotations

import asyncio
import logging

from temporalio import activity

from .shared import (
    STAGE_SECONDS,
    STAGE_TARGETS,
    TICKS_PER_STAGE,
    StageInput,
    StageResult,
)

logger = logging.getLogger(__name__)


@activity.defn
async def run_stage(inp: StageInput) -> StageResult:
    """Simulate one pipeline stage, heartbeating per tick and resuming on retry."""
    nominal_seconds = STAGE_SECONDS[inp.stage]
    sim_speed = max(inp.sim_speed, 0.001)
    tick_seconds = (nominal_seconds / TICKS_PER_STAGE) / sim_speed

    start_tick = 0
    info = activity.info()
    if info.heartbeat_details:
        start_tick = int(info.heartbeat_details[0])
        logger.info(
            "[%s/%s] resuming from heartbeat tick %d/%d",
            inp.task_id,
            inp.stage,
            start_tick,
            TICKS_PER_STAGE,
        )

    for tick in range(start_tick, TICKS_PER_STAGE):
        await asyncio.sleep(tick_seconds)
        activity.heartbeat(tick + 1)

    return StageResult(
        stage=inp.stage,
        task_id=inp.task_id,
        metrics=STAGE_TARGETS[inp.stage],
    )
