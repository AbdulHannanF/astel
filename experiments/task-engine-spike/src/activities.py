"""Stub activities for the 3-stage toy pipeline.

Each activity simulates GPU work by sleeping in small increments and
sending heartbeats. Heartbeats let Temporal detect a dead worker and
let a restarted worker resume from the last heartbeat detail instead
of restarting the activity from zero.
"""

import asyncio
import logging

from temporalio import activity

from shared import StageInput, StageResult

logger = logging.getLogger(__name__)

TICK = 0.5  # seconds per heartbeat tick


@activity.defn
async def run_stage(inp: StageInput) -> StageResult:
    total_ticks = int(inp.seconds / TICK)

    # Resume from last heartbeat if this activity was retried after a
    # worker crash. Temporal redelivers heartbeat details on retry.
    start_tick = 0
    info = activity.info()
    if info.heartbeat_details:
        start_tick = info.heartbeat_details[0]
        logger.info(
            "[%s/%s] resuming from heartbeat tick %d/%d",
            inp.asset_id, inp.stage, start_tick, total_ticks,
        )

    for tick in range(start_tick, total_ticks):
        await asyncio.sleep(TICK)
        activity.heartbeat(tick + 1)
        logger.info(
            "[%s/%s] tick %d/%d",
            inp.asset_id, inp.stage, tick + 1, total_ticks,
        )

    logger.info("[%s/%s] stage complete", inp.asset_id, inp.stage)
    return StageResult(stage=inp.stage, asset_id=inp.asset_id, ok=True)
