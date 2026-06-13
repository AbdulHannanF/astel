"""Connectable Temporal worker entrypoint.

Run with ``python -m astel_api.temporal.worker``. Hosts
:class:`AstelPipelineWorkflow` and ``run_stage`` on the configured task queue,
connecting to ``settings.temporal_address`` / ``settings.temporal_namespace``.
This is what ``astel up`` launches alongside the dev server.
"""

from __future__ import annotations

import asyncio
import logging

from temporalio.client import Client
from temporalio.worker import Worker

from ..config import get_settings
from .activities import run_stage
from .shared import TASK_QUEUE
from .workflows import AstelPipelineWorkflow

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


async def main() -> None:
    settings = get_settings()
    client = await Client.connect(
        settings.temporal_address, namespace=settings.temporal_namespace
    )
    task_queue = settings.temporal_task_queue or TASK_QUEUE
    worker = Worker(
        client,
        task_queue=task_queue,
        workflows=[AstelPipelineWorkflow],
        activities=[run_stage],
    )
    logger.info("astel temporal worker starting on task queue %s", task_queue)
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
