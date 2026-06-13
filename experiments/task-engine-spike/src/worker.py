"""Worker process: hosts the workflow + activity for the spike."""

import asyncio
import logging

from temporalio.client import Client
from temporalio.worker import Worker

from shared import TASK_QUEUE
from workflows import AssetPipelineWorkflow
from activities import run_stage

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


async def main() -> None:
    client = await Client.connect("localhost:7233", namespace="default")
    worker = Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[AssetPipelineWorkflow],
        activities=[run_stage],
    )
    logging.info("worker starting on task queue %s", TASK_QUEUE)
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
