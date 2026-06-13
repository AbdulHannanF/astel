"""Starts a new AssetPipelineWorkflow execution."""

import asyncio
import sys
import uuid

from temporalio.client import Client

from shared import TASK_QUEUE
from workflows import AssetPipelineWorkflow


async def main() -> None:
    asset_id = sys.argv[1] if len(sys.argv) > 1 else f"asset-{uuid.uuid4().hex[:8]}"
    seconds_per_stage = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0

    client = await Client.connect("localhost:7233", namespace="default")
    handle = await client.start_workflow(
        AssetPipelineWorkflow.run,
        args=[asset_id, seconds_per_stage],
        id=f"asset-pipeline-{asset_id}",
        task_queue=TASK_QUEUE,
    )
    print(f"started workflow_id={handle.id} run_id={handle.result_run_id}")


if __name__ == "__main__":
    asyncio.run(main())
