"""Query a running/completed workflow for its progress."""

import asyncio
import sys

from temporalio.client import Client

from workflows import AssetPipelineWorkflow


async def main() -> None:
    workflow_id = sys.argv[1]
    client = await Client.connect("localhost:7233", namespace="default")
    handle = client.get_workflow_handle_for(AssetPipelineWorkflow.run, workflow_id)
    progress = await handle.query(AssetPipelineWorkflow.progress)
    print(progress)

    desc = await handle.describe()
    print("status:", desc.status)


if __name__ == "__main__":
    asyncio.run(main())
