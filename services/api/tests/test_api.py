"""Endpoint tests for the Astel API.

Uses an in-memory-ish file SQLite db per test session and a high sim_speed so
the SSE pipeline completes in well under a second.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator, Iterator

import pytest

# Point the app at a throwaway db + fast sim BEFORE importing the app.
os.environ["ASTEL_DATABASE_URL"] = "sqlite+aiosqlite:///./astel_test.db"
os.environ["ASTEL_SIM_SPEED"] = "400"

import httpx  # noqa: E402
from httpx import ASGITransport  # noqa: E402

from astel_api.config import get_settings  # noqa: E402
from astel_api.db import init_db  # noqa: E402
from astel_api.main import app  # noqa: E402


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> Iterator[None]:
    get_settings.cache_clear()
    yield


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    await init_db()
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_healthz(client: httpx.AsyncClient) -> None:
    resp = await client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["service"] == "astel-api"


async def test_pipeline_stages(client: httpx.AsyncClient) -> None:
    resp = await client.get("/v1/pipeline")
    assert resp.status_code == 200
    stages = resp.json()
    assert [s["stage"] for s in stages] == [
        "L0_SEED",
        "L1_DENSE",
        "L2_COARSE",
        "L3_REFINED",
    ]


async def test_create_generation_returns_task(client: httpx.AsyncClient) -> None:
    resp = await client.post(
        "/v1/generations",
        json={"modality": "text", "prompt": "a worn brass astrolabe"},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["status"] == "queued"
    assert body["modality"] == "text"
    assert body["events_url"] == f"/v1/generations/{body['id']}/events"


async def test_create_generation_rejects_empty_prompt(
    client: httpx.AsyncClient,
) -> None:
    resp = await client.post("/v1/generations", json={"modality": "text", "prompt": ""})
    assert resp.status_code == 422


async def test_create_generation_rejects_bad_modality(
    client: httpx.AsyncClient,
) -> None:
    resp = await client.post(
        "/v1/generations", json={"modality": "hologram", "prompt": "x"}
    )
    assert resp.status_code == 422


async def test_get_unknown_generation_404(client: httpx.AsyncClient) -> None:
    resp = await client.get("/v1/generations/does-not-exist")
    assert resp.status_code == 404


async def test_events_stream_runs_full_pipeline(
    client: httpx.AsyncClient,
) -> None:
    create = await client.post(
        "/v1/generations",
        json={"modality": "image", "prompt": "a ceramic teapot"},
    )
    task_id = create.json()["id"]

    events: list[dict[str, object]] = []
    async with client.stream("GET", f"/v1/generations/{task_id}/events") as stream:
        assert stream.status_code == 200
        async for line in stream.aiter_lines():
            if line.startswith("data:"):
                events.append(json.loads(line[len("data:") :].strip()))

    assert events, "expected progress events"
    # Monotonic progress, terminal success, all four stages observed.
    progresses = [float(e["progress"]) for e in events]  # type: ignore[arg-type]
    assert progresses == sorted(progresses)
    assert events[-1]["status"] == "succeeded"
    assert events[-1]["progress"] == 1.0
    seen_stages = {e["stage"] for e in events if e["stage"]}
    assert {"L0_SEED", "L1_DENSE", "L2_COARSE", "L3_REFINED"} <= seen_stages
    # Final event carries refined-layer metrics.
    final_metrics = events[-1]["metrics"]
    assert isinstance(final_metrics, dict)
    assert final_metrics["splats"] == 48_000


async def test_events_unknown_task_404(client: httpx.AsyncClient) -> None:
    resp = await client.get("/v1/generations/nope/events")
    assert resp.status_code == 404
