"""Endpoint tests for the Astel API.

Uses an in-memory-ish file SQLite db per test session and a high sim_speed so
the SSE pipeline completes in well under a second.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator, Iterator
from unittest.mock import patch

import pytest

# Point the app at a throwaway db + fast sim BEFORE importing the app.
os.environ["ASTEL_DATABASE_URL"] = "sqlite+aiosqlite:///./astel_test.db"
os.environ["ASTEL_SIM_SPEED"] = "400"

import httpx  # noqa: E402
from httpx import ASGITransport  # noqa: E402

from astel_api.config import get_settings  # noqa: E402
from astel_api.db import init_db  # noqa: E402
from astel_api.main import (  # noqa: E402
    _l6_json_artifact_path,
    _spec_longest_axis_m,
    app,
)
from astel_api.storage import LocalArtifactStore  # noqa: E402


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


async def _drain_events(
    client: httpx.AsyncClient, task_id: str
) -> list[dict[str, object]]:
    """Consume a generation's SSE stream to completion, returning all events.

    Production now runs asynchronously in a background job; draining the event
    stream blocks until the job reaches a terminal state, so callers can then
    read back the finalised row (produced/splats/billing).
    """
    events: list[dict[str, object]] = []
    async with client.stream("GET", f"/v1/generations/{task_id}/events") as stream:
        assert stream.status_code == 200
        async for line in stream.aiter_lines():
            if line.startswith("data:"):
                events.append(json.loads(line[len("data:") :].strip()))
    return events


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
    # CPU-stub producer never conditions geometry on the prompt/capture
    # (audit recommendation #2) -- this is the structured signal that would
    # have made the text-smoke gap visible without reading prose caveats.
    assert body["conditioning"] == "none"


async def test_list_generations_returns_produced_assets(
    client: httpx.AsyncClient,
) -> None:
    # Two generations; the stub producer writes l3.ply + sets produced=True.
    first = await client.post(
        "/v1/generations", json={"modality": "text", "prompt": "a brass gear"}
    )
    second = await client.post(
        "/v1/generations", json={"modality": "text", "prompt": "a stone lantern"}
    )
    first_id = first.json()["id"]
    second_id = second.json()["id"]
    # Wait for both background jobs to finish before listing the catalog.
    await _drain_events(client, first_id)
    await _drain_events(client, second_id)

    resp = await client.get("/v1/generations")
    assert resp.status_code == 200
    items = resp.json()
    ids = {item["id"] for item in items}
    assert {first_id, second_id} <= ids
    # Each listed asset is produced and has its viewable l3.ply on disk.
    for item in items:
        assert item["produced"] is True
        assert item["has_asset"] is True
        assert "prompt" in item and "created_at" in item


async def test_pricing_schedule(client: httpx.AsyncClient) -> None:
    resp = await client.get("/v1/pricing")
    assert resp.status_code == 200
    body = resp.json()
    assert body["credit_usd_rate"] == 0.01
    codes = {layer["code"] for layer in body["layers"]}
    assert {"L0", "L1", "L2", "L3"} <= codes
    assert "L0" in body["modes"]["preview"]
    assert "L3" in body["modes"]["refine"]


async def test_create_generation_defaults_to_refine_with_billing(
    client: httpx.AsyncClient,
) -> None:
    resp = await client.post(
        "/v1/generations",
        json={"modality": "text", "prompt": "a worn brass astrolabe"},
    )
    assert resp.status_code == 201
    task_id = resp.json()["id"]
    # Billing is finalised by the background job; read it back once it completes.
    await _drain_events(client, task_id)
    body = (await client.get(f"/v1/generations/{task_id}")).json()
    assert body["mode"] == "refine"
    billing = body["billing"]
    assert billing is not None
    codes = {item["code"] for item in billing["items"]}
    # Default refine bills the full delivered stack (L0 seed + L3 hero).
    assert {"L0", "L3"} <= codes
    assert billing["total_credits"] > 0
    # The ledger is also a downloadable artifact.
    assert any(a["name"] == "credit-ledger.json" for a in body["artifacts"])


async def test_preview_mode_bills_less_than_refine(
    client: httpx.AsyncClient,
) -> None:
    preview = await client.post(
        "/v1/generations",
        json={"modality": "text", "prompt": "a teapot", "mode": "preview"},
    )
    refine = await client.post(
        "/v1/generations",
        json={"modality": "text", "prompt": "a teapot", "mode": "refine"},
    )
    preview_id = preview.json()["id"]
    refine_id = refine.json()["id"]
    await _drain_events(client, preview_id)
    await _drain_events(client, refine_id)
    preview_billing = (await client.get(f"/v1/generations/{preview_id}")).json()[
        "billing"
    ]
    refine_billing = (await client.get(f"/v1/generations/{refine_id}")).json()[
        "billing"
    ]
    assert preview_billing["total_credits"] < refine_billing["total_credits"]
    # Preview never bills the L3 hero layer.
    p_codes = {i["code"] for i in preview_billing["items"]}
    assert "L3" not in p_codes


async def test_refine_of_bills_only_increment(client: httpx.AsyncClient) -> None:
    preview = await client.post(
        "/v1/generations",
        json={"modality": "text", "prompt": "a mug", "mode": "preview"},
    )
    preview_id = preview.json()["id"]
    await _drain_events(client, preview_id)
    refine = await client.post(
        "/v1/generations",
        json={
            "modality": "text",
            "prompt": "a mug",
            "mode": "refine",
            "refine_of": preview_id,
        },
    )
    refine_id = refine.json()["id"]
    assert refine.json()["refine_of"] == preview_id
    await _drain_events(client, refine_id)
    body = (await client.get(f"/v1/generations/{refine_id}")).json()
    codes = {i["code"] for i in body["billing"]["items"]}
    # Keyed refine pays for L3 only — the preview's L0 is not re-charged.
    assert codes == {"L3"}


async def test_get_generation_returns_persisted_billing(
    client: httpx.AsyncClient,
) -> None:
    create = await client.post(
        "/v1/generations",
        json={"modality": "text", "prompt": "a lantern", "mode": "preview"},
    )
    task_id = create.json()["id"]
    await _drain_events(client, task_id)
    fetched = await client.get(f"/v1/generations/{task_id}")
    assert fetched.status_code == 200
    body = fetched.json()
    assert body["mode"] == "preview"
    assert body["billing"] is not None
    assert body["billing"]["mode"] == "preview"


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


async def test_events_stream_reports_real_splat_count(
    client: httpx.AsyncClient,
) -> None:
    """The terminal SUCCEEDED metrics carry the producer's real splat count.

    Today the stub producer's count happens to equal the hardcoded
    _STAGE_TARGETS value (48_000 splats), so this also exercises the
    splats-passthrough path without needing a different fixture.
    """
    create = await client.post(
        "/v1/generations",
        json={"modality": "text", "prompt": "a teacup"},
    )
    task_id = create.json()["id"]

    events: list[dict[str, object]] = []
    async with client.stream("GET", f"/v1/generations/{task_id}/events") as stream:
        async for line in stream.aiter_lines():
            if line.startswith("data:"):
                events.append(json.loads(line[len("data:") :].strip()))

    assert events[-1]["status"] == "succeeded"
    assert events[-1]["message"] == "Asset ready"
    final_metrics = events[-1]["metrics"]
    assert isinstance(final_metrics, dict)
    assert final_metrics["splats"] == 48_000


async def test_events_stream_reports_failure_when_production_failed(
    client: httpx.AsyncClient,
) -> None:
    """A real production failure must yield a terminal FAILED SSE event.

    Simulates the producer raising (audit §2.6/§2.7): the background job
    persists the row with produced=False + the error message, and the SSE
    stream must end on a FAILED event carrying that real error -- never "Asset
    ready" with fabricated metrics, even though zero artifacts exist. The patch
    wraps the stream drain because production now runs in the background job
    while the SSE is consumed, not synchronously inside the POST.
    """
    with patch(
        "astel_api.jobs.produce_artifacts_dispatch",
        side_effect=RuntimeError("simulated CUDA OOM"),
    ):
        create = await client.post(
            "/v1/generations",
            json={"modality": "image", "prompt": "a ceramic teapot"},
        )
        assert create.status_code == 201
        task_id = create.json()["id"]
        # The POST returns before production runs: no billing yet.
        assert create.json()["billing"] is None
        assert create.json()["conditioning"] == "none"

        events = await _drain_events(client, task_id)

    assert events, "expected at least a terminal event"
    terminal = events[-1]
    assert terminal["status"] == "failed"
    assert terminal["message"] == "simulated CUDA OOM"
    assert terminal["metrics"] is None
    # No event ever claims success.
    assert all(e["status"] != "succeeded" for e in events)

    get_resp = await client.get(f"/v1/generations/{task_id}")
    assert get_resp.json()["status"] == "failed"


async def test_events_unknown_task_404(client: httpx.AsyncClient) -> None:
    resp = await client.get("/v1/generations/nope/events")
    assert resp.status_code == 404


# ---- _spec_longest_axis_m (metric-scale grounding seam) ----------------------


def test_spec_longest_axis_m_extracts_estimate() -> None:
    payload: dict[str, object] = {
        "status": "ok",
        "spec": {"target_scale": {"longest_axis_m": 0.18, "confidence": 0.4}},
    }
    assert _spec_longest_axis_m(payload) == 0.18


def test_spec_longest_axis_m_none_for_missing_or_skipped() -> None:
    skipped: dict[str, object] = {"status": "skipped"}
    no_spec: dict[str, object] = {"status": "ok"}
    empty_spec: dict[str, object] = {"status": "ok", "spec": {}}
    assert _spec_longest_axis_m(None) is None
    assert _spec_longest_axis_m(skipped) is None
    assert _spec_longest_axis_m(no_spec) is None
    assert _spec_longest_axis_m(empty_spec) is None


def test_spec_longest_axis_m_none_for_non_positive() -> None:
    zero: dict[str, object] = {
        "status": "ok",
        "spec": {"target_scale": {"longest_axis_m": 0.0}},
    }
    neg: dict[str, object] = {
        "status": "ok",
        "spec": {"target_scale": {"longest_axis_m": -1.0}},
    }
    assert _spec_longest_axis_m(zero) is None
    assert _spec_longest_axis_m(neg) is None


# ---- _l6_json_artifact_path (L6 layer binding seam) --------------------------


def test_l6_json_artifact_path_returns_path_when_ok(tmp_path: object) -> None:
    """A produced L6 layer (status ok) resolves to the stored l6.json path."""
    store = LocalArtifactStore(tmp_path)  # type: ignore[arg-type]
    store.put("t1", "l6.json", b"{}")
    payload: dict[str, object] = {"status": "ok"}
    result = _l6_json_artifact_path("t1", store, payload)
    assert result is not None
    assert result.name == "l6.json"


def test_l6_json_artifact_path_none_for_missing_or_skipped(tmp_path: object) -> None:
    """No L6 layer (None / skipped) -> no path, so the producer omits --l6-json."""
    store = LocalArtifactStore(tmp_path)  # type: ignore[arg-type]
    skipped: dict[str, object] = {"status": "skipped"}
    assert _l6_json_artifact_path("t2", store, None) is None
    assert _l6_json_artifact_path("t2", store, skipped) is None
