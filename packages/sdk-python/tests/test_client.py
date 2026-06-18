"""Unit tests for AstelClient / AsyncAstelClient using respx mock transport."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx

from astel_sdk import AsyncAstelClient
from astel_sdk.types import CaptureRef, Generation

BASE = "http://test-astel"

GEN_PAYLOAD = {
    "id": "abc-123",
    "modality": "text",
    "prompt": "a brass astrolabe",
    "status": "succeeded",
    "created_at": "2026-06-18T00:00:00",
    "events_url": "/v1/generations/abc-123/events",
    "artifacts": [
        {"name": "l3.ply", "url": "/v1/generations/abc-123/artifacts/l3.ply",
         "content_type": "application/octet-stream", "bytes": 1234},
        {"name": "quality-report.json",
         "url": "/v1/generations/abc-123/artifacts/quality-report.json",
         "content_type": "application/json", "bytes": 512},
    ],
    "mode": "refine",
    "refine_of": None,
    "billing": {
        "mode": "refine",
        "refine_of": None,
        "items": [
            {"code": "L3", "label": "Refined surface gaussians", "tier": "refine",
             "credits": 20.0, "usd": 0.20, "detail": ""},
        ],
        "total_credits": 21.0,
        "total_usd": 0.21,
        "credit_usd_rate": 0.01,
        "caveats": [],
    },
    "conditioning": "prompt",
}


@pytest.fixture
def gen_json() -> str:
    return json.dumps(GEN_PAYLOAD)


@respx.mock
@pytest.mark.asyncio
async def test_generate_returns_generation(gen_json: str) -> None:
    respx.post(f"{BASE}/v1/generations").mock(
        return_value=httpx.Response(201, text=gen_json)
    )
    async with AsyncAstelClient(BASE) as client:
        gen = await client.generate(prompt="a brass astrolabe")
    assert isinstance(gen, Generation)
    assert gen.id == "abc-123"
    assert gen.is_ready


@respx.mock
@pytest.mark.asyncio
async def test_get_generation(gen_json: str) -> None:
    respx.get(f"{BASE}/v1/generations/abc-123").mock(
        return_value=httpx.Response(200, text=gen_json)
    )
    async with AsyncAstelClient(BASE) as client:
        gen = await client.get_generation("abc-123")
    assert gen.id == "abc-123"
    assert len(gen.artifacts) == 2


@respx.mock
@pytest.mark.asyncio
async def test_artifact_url_helper(gen_json: str) -> None:
    respx.get(f"{BASE}/v1/generations/abc-123").mock(
        return_value=httpx.Response(200, text=gen_json)
    )
    async with AsyncAstelClient(BASE) as client:
        gen = await client.get_generation("abc-123")
    url = gen.artifact_url("l3.ply")
    assert url is not None
    assert "l3.ply" in url


@respx.mock
@pytest.mark.asyncio
async def test_download_artifact(tmp_path: Path, gen_json: str) -> None:
    content = b"PLY binary data here"
    respx.get(f"{BASE}/v1/generations/abc-123/artifacts/l3.ply").mock(
        return_value=httpx.Response(200, content=content)
    )
    async with AsyncAstelClient(BASE) as client:
        out = await client.download_artifact("abc-123", "l3.ply", tmp_path / "l3.ply")
    assert out.read_bytes() == content


@respx.mock
@pytest.mark.asyncio
async def test_upload_capture() -> None:
    capture_payload = {
        "capture_id": "capture-xyz",
        "filename": "photo.jpg",
        "content_type": "image/jpeg",
        "bytes": 42,
    }
    respx.post(f"{BASE}/v1/captures").mock(
        return_value=httpx.Response(201, json=capture_payload)
    )
    async with AsyncAstelClient(BASE) as client:
        cap = await client.upload_capture(b"fake-image-bytes", "photo.jpg")
    assert isinstance(cap, CaptureRef)
    assert cap.capture_id == "capture-xyz"


@respx.mock
@pytest.mark.asyncio
async def test_health() -> None:
    respx.get(f"{BASE}/healthz").mock(
        return_value=httpx.Response(
            200, json={"status": "ok", "service": "astel-api", "version": "0.1.0"}
        )
    )
    async with AsyncAstelClient(BASE) as client:
        h = await client.health()
    assert h["status"] == "ok"


@respx.mock
@pytest.mark.asyncio
async def test_pricing() -> None:
    # Mirrors the real GET /v1/pricing payload (astel_api.billing.schedule_dict).
    payload = {
        "schema": "astel.pricing/v0",
        "credit_usd_rate": 0.01,
        "layers": [
            {"code": "L0", "label": "Seed cloud", "tier": "preview", "credits": 1.0},
            {"code": "L3", "label": "Refined", "tier": "refine", "credits": 20.0},
        ],
        "modes": {"preview": ["L0", "L1", "L2"], "refine": ["L3", "L4"]},
        "notes": ["L0–L2 previews are cheap; L3 refine is the main spend."],
    }
    respx.get(f"{BASE}/v1/pricing").mock(
        return_value=httpx.Response(200, json=payload)
    )
    async with AsyncAstelClient(BASE) as client:
        p = await client.pricing()
    assert p.credit_usd_rate == 0.01
    assert {layer.code for layer in p.layers} == {"L0", "L3"}
    assert p.modes["refine"] == ["L3", "L4"]


def test_generation_is_ready_for_succeeded() -> None:
    gen = Generation.model_validate({**GEN_PAYLOAD, "status": "SUCCEEDED"})
    assert gen.is_ready
    assert not gen.is_failed


def test_generation_is_failed() -> None:
    gen = Generation.model_validate({**GEN_PAYLOAD, "status": "FAILED"})
    assert gen.is_failed
    assert not gen.is_ready


def test_generation_artifact_url_miss() -> None:
    gen = Generation.model_validate(GEN_PAYLOAD)
    assert gen.artifact_url("does-not-exist.glb") is None


def test_billing_summary_parses_line_items() -> None:
    """Billing mirrors the API: line items + credit_usd_rate are preserved."""
    gen = Generation.model_validate(GEN_PAYLOAD)
    assert gen.billing is not None
    assert gen.billing.total_credits == 21.0
    assert gen.billing.credit_usd_rate == 0.01
    assert len(gen.billing.items) == 1
    assert gen.billing.items[0].code == "L3"
