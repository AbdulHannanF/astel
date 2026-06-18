"""Tests for the artifact producer, store, and serving routes."""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest

# Point the app at a throwaway db + fast sim + tmp artifact dir BEFORE importing.
os.environ["ASTEL_DATABASE_URL"] = "sqlite+aiosqlite:///./astel_test_artifacts.db"
os.environ["ASTEL_SIM_SPEED"] = "400"

import httpx  # noqa: E402
from astel_format.package import AstelPackage  # noqa: E402
from astel_splat_io.ply import read_ply  # noqa: E402
from astel_splat_io.spz import read_spz  # noqa: E402
from httpx import ASGITransport  # noqa: E402

from astel_api.config import get_settings  # noqa: E402
from astel_api.db import init_db  # noqa: E402
from astel_api.main import app  # noqa: E402
from astel_api.producer import (  # noqa: E402
    build_package_quality_report,
    build_quality_report,
    produce_artifacts,
    seed_cloud,
    stable_seed,
    synth_cloud,
)
from astel_api.storage import LocalArtifactStore, _store_for_root  # noqa: E402


@pytest.fixture(autouse=True)
def _artifact_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    monkeypatch.setenv("ASTEL_ARTIFACT_DIR", str(tmp_path))
    get_settings.cache_clear()
    _store_for_root.cache_clear()
    yield tmp_path
    get_settings.cache_clear()
    _store_for_root.cache_clear()


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    await init_db()
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def test_synth_cloud_deterministic() -> None:
    cloud_a = synth_cloud(stable_seed("task-one"))
    cloud_b = synth_cloud(stable_seed("task-one"))
    assert (cloud_a.positions == cloud_b.positions).all()

    cloud_c = synth_cloud(stable_seed("task-two"))
    assert not (cloud_a.positions == cloud_c.positions).all()


def test_produced_ply_roundtrips(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    summary = produce_artifacts("task-roundtrip", "text", "a brass astrolabe", store)

    ply_path = store.path_for("task-roundtrip", "l3.ply")
    assert ply_path is not None
    cloud = read_ply(ply_path)
    assert cloud.count == summary["splats"]


def test_seed_cloud_is_sparse_subsample() -> None:
    cloud = synth_cloud(stable_seed("task-seed"))
    l0 = seed_cloud(cloud)
    assert 0 < l0.count < cloud.count
    # L0 points are a strict subset of L3 points (strided subsample).
    assert (l0.positions[0] == cloud.positions[0]).all()


def test_produces_full_layer_stack(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    summary = produce_artifacts("task-stack", "image", "a ceramic teapot", store)

    names = set(store.list_names("task-stack"))
    assert {
        "l0.ply",
        "l3.ply",
        "l3.spz",
        "l3.sog",
        "package.astel",
        "quality-report.json",
    } <= names
    assert summary["splats"] > summary["seed_splats"] > 0

    # The SPZ export round-trips back to a cloud of the L3 count.
    spz_path = store.path_for("task-stack", "l3.spz")
    assert spz_path is not None
    assert read_spz(spz_path).count == summary["splats"]

    # The KHR_gaussian_splatting glTF export is emitted and round-trips.
    from astel_splat_io.gltf import read_gltf

    glb_path = store.path_for("task-stack", "l3.glb")
    assert glb_path is not None
    assert read_gltf(glb_path).count == summary["splats"]


def test_astel_package_validates_and_binds_layers(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    summary = produce_artifacts("task-pkg", "text", "a brass astrolabe", store)

    astel_path = store.path_for("task-pkg", "package.astel")
    assert astel_path is not None
    # AstelPackage.read re-validates the manifest against the JSON Schema.
    pkg = AstelPackage.read(astel_path)
    assert pkg.manifest.layers.l3 is not None
    assert pkg.manifest.layers.l0 is not None
    assert pkg.manifest.layers.l3.count == summary["splats"]
    assert pkg.manifest.layers.l0.count == summary["seed_splats"]
    # Honest scale provenance: 100% generated, nothing measured.
    assert pkg.manifest.quality_report.hallucination.measured_fraction == 0.0
    assert pkg.manifest.quality_report.hallucination.generated_fraction == 1.0


def test_quality_report_is_honest() -> None:
    report = build_quality_report(count=48_000, modality="text")
    assert report["origin"] == "stub"
    assert isinstance(report["caveats"], list)
    assert len(report["caveats"]) > 0


def test_package_quality_report_metrics_are_null_with_reason() -> None:
    """The typed package report must not fabricate measured numbers (§10.4)."""
    qr = build_package_quality_report(modality="text")
    assert qr.geometric_error.chamfer_mm is None
    assert qr.geometric_error.reason  # non-empty reason required when null
    assert qr.caveats is not None and len(qr.caveats) > 0


async def test_generation_exposes_artifacts(client: httpx.AsyncClient) -> None:
    resp = await client.post(
        "/v1/generations",
        json={"modality": "text", "prompt": "a worn brass astrolabe"},
    )
    assert resp.status_code == 201
    body = resp.json()
    names = {a["name"] for a in body["artifacts"]}
    assert "l3.ply" in names
    assert "quality-report.json" in names
    for artifact in body["artifacts"]:
        expected_url = f"/v1/generations/{body['id']}/artifacts/{artifact['name']}"
        assert artifact["url"] == expected_url
        assert artifact["bytes"] > 0


async def test_get_artifact_serves_and_404s(client: httpx.AsyncClient) -> None:
    create = await client.post(
        "/v1/generations",
        json={"modality": "image", "prompt": "a ceramic teapot"},
    )
    task_id = create.json()["id"]

    resp = await client.get(f"/v1/generations/{task_id}/artifacts/l3.ply")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/octet-stream"
    assert len(resp.content) > 0

    report_resp = await client.get(
        f"/v1/generations/{task_id}/artifacts/quality-report.json"
    )
    assert report_resp.status_code == 200
    report = json.loads(report_resp.content)
    assert report["origin"] == "stub"

    missing = await client.get(f"/v1/generations/{task_id}/artifacts/missing.ply")
    assert missing.status_code == 404

    traversal = await client.get(
        f"/v1/generations/{task_id}/artifacts/..%2f..%2fsecret"
    )
    assert traversal.status_code in (400, 404)

    traversal2 = await client.get(f"/v1/generations/{task_id}/artifacts/../x")
    assert traversal2.status_code in (400, 404)


async def test_new_exports_served_as_octet_stream(client: httpx.AsyncClient) -> None:
    create = await client.post(
        "/v1/generations",
        json={"modality": "text", "prompt": "a brass astrolabe"},
    )
    task_id = create.json()["id"]
    for name in ("l0.ply", "l3.spz", "l3.sog", "package.astel"):
        resp = await client.get(f"/v1/generations/{task_id}/artifacts/{name}")
        assert resp.status_code == 200, name
        assert resp.headers["content-type"] == "application/octet-stream", name
        assert len(resp.content) > 0, name


async def test_capture_upload_returns_reference(client: httpx.AsyncClient) -> None:
    payload = b"\x89PNG\r\n\x1a\n fake image bytes"
    resp = await client.post(
        "/v1/captures",
        files={"file": ("orbit.PNG", payload, "image/png")},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["capture_id"].startswith("capture-")
    assert body["filename"] == "orbit.PNG"
    assert body["content_type"] == "image/png"
    assert body["bytes"] == len(payload)


async def test_capture_rejects_empty_upload(client: httpx.AsyncClient) -> None:
    resp = await client.post(
        "/v1/captures",
        files={"file": ("empty.png", b"", "image/png")},
    )
    assert resp.status_code == 400


async def test_capture_id_flows_into_generation(client: httpx.AsyncClient) -> None:
    cap = await client.post(
        "/v1/captures",
        files={"file": ("clip.mp4", b"fake video bytes", "video/mp4")},
    )
    capture_id = cap.json()["capture_id"]

    gen = await client.post(
        "/v1/generations",
        json={
            "modality": "video",
            "prompt": "an orbit capture",
            "capture_id": capture_id,
        },
    )
    assert gen.status_code == 201
    task_id = gen.json()["id"]

    # The link is persisted on the generation row.
    fetched = await client.get(f"/v1/generations/{task_id}")
    assert fetched.status_code == 200
    # capture_id is internal (not in GenerationResource), but the stub still
    # produces its splat stack regardless of the capture.
    names = {a["name"] for a in fetched.json()["artifacts"]}
    assert "l3.ply" in names
