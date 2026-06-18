"""FastAPI application: health, generation submission, and SSE progress.

Routes are thin; the pipeline runs behind the :class:`TaskEngine` seam. The SSE
endpoint re-runs the stub engine per connection (M1 simplicity); with Temporal
the same endpoint will subscribe to a durable workflow's event history instead.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from . import __version__
from .billing import price_generation, schedule_dict
from .config import Settings, get_settings
from .db import Generation, get_session, init_db
from .engine import InProcessStubEngine, TaskEngine, TemporalTaskEngine
from .generation_spec_stage import (
    apply_spec_scale_to_report,
    run_generation_spec_stage,
)
from .gpu_producer import produce_artifacts_dispatch
from .physics_material_stage import run_physics_material_stage
from .schemas import (
    PIPELINE,
    ArtifactRef,
    BillingSummary,
    CaptureRef,
    CreateGenerationRequest,
    GenerationMode,
    GenerationResource,
    Modality,
    PricingResource,
    StageSpec,
    TaskStatus,
)
from .storage import ArtifactStore, get_artifact_store

logger = logging.getLogger("astel_api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    await init_db()
    yield


app = FastAPI(
    title="Astel API",
    version=__version__,
    summary="Layered Gaussian-splat generation gateway",
    lifespan=lifespan,
)

_settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(_settings.cors_origins),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def get_engine(
    settings: Annotated[Settings, Depends(get_settings)],
) -> TaskEngine:
    """Provide the active task engine, selected by ``settings.engine``."""
    if settings.engine == "temporal":
        return TemporalTaskEngine(
            address=settings.temporal_address,
            namespace=settings.temporal_namespace,
            task_queue=settings.temporal_task_queue,
            sim_speed=settings.sim_speed,
        )
    return InProcessStubEngine(sim_speed=settings.sim_speed)


def get_store(
    settings: Annotated[Settings, Depends(get_settings)],
) -> ArtifactStore:
    """Provide the active artifact store."""
    return get_artifact_store(settings)


SessionDep = Annotated[AsyncSession, Depends(get_session)]
EngineDep = Annotated[TaskEngine, Depends(get_engine)]
StoreDep = Annotated[ArtifactStore, Depends(get_store)]


_CONTENT_TYPES: dict[str, str] = {
    ".ply": "application/octet-stream",
    ".json": "application/json",
    # No IANA-registered media types exist for these splat/package formats yet
    # (SPZ is on the Khronos standards track; .astel is our own zip-based
    # package — see manifest-v0 §1, which assigns the internal mimetype member
    # "application/vnd.astel.package+zip"). Serve them as opaque binary.
    ".spz": "application/octet-stream",
    ".sog": "application/octet-stream",
    ".astel": "application/octet-stream",
}


def _content_type_for(name: str) -> str:
    return _CONTENT_TYPES.get(Path(name).suffix, "application/octet-stream")


_LEDGER_ARTIFACT = "credit-ledger.json"

_CONDITIONING_VALUES = frozenset({"prompt", "image", "video", "none"})


def _conditioning_of(
    row_value: str | None,
) -> Literal["prompt", "image", "video", "none"] | None:
    """Narrow a DB ``conditioning`` string to the schema's closed literal.

    Returns ``None`` for rows written before this column existed (or any
    unexpected value) -- callers must treat ``None`` as "unknown", not as a
    fourth honest value.
    """
    if row_value in _CONDITIONING_VALUES:
        return row_value  # type: ignore[return-value]
    return None


def _spec_longest_axis_m(spec_payload: dict[str, object] | None) -> float | None:
    """Pull the metric longest-axis estimate (metres) from a successful spec.

    Returns ``None`` unless the Generation Spec stage produced a usable positive
    size estimate -- so the producer stays honestly ungrounded rather than
    fabricating a metric scale (CLAUDE.md §10.4). Used to ground the produced
    asset's L5/L6 mass + package ``meters_per_unit``.
    """
    if not spec_payload or spec_payload.get("status") != "ok":
        return None
    spec = spec_payload.get("spec")
    if not isinstance(spec, dict):
        return None
    target_scale = spec.get("target_scale")
    if not isinstance(target_scale, dict):
        return None
    value = target_scale.get("longest_axis_m")
    if isinstance(value, int | float) and not isinstance(value, bool) and value > 0:
        return float(value)
    return None


def _l6_json_artifact_path(
    task_id: str, store: ArtifactStore, physics_payload: dict[str, object] | None
) -> Path | None:
    """Resolve the stored ``l6.json`` path when the physics stage produced one.

    The physics-material stage writes ``l6.json`` only on success (status
    ``"ok"``); a fixture-miss / skip writes a non-billable note instead. Returns
    the local path so the GPU producer can bind the L6 layer into the package and
    run the L6<->L5 mass join, or ``None`` (no L6 data, or a non-local store --
    the same seam where an S3 store would download to a temp file first).
    """
    if not physics_payload or physics_payload.get("status") != "ok":
        return None
    return store.path_for(task_id, "l6.json")


def _build_and_store_billing(
    task_id: str,
    mode: str,
    refine_of: str | None,
    store: ArtifactStore,
    spec_payload: dict[str, object] | None,
) -> BillingSummary:
    """Price the generation from delivered artifacts, store + return the ledger.

    The LLM line is folded in only when the Generation Spec stage actually
    incurred a measured cost this task (text path, fixture hit / live). A refine
    keyed on a prior preview never runs the spec stage, so it never re-charges
    it (CLAUDE.md §7).
    """
    llm_cost_usd: float | None = None
    if spec_payload and spec_payload.get("status") == "ok":
        ledger = spec_payload.get("ledger")
        if isinstance(ledger, dict):
            cost = ledger.get("cost_usd")
            if isinstance(cost, (int, float)):
                llm_cost_usd = float(cost)

    credit_ledger = price_generation(
        mode=mode,
        delivered_artifacts=store.list_names(task_id),
        llm_cost_usd=llm_cost_usd,
        refine_of=refine_of,
    )
    store.put(
        task_id,
        _LEDGER_ARTIFACT,
        json.dumps(credit_ledger.to_dict(), indent=2).encode("utf-8"),
    )
    return BillingSummary.model_validate(credit_ledger.to_dict())


def _load_billing(task_id: str, store: ArtifactStore) -> BillingSummary | None:
    """Return the stored credit ledger as a :class:`BillingSummary`, if any."""
    path = store.path_for(task_id, _LEDGER_ARTIFACT)
    if path is None:
        return None
    try:
        return BillingSummary.model_validate(json.loads(path.read_text()))
    except Exception:  # a malformed ledger must not break the read path
        logger.exception("failed to load billing for %s", task_id)
        return None


def _artifacts_for(task_id: str, store: ArtifactStore) -> list[ArtifactRef]:
    """Build :class:`ArtifactRef` entries for everything currently on disk."""
    refs: list[ArtifactRef] = []
    for name in store.list_names(task_id):
        path = store.path_for(task_id, name)
        if path is None:
            continue
        refs.append(
            ArtifactRef(
                name=name,
                url=f"/v1/generations/{task_id}/artifacts/{name}",
                content_type=_content_type_for(name),
                bytes=path.stat().st_size,
            )
        )
    return refs


@app.get("/healthz", tags=["meta"])
async def healthz() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok", "service": "astel-api", "version": __version__}


@app.get("/v1/pipeline", tags=["meta"], response_model=list[StageSpec])
async def pipeline() -> list[StageSpec]:
    """The static stage definitions the client renders on its progress rail."""
    return list(PIPELINE)


@app.get("/v1/pricing", tags=["meta"], response_model=PricingResource)
async def pricing() -> PricingResource:
    """The credit price schedule: per-layer costs + preview/refine tiers."""
    return PricingResource.model_validate(schedule_dict())


# Captures are stored in the same artifact store under a dedicated namespace so
# their bytes share the LocalArtifactStore -> S3 swap seam. The raw upload is
# saved under a fixed "source<ext>" name; the original filename is preserved
# only as returned metadata (the store's name regex forbids most real
# filenames, and a fixed name avoids any traversal surface).
_CAPTURE_PREFIX = "capture-"
_DEFAULT_CAPTURE_NAME = "source"


def _sanitize_capture_ext(filename: str | None) -> str:
    """Return a safe ``.ext`` (lowercased, validated) for a capture filename.

    Only the suffix is kept and only if it matches the store's name charset;
    anything else yields ``""`` (extension-less). The stored member name is
    always ``source<ext>`` — never the user-supplied name — so this cannot
    introduce a path-traversal or odd-character key.
    """
    if not filename:
        return ""
    suffix = Path(filename).suffix.lower()
    # suffix includes the leading dot; allow only dot + alphanumerics.
    if suffix and all(c.isalnum() or c == "." for c in suffix):
        return suffix
    return ""


@app.post("/v1/captures", tags=["captures"], response_model=CaptureRef, status_code=201)
async def create_capture(file: UploadFile, store: StoreDep) -> CaptureRef:
    """Upload raw input bytes (an image or video) for a later generation.

    Stores the bytes in the artifact store under a generated ``capture_id``
    namespace and returns a reference. The returned ``capture_id`` is threaded
    into ``POST /v1/generations`` via its optional ``capture_id`` field. The
    stub producer does not consume the bytes yet — but the upload is real and
    the id really flows through (the M2 GPU path will fetch them).
    """
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty upload")

    capture_id = f"{_CAPTURE_PREFIX}{uuid.uuid4()}"
    ext = _sanitize_capture_ext(file.filename)
    name = f"{_DEFAULT_CAPTURE_NAME}{ext}"
    store.put(capture_id, name, data)

    return CaptureRef(
        capture_id=capture_id,
        filename=file.filename or name,
        content_type=file.content_type or "application/octet-stream",
        bytes=len(data),
    )


@app.post(
    "/v1/generations",
    tags=["generations"],
    response_model=GenerationResource,
    status_code=201,
)
async def create_generation(
    body: CreateGenerationRequest,
    session: SessionDep,
    store: StoreDep,
    settings: Annotated[Settings, Depends(get_settings)],
) -> GenerationResource:
    """Submit a generation. Returns a task id and its SSE events URL."""
    task_id = str(uuid.uuid4())
    row = Generation(
        id=task_id,
        modality=body.modality.value,
        prompt=body.prompt,
        status=TaskStatus.QUEUED.value,
        capture_id=body.capture_id,
        mode=body.mode.value,
        refine_of=body.refine_of,
    )
    session.add(row)
    await session.commit()

    billing: BillingSummary | None = None
    try:
        # Text-pipeline Generation Spec stage (offline by default; founder-gated
        # for live spend) runs FIRST so its metric size estimate can ground the
        # produced asset's scale + L5/L6 mass (CLAUDE.md §3 L1, §4: the spec
        # conditions generation). A refine keyed on a prior preview inherits that
        # preview's spec, so it is skipped here — the LLM spend (and the credit
        # line) belong to the preview, not the refine (CLAUDE.md §7).
        spec_payload: dict[str, object] | None = None
        physics_payload: dict[str, object] | None = None
        if body.refine_of is None:
            spec_payload = run_generation_spec_stage(
                task_id, body.modality.value, body.prompt, store, settings
            )
            # L6 physics-material reasons over the Generation Spec (not the
            # produced asset), so it runs BEFORE produce: this lets the producer
            # bind the l6.json layer into the .astel package AND compute the
            # L6<->L5 mass join (write_layer_stack reads l6.json from its output
            # dir). Running it after produce -- as it did previously -- left L6
            # unbound in every shipped package and the mass join firing only in
            # tests. A refine keyed on a preview inherits that preview's L6 and so
            # is skipped here (the LLM spend belongs to the preview, CLAUDE.md §7).
            physics_payload = run_physics_material_stage(
                task_id, body.modality.value, spec_payload, store, settings
            )

        production = produce_artifacts_dispatch(
            task_id,
            body.modality.value,
            body.prompt,
            store,
            capture_id=body.capture_id,
            longest_axis_m=_spec_longest_axis_m(spec_payload),
            l6_json_path=_l6_json_artifact_path(task_id, store, physics_payload),
        )
        row.produced = True
        row.splats = production.get("splats")
        row.production_error = None
        conditioning = production.get("conditioning")
        row.conditioning = conditioning if isinstance(conditioning, str) else "none"
        await session.commit()

        if body.refine_of is None:
            # Thread the LLM size estimate into the freshly-written quality report
            # (best-effort scale patch; the L6 layer is already bound by produce).
            apply_spec_scale_to_report(task_id, store, spec_payload)
        # Price the generation by its billing tier and store the credit ledger.
        billing = _build_and_store_billing(
            task_id, body.mode.value, body.refine_of, store, spec_payload
        )
        row.credits = billing.total_credits
        await session.commit()
    except Exception as exc:  # production failure must not fail the submit
        logger.exception("artifact production failed for %s", task_id)
        row.produced = False
        row.splats = None
        row.production_error = str(exc)
        row.conditioning = "none"
        await session.commit()

    return GenerationResource(
        id=task_id,
        modality=body.modality,
        prompt=body.prompt,
        status=TaskStatus.QUEUED,
        created_at=row.created_at.isoformat(),
        events_url=f"/v1/generations/{task_id}/events",
        artifacts=_artifacts_for(task_id, store),
        mode=body.mode,
        refine_of=body.refine_of,
        billing=billing,
        conditioning=_conditioning_of(row.conditioning),
    )


@app.get("/v1/generations/{task_id}", tags=["generations"])
async def get_generation(
    task_id: str, session: SessionDep, store: StoreDep
) -> GenerationResource:
    row = await session.get(Generation, task_id)
    if row is None:
        raise HTTPException(status_code=404, detail="generation not found")
    return GenerationResource(
        id=row.id,
        modality=Modality(row.modality),
        prompt=row.prompt,
        status=TaskStatus(row.status),
        created_at=row.created_at.isoformat(),
        events_url=f"/v1/generations/{task_id}/events",
        artifacts=_artifacts_for(task_id, store),
        mode=GenerationMode(row.mode),
        refine_of=row.refine_of,
        billing=_load_billing(task_id, store),
        conditioning=_conditioning_of(row.conditioning),
    )


@app.get("/v1/generations/{task_id}/events", tags=["generations"])
async def generation_events(
    task_id: str, session: SessionDep, engine: EngineDep
) -> EventSourceResponse:
    """Stream pipeline progress as Server-Sent Events.

    Each ``message`` event carries a JSON :class:`ProgressEvent`. The connection
    closes when the engine reaches a terminal status. The DB row's status is
    advanced to RUNNING on first connect and SUCCEEDED on completion.
    """
    row = await session.get(Generation, task_id)
    if row is None:
        raise HTTPException(status_code=404, detail="generation not found")

    async def event_stream() -> AsyncIterator[dict[str, str]]:
        async with get_session_cm() as inner:
            db_row = await inner.get(Generation, task_id)
            if db_row is not None and db_row.status == TaskStatus.QUEUED.value:
                db_row.status = TaskStatus.RUNNING.value
                await inner.commit()

            produced = db_row.produced if db_row is not None else True
            splats = db_row.splats if db_row is not None else None
            production_error = (
                db_row.production_error if db_row is not None else None
            )

            async for event in engine.run(
                task_id,
                produced=produced,
                splats=splats,
                error=production_error,
            ):
                yield {"event": "progress", "data": event.model_dump_json()}
                terminal = event.status in (TaskStatus.SUCCEEDED, TaskStatus.FAILED)
                if terminal and db_row is not None:
                    db_row.status = event.status.value
                    await inner.commit()

    return EventSourceResponse(event_stream())


@app.get("/v1/generations/{task_id}/artifacts/{name}", tags=["generations"])
async def get_artifact(task_id: str, name: str, store: StoreDep) -> FileResponse:
    """Serve a generated artifact (e.g. ``l3.ply``, ``quality-report.json``)."""
    try:
        path = store.path_for(task_id, name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="invalid artifact name") from exc
    if path is None:
        raise HTTPException(status_code=404, detail="artifact not found")
    return FileResponse(path, media_type=_content_type_for(name))


# Standalone session context manager for the generator (the Depends-injected
# session is closed when the request handler returns, before the stream drains).
def get_session_cm() -> AsyncSession:
    from .db import SessionLocal

    return SessionLocal()


def main() -> None:
    """Entry point for ``astel-api`` console script / ``python -m``."""
    import uvicorn

    uvicorn.run("astel_api.main:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
