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
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from . import __version__
from .billing import schedule_dict
from .config import Settings, get_settings
from .db import Generation, get_session, init_db
from .engine import InProcessStubEngine, TaskEngine, TemporalTaskEngine
from .jobs import (
    JOB_MANAGER,
    run_production_sync,
    submit_conditioning,
)
from .jobs import _l6_json_artifact_path as _l6_json_artifact_path
from .jobs import _spec_longest_axis_m as _spec_longest_axis_m
from .schemas import (
    PIPELINE,
    ArtifactRef,
    BillingSummary,
    CaptureRef,
    CreateGenerationRequest,
    GenerationMode,
    GenerationResource,
    GenerationSummary,
    LayerStage,
    Modality,
    PricingResource,
    ProgressEvent,
    StageMetrics,
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
    """Submit a generation. Returns immediately with a queued task + SSE URL.

    Production runs ASYNCHRONOUSLY in a background job (:mod:`astel_api.jobs`):
    this handler returns as soon as the row is persisted, so the request never
    blocks on the ~1-2 min GPU run. Real per-stage progress streams over the SSE
    events endpoint, and ``produced``/``splats``/``billing`` are finalised on the
    row when the job completes (read them back via ``GET /v1/generations/{id}``).

    The Temporal engine keeps the legacy synchronous-produce path: it produces
    inline here and the durable workflow drives the SSE.
    """
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

    if settings.engine == "temporal":
        # Legacy synchronous path (durable-engine deployment): produce inline,
        # persist the outcome, return the priced resource. The Temporal workflow
        # drives the SSE rail separately.
        result = run_production_sync(
            task_id,
            body.modality.value,
            body.prompt,
            store,
            settings,
            capture_id=body.capture_id,
            refine_of=body.refine_of,
            mode=body.mode.value,
        )
        row.produced = result.produced
        row.splats = result.splats
        row.production_error = result.error
        row.conditioning = result.conditioning
        if result.billing is not None:
            row.credits = result.billing.total_credits
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
            billing=result.billing,
            conditioning=_conditioning_of(result.conditioning),
        )

    # Default async path: persist a best-effort conditioning label (so the Truth
    # Meter pill is honest immediately), kick off the background job, and return.
    conditioning = submit_conditioning(
        body.modality.value, body.prompt, store, body.capture_id
    )
    row.conditioning = conditioning
    await session.commit()

    JOB_MANAGER.submit(
        task_id,
        body.modality.value,
        body.prompt,
        store,
        settings,
        capture_id=body.capture_id,
        refine_of=body.refine_of,
        mode=body.mode.value,
    )

    return GenerationResource(
        id=task_id,
        modality=body.modality,
        prompt=body.prompt,
        status=TaskStatus.QUEUED,
        created_at=row.created_at.isoformat(),
        events_url=f"/v1/generations/{task_id}/events",
        artifacts=[],
        mode=body.mode,
        refine_of=body.refine_of,
        billing=None,
        conditioning=_conditioning_of(conditioning),
    )


@app.get(
    "/v1/generations",
    tags=["generations"],
    response_model=list[GenerationSummary],
)
async def list_generations(
    session: SessionDep, store: StoreDep, limit: int = 200
) -> list[GenerationSummary]:
    """List produced generations, newest first — the gallery catalog source.

    This is what makes every generated splat show up in the gallery by default:
    the gallery fetches this list and renders one tile per asset. Only rows that
    actually produced their viewable ``l3.ply`` on disk are returned, so the
    catalog never links to a failed or empty task. Capped at ``limit`` (1..500,
    default 200) newest rows.
    """
    capped = max(1, min(limit, 500))
    result = await session.execute(
        select(Generation)
        .where(Generation.produced.is_(True))
        .order_by(Generation.created_at.desc())
        .limit(capped)
    )
    summaries: list[GenerationSummary] = []
    for row in result.scalars().all():
        # Skip rows whose viewable artifact is gone (store pruned, partial write).
        if store.path_for(row.id, "l3.ply") is None:
            continue
        summaries.append(
            GenerationSummary(
                id=row.id,
                modality=Modality(row.modality),
                prompt=row.prompt,
                created_at=row.created_at.isoformat(),
                produced=row.produced,
                splats=row.splats,
                conditioning=_conditioning_of(row.conditioning),
                has_asset=True,
            )
        )
    return summaries


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


def _terminal_event_from_row(row: Generation | None) -> ProgressEvent:
    """Rebuild a single terminal :class:`ProgressEvent` from a persisted row.

    Used when an SSE client connects to a job that is no longer held in memory
    (finished and evicted, or the process restarted): the true outcome still
    lives on the row, so the reconnect resolves honestly rather than hanging.
    """
    count = len(PIPELINE)
    if row is None or not row.produced:
        return ProgressEvent(
            task_id=row.id if row is not None else "",
            status=TaskStatus.FAILED,
            stage=None,
            stage_label=None,
            stage_index=0,
            stage_count=count,
            progress=0.0,
            message=(
                row.production_error
                if row is not None and row.production_error
                else "Generation produced no artifacts"
            ),
        )
    return ProgressEvent(
        task_id=row.id,
        status=TaskStatus.SUCCEEDED,
        stage=LayerStage.L3_REFINED,
        stage_label="Complete",
        stage_index=count,
        stage_count=count,
        progress=1.0,
        message="Asset ready",
        metrics=StageMetrics(splats=row.splats),
    )


@app.get("/v1/generations/{task_id}/events", tags=["generations"])
async def generation_events(
    task_id: str,
    session: SessionDep,
    engine: EngineDep,
    settings: Annotated[Settings, Depends(get_settings)],
) -> EventSourceResponse:
    """Stream pipeline progress as Server-Sent Events.

    Each ``message`` event carries a JSON :class:`ProgressEvent`; the connection
    closes on a terminal status. The default path streams the *real* events
    published by the background job (:mod:`astel_api.jobs`); a reconnect to an
    already-evicted job replays a single terminal event from the row. The
    Temporal engine instead drives the rail from its durable workflow.
    """
    row = await session.get(Generation, task_id)
    if row is None:
        raise HTTPException(status_code=404, detail="generation not found")

    if settings.engine == "temporal":

        async def temporal_stream() -> AsyncIterator[dict[str, str]]:
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
                    terminal = event.status in (
                        TaskStatus.SUCCEEDED,
                        TaskStatus.FAILED,
                    )
                    if terminal and db_row is not None:
                        db_row.status = event.status.value
                        await inner.commit()

        return EventSourceResponse(temporal_stream())

    async def job_stream() -> AsyncIterator[dict[str, str]]:
        if JOB_MANAGER.has(task_id):
            async for event in JOB_MANAGER.stream(task_id):
                yield {"event": "progress", "data": event.model_dump_json()}
            return
        # Not in memory: rebuild the terminal state from the persisted row.
        async with get_session_cm() as inner:
            db_row = await inner.get(Generation, task_id)
        event = _terminal_event_from_row(db_row)
        yield {"event": "progress", "data": event.model_dump_json()}

    return EventSourceResponse(job_stream())


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
