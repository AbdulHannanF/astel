"""In-process async generation jobs: real production + real streamed progress.

This is the default (non-Temporal) task engine for on-site generation. It
replaces the old design where ``POST /v1/generations`` ran the *entire*
production synchronously (blocking the request for the full ~1-2 min GPU run)
and the SSE endpoint then *replayed a fixed animation* of work that had already
finished.

Instead:

* ``POST /v1/generations`` creates the row, calls :meth:`JobManager.submit`, and
  returns immediately with status ``queued``.
* :meth:`JobManager.submit` schedules a background ``asyncio`` task that runs the
  blocking pipeline (Generation Spec -> physics-material -> the GPU/stub
  producer -> billing) in a thread executor, publishing :class:`ProgressEvent`s
  as it crosses each real stage boundary, with an asymptotic easing fill during
  the long opaque produce step so the bar moves without ever claiming "done".
* The SSE endpoint subscribes to the job and streams those *real* events. The
  terminal event reflects the true outcome: ``succeeded`` only when the asset
  actually exists on disk, ``failed`` (with the real error) otherwise -- never a
  fabricated "Asset ready".

The heavy GPU produce step is serialised by a process-wide semaphore so two
concurrent submissions on a single-GPU box cannot fight over VRAM (the second
waits, which is the honest behaviour).
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import time
from collections import OrderedDict
from collections.abc import AsyncIterator
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from .billing import price_generation
from .config import Settings
from .db import Generation, SessionLocal
from .engine import _STAGE_TARGETS
from .generation_spec_stage import (
    apply_spec_scale_to_report,
    run_generation_spec_stage,
)
from .gpu_producer import (
    _gpu_conditioning,
    _resolve_capture_image,
    produce_artifacts_dispatch,
)
from .physics_material_stage import run_physics_material_stage
from .schemas import (
    PIPELINE,
    BillingSummary,
    LayerStage,
    ProgressEvent,
    StageMetrics,
    TaskStatus,
)
from .storage import ArtifactStore

logger = logging.getLogger("astel_api.jobs")

_LEDGER_ARTIFACT = "credit-ledger.json"

# How many finished jobs to retain in memory so a late / reconnecting SSE client
# can still replay the full event history. Older jobs are evicted (a reconnect
# to an evicted job falls back to a single terminal event rebuilt from the DB).
_MAX_RETAINED_JOBS = 256

# Nominal seconds the produce step is expected to take, per producer, used only
# to shape the easing curve (never to claim completion). The GPU generative path
# is ~1-2 min; the CPU stub is sub-second.
_NOMINAL_PRODUCE_SECONDS_GPU = 90.0
_NOMINAL_PRODUCE_SECONDS_STUB = 6.0


# ---------------------------------------------------------------------------
# Production (synchronous core, run inside a thread executor)
# ---------------------------------------------------------------------------


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
    run the L6<->L5 mass join, or ``None`` (no L6 data, or a non-local store).
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
    """Price the generation from delivered artifacts, store + return the ledger."""
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


@dataclass
class ProductionResult:
    """Outcome of :func:`run_production_sync` (no DB coupling)."""

    produced: bool
    splats: int | None
    conditioning: str
    error: str | None
    billing: BillingSummary | None


def run_production_sync(
    task_id: str,
    modality: str,
    prompt: str,
    store: ArtifactStore,
    settings: Settings,
    *,
    capture_id: str | None,
    refine_of: str | None,
    mode: str,
) -> ProductionResult:
    """Run the full production pipeline for one task. Never raises.

    This is the blocking core shared by the async job runner (executed in a
    thread) and the Temporal engine branch (called inline). It runs the
    Generation Spec + physics-material stages (skipped for a keyed refine, whose
    LLM spend belongs to its preview -- CLAUDE.md §7), invokes the active
    producer (stub or GPU), patches the LLM scale estimate into the report, and
    prices the delivered stack. On any failure it returns ``produced=False`` with
    the error string so the caller can persist an honest failure (CLAUDE.md
    §10.4) -- the SSE then surfaces it rather than claiming success.
    """
    try:
        spec_payload: dict[str, object] | None = None
        physics_payload: dict[str, object] | None = None
        if refine_of is None:
            spec_payload = run_generation_spec_stage(
                task_id, modality, prompt, store, settings
            )
            physics_payload = run_physics_material_stage(
                task_id, modality, spec_payload, store, settings
            )

        production = produce_artifacts_dispatch(
            task_id,
            modality,
            prompt,
            store,
            capture_id=capture_id,
            longest_axis_m=_spec_longest_axis_m(spec_payload),
            l6_json_path=_l6_json_artifact_path(task_id, store, physics_payload),
        )
        conditioning = production.get("conditioning")
        conditioning = conditioning if isinstance(conditioning, str) else "none"
        splats = production.get("splats")

        if refine_of is None:
            apply_spec_scale_to_report(task_id, store, spec_payload)
        billing = _build_and_store_billing(
            task_id, mode, refine_of, store, spec_payload
        )
        return ProductionResult(
            produced=True,
            splats=splats if isinstance(splats, int) else None,
            conditioning=conditioning,
            error=None,
            billing=billing,
        )
    except Exception as exc:  # production failure must not crash the worker
        logger.exception("artifact production failed for %s", task_id)
        return ProductionResult(
            produced=False,
            splats=None,
            conditioning="none",
            error=str(exc),
            billing=None,
        )


def submit_conditioning(
    modality: str,
    prompt: str,
    store: ArtifactStore,
    capture_id: str | None,
) -> str:
    """Best-effort conditioning label known *before* production runs.

    Lets the POST response (and the Truth Meter pill) state honestly what the
    geometry will be conditioned on, without waiting for the producer. The real
    value from the producer overwrites it on completion. Mirrors the GPU
    producer's own logic; the stub is always ``"none"``.
    """
    if os.environ.get("ASTEL_PRODUCER") != "gpu":
        return "none"
    image_path = _resolve_capture_image(capture_id, store)
    return _gpu_conditioning(modality, prompt, image_path)


# ---------------------------------------------------------------------------
# Async job manager (background runner + progress broadcast)
# ---------------------------------------------------------------------------


def _stage_metrics_for_splats(splats: int | None) -> StageMetrics:
    """Terminal L3 metrics, with the producer's real splat count if known."""
    target = _STAGE_TARGETS[LayerStage.L3_REFINED]
    if splats is not None:
        return target.model_copy(update={"splats": splats})
    return target


def _stage_for_progress(overall: float) -> LayerStage:
    """Map an overall fraction onto the layer rail for the active-stage marker."""
    if overall < 0.12:
        return LayerStage.L0_SEED
    if overall < 0.28:
        return LayerStage.L1_DENSE
    if overall < 0.58:
        return LayerStage.L2_COARSE
    return LayerStage.L3_REFINED


_STAGE_INDEX = {spec.stage: i for i, spec in enumerate(PIPELINE)}
_STAGE_LABEL = {spec.stage: spec.label for spec in PIPELINE}
_STAGE_DESC = {spec.stage: spec.description for spec in PIPELINE}


class _JobState:
    """Per-task event log + a condition for fan-out to SSE subscribers."""

    def __init__(self) -> None:
        self.events: list[ProgressEvent] = []
        self.cond = asyncio.Condition()
        self.terminal = False
        self.task: asyncio.Task[None] | None = None


class JobManager:
    """Owns the in-flight (and recently-finished) generation jobs."""

    def __init__(self) -> None:
        self._jobs: OrderedDict[str, _JobState] = OrderedDict()
        # One in-flight heavy produce at a time, to protect single-GPU VRAM.
        self._produce_sema = asyncio.Semaphore(1)

    def has(self, task_id: str) -> bool:
        return task_id in self._jobs

    def submit(
        self,
        task_id: str,
        modality: str,
        prompt: str,
        store: ArtifactStore,
        settings: Settings,
        *,
        capture_id: str | None,
        refine_of: str | None,
        mode: str,
    ) -> None:
        """Schedule the background runner for ``task_id`` (returns immediately)."""
        state = _JobState()
        self._jobs[task_id] = state
        while len(self._jobs) > _MAX_RETAINED_JOBS:
            self._jobs.popitem(last=False)
        state.task = asyncio.create_task(
            self._run(
                state,
                task_id,
                modality,
                prompt,
                store,
                settings,
                capture_id=capture_id,
                refine_of=refine_of,
                mode=mode,
            )
        )

    async def _publish(self, state: _JobState, event: ProgressEvent) -> None:
        async with state.cond:
            state.events.append(event)
            if event.status in (TaskStatus.SUCCEEDED, TaskStatus.FAILED):
                state.terminal = True
            state.cond.notify_all()

    def _event(
        self,
        task_id: str,
        *,
        status: TaskStatus,
        stage: LayerStage | None,
        progress: float,
        message: str,
        metrics: StageMetrics | None = None,
    ) -> ProgressEvent:
        count = len(PIPELINE)
        index = _STAGE_INDEX[stage] if stage is not None else 0
        return ProgressEvent(
            task_id=task_id,
            status=status,
            stage=stage,
            stage_label=_STAGE_LABEL.get(stage) if stage else None,
            stage_index=index,
            stage_count=count,
            progress=round(max(0.0, min(1.0, progress)), 4),
            message=message,
            metrics=metrics,
        )

    async def _run(
        self,
        state: _JobState,
        task_id: str,
        modality: str,
        prompt: str,
        store: ArtifactStore,
        settings: Settings,
        *,
        capture_id: str | None,
        refine_of: str | None,
        mode: str,
    ) -> None:
        loop = asyncio.get_running_loop()
        is_gpu = os.environ.get("ASTEL_PRODUCER") == "gpu"
        nominal = (
            _NOMINAL_PRODUCE_SECONDS_GPU if is_gpu else _NOMINAL_PRODUCE_SECONDS_STUB
        )
        try:
            await self._publish(
                state,
                self._event(
                    task_id,
                    status=TaskStatus.RUNNING,
                    stage=None,
                    progress=0.0,
                    message="Generation accepted; starting pipeline",
                ),
            )
            # Cosmetic-but-ordered rail steps for the cheap conditioning tiers,
            # emitted before the heavy produce begins so the rail reads L0->L3.
            for stage, frac in (
                (LayerStage.L0_SEED, 0.05),
                (LayerStage.L1_DENSE, 0.12),
                (LayerStage.L2_COARSE, 0.20),
            ):
                await self._publish(
                    state,
                    self._event(
                        task_id,
                        status=TaskStatus.RUNNING,
                        stage=stage,
                        progress=frac,
                        message=_STAGE_DESC[stage],
                    ),
                )
                await asyncio.sleep(0.05)

            # The heavy, opaque produce step runs in a thread; ease the bar while
            # it runs (asymptotic, capped < 1.0 -- it can never claim "done").
            ctx = self._produce_sema if is_gpu else nullcontext()
            emitted_l3 = False
            async with ctx:
                fut = loop.run_in_executor(
                    None,
                    partial(
                        run_production_sync,
                        task_id,
                        modality,
                        prompt,
                        store,
                        settings,
                        capture_id=capture_id,
                        refine_of=refine_of,
                        mode=mode,
                    ),
                )
                start = time.monotonic()
                while not fut.done():
                    elapsed = time.monotonic() - start
                    eased = 1.0 - math.exp(-elapsed / nominal)
                    overall = 0.20 + 0.72 * eased  # 0.20 -> 0.92
                    stage = _stage_for_progress(overall)
                    if stage == LayerStage.L3_REFINED:
                        emitted_l3 = True
                    await self._publish(
                        state,
                        self._event(
                            task_id,
                            status=TaskStatus.RUNNING,
                            stage=stage,
                            progress=overall,
                            message=_STAGE_DESC[stage],
                        ),
                    )
                    await asyncio.sleep(0.5)
                result: ProductionResult = await fut

            await self._finalize_db(task_id, result)

            if not result.produced:
                await self._publish(
                    state,
                    self._event(
                        task_id,
                        status=TaskStatus.FAILED,
                        stage=None,
                        progress=0.0,
                        message=result.error or "Generation produced no artifacts",
                    ),
                )
                return

            # Guarantee the refine (L3) stage is observed even on the fast stub
            # path where the executor returned before any easing tick fired.
            if not emitted_l3:
                await self._publish(
                    state,
                    self._event(
                        task_id,
                        status=TaskStatus.RUNNING,
                        stage=LayerStage.L3_REFINED,
                        progress=0.95,
                        message=_STAGE_DESC[LayerStage.L3_REFINED],
                    ),
                )
            await self._publish(
                state,
                self._event(
                    task_id,
                    status=TaskStatus.SUCCEEDED,
                    stage=LayerStage.L3_REFINED,
                    progress=1.0,
                    message="Asset ready",
                    metrics=_stage_metrics_for_splats(result.splats),
                ),
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("generation job crashed for %s", task_id)
            await self._finalize_db(
                task_id,
                ProductionResult(False, None, "none", str(exc), None),
            )
            await self._publish(
                state,
                self._event(
                    task_id,
                    status=TaskStatus.FAILED,
                    stage=None,
                    progress=0.0,
                    message=str(exc),
                ),
            )

    async def _finalize_db(self, task_id: str, result: ProductionResult) -> None:
        """Persist the production outcome to the generation row."""
        async with SessionLocal() as session:
            row = await session.get(Generation, task_id)
            if row is None:
                return
            row.produced = result.produced
            row.splats = result.splats
            row.conditioning = result.conditioning
            row.production_error = result.error
            row.status = (
                TaskStatus.SUCCEEDED.value
                if result.produced
                else TaskStatus.FAILED.value
            )
            if result.billing is not None:
                row.credits = result.billing.total_credits
            await session.commit()

    async def stream(self, task_id: str) -> AsyncIterator[ProgressEvent]:
        """Yield this job's events as they are published, until terminal.

        Replays the full history first (so a late subscriber to an already
        finished, still-retained job sees every stage), then follows live.
        """
        state = self._jobs.get(task_id)
        if state is None:  # evicted or never submitted here
            return
        cursor = 0
        while True:
            async with state.cond:
                while cursor >= len(state.events) and not state.terminal:
                    await state.cond.wait()
                pending = state.events[cursor:]
                cursor = len(state.events)
                terminal = state.terminal
            for event in pending:
                yield event
            if terminal and cursor >= len(state.events):
                return


# Module-level singleton: one manager per process.
JOB_MANAGER = JobManager()
