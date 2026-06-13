"""Shared types/constants for the Temporal-backed task engine.

The stage list is DERIVED from :data:`astel_api.schemas.PIPELINE` so the
workflow/activities can never silently diverge from the stage definitions the
API and stub engine already use.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..schemas import PIPELINE, LayerStage, StageMetrics

# Ordered stage ids, derived from PIPELINE — single source of truth.
STAGE_IDS: tuple[str, ...] = tuple(spec.stage.value for spec in PIPELINE)

# Nominal duration (seconds) per stage id, derived from PIPELINE.
STAGE_SECONDS: dict[str, float] = {
    spec.stage.value: spec.nominal_seconds for spec in PIPELINE
}

# Terminal metrics each stage "achieves" once complete. Mirrors
# ``engine._STAGE_TARGETS`` so the Temporal path reports the same shapes as
# the stub path.
STAGE_TARGETS: dict[str, StageMetrics] = {
    LayerStage.L0_SEED.value: StageMetrics(splats=4_800, vram_gb=1.2),
    LayerStage.L1_DENSE.value: StageMetrics(splats=22_000, chamfer_mm=4.1, vram_gb=2.8),
    LayerStage.L2_COARSE.value: StageMetrics(
        splats=48_000, psnr_db=24.6, chamfer_mm=2.7, vram_gb=5.1
    ),
    LayerStage.L3_REFINED.value: StageMetrics(
        splats=48_000, psnr_db=31.2, chamfer_mm=0.9, vram_gb=7.4
    ),
}

# Ticks per stage for heartbeats — keeps heartbeat cadence reasonable without
# flooding the server (mirrors the stub engine's _TICKS_PER_STAGE).
TICKS_PER_STAGE = 6

TASK_QUEUE = "astel-pipeline"


@dataclass
class StageInput:
    """Activity input: which stage to run, scaled by ``sim_speed``."""

    stage: str
    task_id: str
    sim_speed: float = 1.0


@dataclass
class StageResult:
    """Activity output: terminal metrics for the completed stage."""

    stage: str
    task_id: str
    metrics: StageMetrics = field(default_factory=StageMetrics)


@dataclass
class WorkflowProgress:
    """Snapshot of pipeline progress, returned by the ``progress`` query."""

    completed_stages: list[str]
    current_stage: str | None
    current_index: int
    total: int
    done: bool
    failed: bool = False
    metrics: StageMetrics | None = None
