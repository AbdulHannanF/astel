"""Pydantic request/response schemas and pipeline-stage definitions.

The stage list mirrors the Astel layer model (CLAUDE.md §3): L0 Seed -> L1 Dense
-> L2 Coarse -> L3 Refined. M1 simulates these four; L4-L7 exist in the layer
model but are not produced by the stub pipeline yet.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class Modality(StrEnum):
    """Input modality for a generation (CLAUDE.md §0)."""

    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"


class TaskStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class LayerStage(StrEnum):
    """The pipeline stages the stub engine emits progress for."""

    L0_SEED = "L0_SEED"
    L1_DENSE = "L1_DENSE"
    L2_COARSE = "L2_COARSE"
    L3_REFINED = "L3_REFINED"


class StageSpec(BaseModel):
    """Static description of a stage: id, human label, layer, nominal duration."""

    stage: LayerStage
    layer: str
    label: str
    description: str
    nominal_seconds: float


# Ordered pipeline. Durations follow the brief's "cheap preview, expensive
# refine" shape: L0-L2 are quick, L3 dominates.
PIPELINE: tuple[StageSpec, ...] = (
    StageSpec(
        stage=LayerStage.L0_SEED,
        layer="L0",
        label="Seeding",
        description="Sparse point cloud from conditioning",
        nominal_seconds=2.0,
    ),
    StageSpec(
        stage=LayerStage.L1_DENSE,
        layer="L1",
        label="Densifying",
        description="Metric-scaled dense cloud with normals",
        nominal_seconds=3.5,
    ),
    StageSpec(
        stage=LayerStage.L2_COARSE,
        layer="L2",
        label="Coarse gaussians",
        description="Feed-forward gaussians from L1",
        nominal_seconds=3.0,
    ),
    StageSpec(
        stage=LayerStage.L3_REFINED,
        layer="L3",
        label="Refining surface",
        description="Surface-aligned optimization (hero layer)",
        nominal_seconds=8.0,
    ),
)


class CaptureRef(BaseModel):
    """A stored input capture (uploaded image/video bytes).

    Returned by ``POST /v1/captures``; ``capture_id`` is later threaded into a
    :class:`CreateGenerationRequest` so the (future) reconstruction pipeline can
    fetch the real input. In the stub it is stored and referenced but not yet
    consumed by the producer.
    """

    capture_id: str
    filename: str
    content_type: str
    bytes: int


class CreateGenerationRequest(BaseModel):
    modality: Modality
    prompt: str = Field(min_length=1, max_length=2000)
    # Optional reference to a prior ``POST /v1/captures`` upload. The stub
    # producer does not consume it yet (it still emits the procedural splat),
    # but the id is persisted so the capture→generation link is real and
    # threadable into the GPU path (M2).
    capture_id: str | None = None


class ArtifactRef(BaseModel):
    """A generated artifact, served via the artifact-serving route."""

    name: str
    url: str
    content_type: str
    bytes: int


class GenerationResource(BaseModel):
    id: str
    modality: Modality
    prompt: str
    status: TaskStatus
    created_at: str
    events_url: str
    artifacts: list[ArtifactRef] = []


class StageMetrics(BaseModel):
    """Fake-but-shaped per-stage telemetry (CLAUDE.md §10.3 measures everything)."""

    splats: int | None = None
    psnr_db: float | None = None
    chamfer_mm: float | None = None
    vram_gb: float | None = None
    wall_seconds: float | None = None


class ProgressEvent(BaseModel):
    """One SSE payload describing pipeline progress."""

    task_id: str
    status: TaskStatus
    stage: LayerStage | None
    stage_label: str | None
    stage_index: int
    stage_count: int
    progress: float = Field(ge=0.0, le=1.0)
    message: str
    metrics: StageMetrics | None = None
