"""Pydantic request/response schemas and pipeline-stage definitions.

The stage list mirrors the Astel layer model (CLAUDE.md §3): L0 Seed -> L1 Dense
-> L2 Coarse -> L3 Refined. M1 simulates these four; L4-L7 exist in the layer
model but are not produced by the stub pipeline yet.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class Modality(StrEnum):
    """Input modality for a generation (CLAUDE.md §0)."""

    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"


class GenerationMode(StrEnum):
    """Billing tier for a generation (CLAUDE.md §7, Meshy two-stage model).

    ``preview`` charges only the cheap L0–L2 exploration layers; ``refine``
    charges the L3 hero layer (+ any add-ons), and when keyed on a prior preview
    via ``refine_of`` it pays only for the new work. See ``billing.py``.
    """

    PREVIEW = "preview"
    REFINE = "refine"


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
    # Billing tier (CLAUDE.md §7). Defaults to ``refine`` (a full generation);
    # set ``preview`` for the cheap L0–L2 exploration tier.
    mode: GenerationMode = GenerationMode.REFINE
    # When this is a follow-up refine of a prior preview, its task id. Billing
    # then charges only the new refine work, never re-charging the preview.
    refine_of: str | None = None


class ArtifactRef(BaseModel):
    """A generated artifact, served via the artifact-serving route."""

    name: str
    url: str
    content_type: str
    bytes: int


class CreditLineItem(BaseModel):
    """One charge on a generation's credit ledger (see ``billing.py``)."""

    code: str
    label: str
    tier: str
    credits: float
    usd: float
    detail: str = ""


class BillingSummary(BaseModel):
    """The credit accounting returned with a generation (CLAUDE.md §7)."""

    mode: GenerationMode
    refine_of: str | None = None
    items: list[CreditLineItem] = []
    total_credits: float
    total_usd: float
    credit_usd_rate: float
    caveats: list[str] = []


class GenerationResource(BaseModel):
    id: str
    modality: Modality
    prompt: str
    status: TaskStatus
    created_at: str
    events_url: str
    artifacts: list[ArtifactRef] = []
    mode: GenerationMode = GenerationMode.REFINE
    refine_of: str | None = None
    billing: BillingSummary | None = None
    # What the L3 geometry was actually conditioned on for this task (audit
    # recommendation #2). "none" means a prompt/capture-independent
    # placeholder was produced — distinct from "this asset is unconditioned
    # by design" being buried only in prose caveats.
    conditioning: Literal["prompt", "image", "video", "none"] | None = None


class GenerationSummary(BaseModel):
    """A compact generation record for the catalog list (``GET /v1/generations``).

    Lighter than :class:`GenerationResource` (no artifact list / billing) so the
    gallery can render every produced asset cheaply. ``has_asset`` is True only
    when the viewable ``l3.ply`` is actually on disk, so the gallery never links
    to an asset whose production failed.
    """

    id: str
    modality: Modality
    prompt: str
    created_at: str
    produced: bool
    splats: int | None = None
    conditioning: Literal["prompt", "image", "video", "none"] | None = None
    has_asset: bool = False


class LayerPriceRef(BaseModel):
    """A layer's credit cost in the public price schedule."""

    code: str
    label: str
    tier: str
    credits: float


class PricingResource(BaseModel):
    """The ``/v1/pricing`` schedule: per-layer credit costs + mode tiers."""

    credit_usd_rate: float
    layers: list[LayerPriceRef]
    modes: dict[str, list[str]]
    notes: list[str] = []


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
