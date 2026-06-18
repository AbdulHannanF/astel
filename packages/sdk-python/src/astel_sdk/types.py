"""Typed response models mirroring the Astel REST API schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class ArtifactRef(BaseModel):
    name: str
    url: str
    content_type: str
    bytes: int


class CreditLineItem(BaseModel):
    """One charge on a generation's credit ledger (mirrors the API)."""

    code: str
    label: str
    tier: str
    credits: float
    usd: float
    detail: str = ""


class BillingSummary(BaseModel):
    """The credit accounting returned with a generation (CLAUDE.md §7).

    Mirrors ``astel_api.schemas.BillingSummary`` exactly.
    """

    mode: str
    refine_of: str | None = None
    items: list[CreditLineItem] = []
    total_credits: float
    total_usd: float
    credit_usd_rate: float = 0.01
    caveats: list[str] = []


class Generation(BaseModel):
    id: str
    modality: Literal["text", "image", "video"]
    prompt: str | None
    status: str
    created_at: str
    events_url: str
    artifacts: list[ArtifactRef] = []
    mode: str = "refine"
    refine_of: str | None = None
    billing: BillingSummary | None = None
    conditioning: str | None = None

    @property
    def is_ready(self) -> bool:
        return self.status in ("succeeded", "SUCCEEDED")

    @property
    def is_failed(self) -> bool:
        return self.status in ("failed", "FAILED")

    def artifact_url(self, name: str) -> str | None:
        for a in self.artifacts:
            if a.name == name:
                return a.url
        return None


class CaptureRef(BaseModel):
    capture_id: str
    filename: str
    content_type: str
    bytes: int


class LayerPriceRef(BaseModel):
    """A layer's credit cost in the public price schedule."""

    code: str
    label: str
    tier: str
    credits: float


class PricingResource(BaseModel):
    """The ``/v1/pricing`` schedule (mirrors ``astel_api.schemas.PricingResource``)."""

    credit_usd_rate: float
    layers: list[LayerPriceRef] = []
    modes: dict[str, list[str]] = {}
    notes: list[str] = []


class CreateGenerationRequest(BaseModel):
    modality: Literal["text", "image", "video"] = "text"
    prompt: str | None = None
    capture_id: str | None = None
    mode: Literal["preview", "refine"] = "refine"
    refine_of: str | None = None
