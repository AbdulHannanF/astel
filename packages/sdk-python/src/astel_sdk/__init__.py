"""Astel Python SDK — generate and retrieve layered Gaussian splat assets.

Quick start:
    from astel_sdk import AstelClient

    client = AstelClient("http://localhost:8000")
    gen = client.generate(prompt="a brass astrolabe on a wooden base")
    client.download_all_artifacts(gen.id, "out/")
"""

from .client import AstelClient, AsyncAstelClient
from .types import (
    ArtifactRef,
    BillingSummary,
    CaptureRef,
    CreateGenerationRequest,
    CreditLineItem,
    Generation,
    LayerPriceRef,
    PricingResource,
)

__version__ = "0.1.0"

__all__ = [
    "AstelClient",
    "AsyncAstelClient",
    "ArtifactRef",
    "BillingSummary",
    "CaptureRef",
    "CreateGenerationRequest",
    "CreditLineItem",
    "Generation",
    "LayerPriceRef",
    "PricingResource",
]
