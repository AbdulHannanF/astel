"""Astel MCP server — exposes Astel generation as MCP tools.

Tools:
  generate_asset  — submit a text or image generation, return the generation ID.
  get_asset       — poll a generation by ID; return status + artifact list.
  list_pricing    — return the credit price schedule.

Run with:
  astel-mcp                        # stdio transport (default, for Claude Desktop)
  astel-mcp --transport sse        # SSE transport for web integrations

The server reads ASTEL_API_URL (default http://localhost:8000) and
ASTEL_API_KEY (optional bearer token).
"""

from __future__ import annotations

import os
import sys
from typing import Any

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:
    sys.exit(
        "astel-mcp requires the 'mcp' extra: pip install 'astel-sdk[mcp]'\n"
        f"Error: {exc}"
    )

from .client import AsyncAstelClient

_BASE_URL = os.environ.get("ASTEL_API_URL", "http://localhost:8000")
_API_KEY = os.environ.get("ASTEL_API_KEY")

mcp = FastMCP("Astel")


def _client() -> AsyncAstelClient:
    return AsyncAstelClient(_BASE_URL, api_key=_API_KEY)


@mcp.tool()
async def generate_asset(
    prompt: str,
    modality: str = "text",
    mode: str = "refine",
    capture_id: str | None = None,
) -> dict[str, Any]:
    """Generate a layered Gaussian splat asset.

    Args:
        prompt:     Text description of the object to generate.
        modality:   'text' (default), 'image', or 'video'.
        mode:       'refine' (default, full pipeline) or 'preview' (fast, cheap).
        capture_id: For image/video modalities: the capture_id returned by
                    the /v1/captures endpoint (upload image first).

    Returns a dict with:
        generation_id: str   — poll with get_asset().
        status: str
        artifacts: list[str] — artifact names immediately available (may be empty
                               until the pipeline completes).
        billing: dict        — credit cost summary.
    """
    async with _client() as c:
        gen = await c.generate(
            prompt=prompt,
            modality=modality,
            mode=mode,
            capture_id=capture_id,
        )
    return {
        "generation_id": gen.id,
        "status": gen.status,
        "artifacts": [a.name for a in gen.artifacts],
        "billing": gen.billing.model_dump() if gen.billing else None,
        "conditioning": gen.conditioning,
    }


@mcp.tool()
async def get_asset(generation_id: str) -> dict[str, Any]:
    """Get the status and artifacts of a generation.

    Args:
        generation_id: The ID returned by generate_asset.

    Returns a dict with:
        status: str             — 'queued', 'running', 'succeeded', 'failed'.
        artifacts: list[dict]   — [{name, url, bytes}, …].
        quality_report_url: str — URL for the Truth Meter quality report JSON.
        billing: dict | None
    """
    async with _client() as c:
        gen = await c.get_generation(generation_id)
    qr_url = gen.artifact_url("quality-report.json")
    return {
        "status": gen.status,
        "is_ready": gen.is_ready,
        "is_failed": gen.is_failed,
        "artifacts": [
            {"name": a.name, "url": _BASE_URL.rstrip("/") + a.url, "bytes": a.bytes}
            for a in gen.artifacts
        ],
        "quality_report_url": (_BASE_URL.rstrip("/") + qr_url) if qr_url else None,
        "billing": gen.billing.model_dump() if gen.billing else None,
    }


@mcp.tool()
async def list_pricing() -> dict[str, Any]:
    """Return the live Astel credit price schedule from GET /v1/pricing.

    Credits are the internal billing unit (1 credit ≈ 1¢ notional). The
    returned dict carries ``credit_usd_rate``, the per-layer ``layers`` costs,
    the preview/refine ``modes`` tiers, and ``notes`` — fetched live, never
    hard-coded here (the schedule is authoritative on the server).
    """
    async with _client() as c:
        pricing = await c.pricing()
    return pricing.model_dump()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Astel MCP server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="MCP transport (default: stdio)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    args = parser.parse_args()

    if args.transport == "sse":
        mcp.run(transport="sse", host=args.host, port=args.port)
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
