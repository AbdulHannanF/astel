"""Async and synchronous clients for the Astel REST API.

Async (recommended):
    async with AsyncAstelClient("http://localhost:8000") as client:
        gen = await client.generate(prompt="a brass astrolabe")
        await client.download_artifact(gen.id, "l3.ply", Path("out.ply"))

Sync (convenience wrapper):
    client = AstelClient("http://localhost:8000")
    gen = client.generate(prompt="a brass astrolabe")
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, BinaryIO

import httpx

from .types import (
    ArtifactRef,
    CaptureRef,
    CreateGenerationRequest,
    Generation,
    PricingResource,
)


class AsyncAstelClient:
    """Async HTTP client wrapping the Astel REST API."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        *,
        timeout: float = 120.0,
        api_key: str | None = None,
    ) -> None:
        headers: dict[str, str] = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(
            base_url=base_url, timeout=timeout, headers=headers
        )

    async def __aenter__(self) -> AsyncAstelClient:
        await self._client.__aenter__()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self._client.__aexit__(*args)

    async def close(self) -> None:
        await self._client.aclose()

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    async def health(self) -> dict[str, str]:
        r = await self._client.get("/healthz")
        r.raise_for_status()
        return dict(r.json())

    # ------------------------------------------------------------------
    # Pricing
    # ------------------------------------------------------------------

    async def pricing(self) -> PricingResource:
        r = await self._client.get("/v1/pricing")
        r.raise_for_status()
        return PricingResource.model_validate(r.json())

    # ------------------------------------------------------------------
    # Captures
    # ------------------------------------------------------------------

    async def upload_capture(
        self,
        file: BinaryIO | bytes,
        filename: str = "capture.jpg",
        content_type: str = "image/jpeg",
    ) -> CaptureRef:
        """Upload a raw image or video for use in an image/video generation."""
        data = file if isinstance(file, bytes) else file.read()
        r = await self._client.post(
            "/v1/captures",
            files={"file": (filename, data, content_type)},
        )
        r.raise_for_status()
        return CaptureRef.model_validate(r.json())

    # ------------------------------------------------------------------
    # Generations
    # ------------------------------------------------------------------

    async def generate(
        self,
        *,
        prompt: str | None = None,
        modality: str = "text",
        capture_id: str | None = None,
        mode: str = "refine",
        refine_of: str | None = None,
    ) -> Generation:
        """Submit a generation and return the initial resource.

        For a text generation, provide ``prompt``.
        For an image generation, upload first with :meth:`upload_capture`
        and pass the returned ``capture_id``.
        """
        body = CreateGenerationRequest(
            modality=modality,  # type: ignore[arg-type]
            prompt=prompt,
            capture_id=capture_id,
            mode=mode,  # type: ignore[arg-type]
            refine_of=refine_of,
        )
        r = await self._client.post(
            "/v1/generations",
            content=body.model_dump_json(),
            headers={"Content-Type": "application/json"},
        )
        r.raise_for_status()
        return Generation.model_validate(r.json())

    async def get_generation(self, generation_id: str) -> Generation:
        r = await self._client.get(f"/v1/generations/{generation_id}")
        r.raise_for_status()
        return Generation.model_validate(r.json())

    async def wait_for_generation(
        self,
        generation_id: str,
        poll_interval: float = 2.0,
        max_wait: float = 600.0,
    ) -> Generation:
        """Poll until the generation is terminal (succeeded or failed)."""
        waited = 0.0
        while waited < max_wait:
            gen = await self.get_generation(generation_id)
            if gen.is_ready or gen.is_failed:
                return gen
            await asyncio.sleep(poll_interval)
            waited += poll_interval
        raise TimeoutError(
            f"generation {generation_id} did not complete within {max_wait}s"
        )

    # ------------------------------------------------------------------
    # Artifacts
    # ------------------------------------------------------------------

    async def list_artifacts(self, generation_id: str) -> list[ArtifactRef]:
        gen = await self.get_generation(generation_id)
        return gen.artifacts

    async def download_artifact(
        self,
        generation_id: str,
        artifact_name: str,
        dest: str | Path,
    ) -> Path:
        """Download a named artifact to ``dest``.  Returns the written path."""
        r = await self._client.get(
            f"/v1/generations/{generation_id}/artifacts/{artifact_name}"
        )
        r.raise_for_status()
        path = Path(dest)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(r.content)
        return path

    async def download_all_artifacts(
        self,
        generation_id: str,
        dest_dir: str | Path,
    ) -> list[Path]:
        """Download all artifacts to ``dest_dir``.  Returns written paths."""
        dest = Path(dest_dir)
        gen = await self.get_generation(generation_id)
        paths = []
        for art in gen.artifacts:
            out = await self.download_artifact(generation_id, art.name, dest / art.name)
            paths.append(out)
        return paths


class AstelClient:
    """Synchronous wrapper around :class:`AsyncAstelClient`."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        *,
        timeout: float = 120.0,
        api_key: str | None = None,
    ) -> None:
        self._async = AsyncAstelClient(base_url, timeout=timeout, api_key=api_key)

    def _run(self, coro: Any) -> Any:  # noqa: ANN401
        return asyncio.run(coro)

    def health(self) -> dict[str, str]:
        return asyncio.run(self._async.health())

    def pricing(self) -> PricingResource:
        return asyncio.run(self._async.pricing())

    def upload_capture(
        self,
        file: BinaryIO | bytes,
        filename: str = "capture.jpg",
        content_type: str = "image/jpeg",
    ) -> CaptureRef:
        return asyncio.run(self._async.upload_capture(file, filename, content_type))

    def generate(
        self,
        *,
        prompt: str | None = None,
        modality: str = "text",
        capture_id: str | None = None,
        mode: str = "refine",
        refine_of: str | None = None,
    ) -> Generation:
        return asyncio.run(
            self._async.generate(
                prompt=prompt,
                modality=modality,
                capture_id=capture_id,
                mode=mode,
                refine_of=refine_of,
            )
        )

    def get_generation(self, generation_id: str) -> Generation:
        return asyncio.run(self._async.get_generation(generation_id))

    def wait_for_generation(
        self,
        generation_id: str,
        poll_interval: float = 2.0,
        max_wait: float = 600.0,
    ) -> Generation:
        return asyncio.run(
            self._async.wait_for_generation(generation_id, poll_interval, max_wait)
        )

    def list_artifacts(self, generation_id: str) -> list[ArtifactRef]:
        return asyncio.run(self._async.list_artifacts(generation_id))

    def download_artifact(
        self,
        generation_id: str,
        artifact_name: str,
        dest: str | Path,
    ) -> Path:
        return asyncio.run(
            self._async.download_artifact(generation_id, artifact_name, dest)
        )

    def download_all_artifacts(
        self,
        generation_id: str,
        dest_dir: str | Path,
    ) -> list[Path]:
        return asyncio.run(self._async.download_all_artifacts(generation_id, dest_dir))
