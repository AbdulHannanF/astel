"""Runtime configuration, sourced from environment with sane dev defaults."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Process settings.

    Dev defaults to SQLite (aiosqlite). Set ``ASTEL_DATABASE_URL`` to a Postgres
    async URL (``postgresql+asyncpg://...``) for prod; the rest of the app is
    driver-agnostic via SQLAlchemy 2's async engine.
    """

    model_config = SettingsConfigDict(env_prefix="ASTEL_", env_file=".env")

    database_url: str = "sqlite+aiosqlite:///./astel_dev.db"

    # CORS: the Vite dev server. Vite proxies /v1 + /healthz in dev, so this is
    # belt-and-suspenders for any direct cross-origin calls.
    cors_origins: tuple[str, ...] = ("http://localhost:5173", "http://127.0.0.1:5173")

    # Speed multiplier for the simulated pipeline. 1.0 = realistic seconds;
    # tests crank this up so the SSE stream finishes near-instantly.
    sim_speed: float = 1.0

    # Which TaskEngine implementation to run. "stub" (default) needs no
    # external services; "temporal" requires a reachable Temporal server.
    engine: Literal["stub", "temporal"] = "stub"

    # Temporal connection settings (only used when engine == "temporal").
    temporal_address: str = "localhost:7233"
    temporal_namespace: str = "default"
    temporal_task_queue: str = "astel-pipeline"

    # Local filesystem root for generated artifacts (l3.ply, quality-report.json).
    # Swappable for an S3-backed store later via the ArtifactStore seam.
    artifact_dir: Path = Path("./.astel-artifacts")

    # Generation Spec LLM stage (CLAUDE.md §4). Offline by default: the stage
    # replays cached fixtures from ``llm_fixtures_dir`` and never spends. Going
    # LIVE (real Anthropic calls, real spend) is the founder gate R-O2 -- it
    # requires BOTH ``llm_live=True`` (ASTEL_LLM_LIVE) AND an ``ANTHROPIC_API_KEY``
    # in the environment, so an API key present for other reasons can never
    # silently trigger spend.
    llm_live: bool = False
    llm_fixtures_dir: Path = Path("./.astel-llm-fixtures")


@lru_cache
def get_settings() -> Settings:
    return Settings()
