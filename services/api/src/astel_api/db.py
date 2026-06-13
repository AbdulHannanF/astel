"""SQLAlchemy 2 async engine, session, and ORM models.

Dev uses SQLite via aiosqlite; prod swaps in Postgres (asyncpg) purely by
changing ``ASTEL_DATABASE_URL`` — no model changes. Kept deliberately small for
M1: one ``generations`` table. Credits/users/assets land in later milestones.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import AsyncIterator

from sqlalchemy import String, Text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from .config import get_settings


class Base(DeclarativeBase):
    pass


class Generation(Base):
    """A single generation task row."""

    __tablename__ = "generations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    modality: Mapped[str] = mapped_column(String(16))
    prompt: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(16), default="queued")
    # Optional link to a prior /v1/captures upload (image/video bytes). Null for
    # text generations and any submit that did not reference a capture.
    capture_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(
        default=lambda: dt.datetime.now(dt.UTC)
    )


_settings = get_settings()
engine = create_async_engine(_settings.database_url, future=True)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def init_db() -> None:
    """Create tables if they don't exist (dev/test convenience).

    Schema changes are now tracked via Alembic (``services/api/migrations/``).
    Run ``uv run alembic upgrade head`` to apply migrations against a real
    (e.g. Postgres) database; see docs/architecture/ARCHITECTURE.md. This
    ``create_all`` remains for fresh dev SQLite DBs and the test suite, where
    a from-scratch schema is fine and migrations would be unnecessary ceremony.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency yielding a scoped async session."""
    async with SessionLocal() as session:
        yield session
