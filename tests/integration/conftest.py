"""
Shared fixtures and configuration for integration tests.

Integration tests require:
1. A running PostgreSQL + pgvector instance (see docker-compose.yml, port 5434).
2. Alembic migrations applied (`uv run alembic upgrade head`).

If the database is unreachable, all integration tests are automatically
skipped with a clear message — they do not fail hard in a CI environment
that does not have a database configured.

Run integration tests with:
    uv run pytest tests/integration/ -m integration -v

Skip integration tests (unit only):
    uv run pytest -m "not integration and not model_integration" -v

Default Docker Compose database URL (port 5434 avoids conflict with local postgres):
    postgresql+asyncpg://documind:documind@localhost:5434/documind
"""

from __future__ import annotations

import asyncio
import os

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from db.session import get_session_factory


def _integration_db_url() -> str:
    """Return the database URL for integration tests."""
    return os.environ.get(
        "DOCUMIND_DATABASE_URL",
        "postgresql+asyncpg://documind:documind@localhost:5434/documind",
    )


def _db_is_reachable(url: str) -> bool:
    """Return True if we can connect to the database at *url*."""
    import asyncpg  # type: ignore[import]

    async def _check() -> bool:
        try:
            pg_url = url.replace("postgresql+asyncpg://", "postgresql://")
            conn = await asyncpg.connect(pg_url, timeout=3)
            await conn.close()
            return True
        except Exception:
            return False

    try:
        return asyncio.run(_check())
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Skip decorator — evaluated once at collection time
# ─────────────────────────────────────────────────────────────────────────────

_db_url = _integration_db_url()
_db_available = _db_is_reachable(_db_url)

requires_db = pytest.mark.skipif(
    not _db_available,
    reason=(
        "Integration tests require PostgreSQL + pgvector on port 5434. "
        "Start with: docker compose up -d && uv run alembic upgrade head"
    ),
)


# ─────────────────────────────────────────────────────────────────────────────
# Session fixture — function-scoped to avoid cross-loop contamination.
#
# asyncpg connections are tied to the event loop they were created on.
# pytest-asyncio (with asyncio_mode=auto) creates a new event loop per test.
# A shared (session-scoped) engine would hold connections from loop #1 that
# cannot be reused in loop #2 → "Future attached to a different loop".
#
# Solution: create a fresh AsyncEngine + dispose it for every test.
# This is ~10–20ms of overhead per test — acceptable for integration tests.
#
# We also skip nested savepoints (begin_nested) because asyncpg's internal
# state machine raises InFailedSQLTransactionError after a RELEASE SAVEPOINT
# when an inner statement has already failed.  Instead, each test uses unique
# chunk_id values so rows from different tests never conflict.
# ─────────────────────────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def db_session() -> AsyncSession:
    """Provide a per-test AsyncSession backed by a fresh engine."""
    engine = create_async_engine(
        _db_url,
        echo=False,
        pool_size=2,
        max_overflow=0,
    )
    factory = get_session_factory(engine)

    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

    await engine.dispose()
