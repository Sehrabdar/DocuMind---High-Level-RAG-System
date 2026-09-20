"""
Async SQLAlchemy engine and session factory for DocuMind Phase 3.

Usage
-----
    from db.session import get_engine, get_session_factory, get_async_session

    engine = get_engine()
    async_session = get_session_factory(engine)
    async with get_async_session(async_session) as session:
        # work with session
        ...

Design decisions
----------------
- One ``AsyncEngine`` per process is the SQLAlchemy recommendation.
  The module-level ``get_engine()`` creates a singleton per URL.
- ``AsyncSession`` is not thread-safe; use one per request/coroutine.
- ``expire_on_commit=False`` is used because after ``session.commit()``,
  accessing attributes on ORM objects (e.g. for returning the inserted row)
  would trigger a new lazy load that cannot be awaited in plain Python.
  This is the standard recommendation for async SQLAlchemy.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

logger = logging.getLogger(__name__)

# Module-level cache: database_url → AsyncEngine
# avoids creating a new engine (and connection pool) per call
_engines: dict[str, AsyncEngine] = {}


def get_engine(database_url: str | None = None) -> AsyncEngine:
    """Return (or create) an async SQLAlchemy engine for *database_url*.

    The engine is cached by URL, so repeated calls with the same URL
    return the same engine instance.

    Parameters
    ----------
    database_url:
        SQLAlchemy async URL (e.g. ``postgresql+asyncpg://...``).
        Defaults to the project configuration if not supplied.
    """
    if database_url is None:
        from config import settings
        database_url = settings.database_url

    if database_url not in _engines:
        logger.debug("Creating async engine for: %s", _redact_url(database_url))
        _engines[database_url] = create_async_engine(
            database_url,
            echo=False,          # set to True for SQL debug logging
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,  # verify connections before use
        )
    return _engines[database_url]


def get_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Return an ``async_sessionmaker`` bound to *engine*.

    The returned factory creates ``AsyncSession`` instances with
    ``expire_on_commit=False`` so that ORM objects remain accessible
    after a commit without triggering an implicit lazy load.
    """
    return async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )


@asynccontextmanager
async def get_async_session(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncGenerator[AsyncSession, None]:
    """Context manager providing a single ``AsyncSession``.

    Commits on clean exit, rolls back on any exception, and always closes.

    Usage::

        async with get_async_session(session_factory) as session:
            await repository.do_something(session)
    """
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


def _redact_url(url: str) -> str:
    """Return the URL with the password replaced by '***'."""
    try:
        from urllib.parse import urlparse, urlunparse

        parsed = urlparse(url)
        if parsed.password:
            netloc = parsed.netloc.replace(parsed.password, "***")
            return urlunparse(parsed._replace(netloc=netloc))
        return url
    except Exception:
        return "<unparseable-url>"
