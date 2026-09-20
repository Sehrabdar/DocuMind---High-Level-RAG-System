"""
Alembic environment configuration for DocuMind Phase 3.

This env.py uses async SQLAlchemy (asyncpg) because the rest of the project
uses async SQLAlchemy.  Alembic 1.13+ supports async migrations via
``run_async_migrations()``.

The database URL is read from project configuration (``settings.database_url``)
so that migrations use the same credentials and connection string as the
application, without duplicating configuration in alembic.ini.
"""

from __future__ import annotations

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy.ext.asyncio import create_async_engine

# Import the project settings and SQLAlchemy metadata
from config import settings
from db.models import Base

# Alembic config object — provides access to .ini file values
config = context.config

# Configure Python logging from the .ini file
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Target metadata for autogenerate support
target_metadata = Base.metadata


def get_url() -> str:
    """Return the database URL from project settings."""
    return settings.database_url


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (no live DB connection).

    Generates SQL to stdout rather than executing it.  Useful for reviewing
    migration SQL before applying it.
    """
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection):  # type: ignore[no-untyped-def]
    context.configure(connection=connection, target_metadata=target_metadata)

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Run migrations against a live database using an async engine."""
    connectable = create_async_engine(get_url())

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode (connected to a live database)."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
