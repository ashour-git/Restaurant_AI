"""
Database configuration and session management.

Provides SQLAlchemy async engine, session factory, and base model class.
Supports both PostgreSQL (production) and SQLite (development).
"""

from collections.abc import AsyncGenerator

from sqlalchemy import MetaData
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.core.config import settings

# Naming convention for constraints (important for Alembic migrations)
convention = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

metadata = MetaData(naming_convention=convention)


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""

    metadata = metadata


# Convert sync URL to async URL
def get_async_database_url() -> str:
    """
    Convert synchronous database URL to async format.

    Supports:
    - PostgreSQL: postgresql:// -> postgresql+asyncpg://
    - SQLite: sqlite:// -> sqlite+aiosqlite://
    """
    url = settings.DATABASE_URL

    # Handle SQLite
    if url.startswith("sqlite://"):
        return url.replace("sqlite://", "sqlite+aiosqlite://", 1)
    if url.startswith("sqlite+aiosqlite://"):
        return url

    # Handle PostgreSQL
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    if url.startswith("postgresql+asyncpg://"):
        return url

    return url


def is_sqlite() -> bool:
    """Check if using SQLite database."""
    return "sqlite" in settings.DATABASE_URL.lower()


# Create async engine with appropriate settings
engine_kwargs = {
    "echo": settings.DEBUG,
}

# SQLite doesn't support pool_size/max_overflow
if not is_sqlite():
    engine_kwargs.update(
        {
            "pool_size": settings.DATABASE_POOL_SIZE,
            "max_overflow": settings.DATABASE_MAX_OVERFLOW,
        }
    )
else:
    # For SQLite, enable check_same_thread=False for async
    engine_kwargs["connect_args"] = {"check_same_thread": False}

engine = create_async_engine(get_async_database_url(), **engine_kwargs)

# Session factory
async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency that provides a database session.

    Yields:
        AsyncSession: Database session.

    Example:
        >>> @router.get("/items")
        >>> async def get_items(db: AsyncSession = Depends(get_db)):
        >>>     result = await db.execute(select(Item))
        >>>     return result.scalars().all()
    """
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
