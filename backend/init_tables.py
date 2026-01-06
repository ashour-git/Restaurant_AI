#!/usr/bin/env python3
"""Initialize database tables."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.database import Base, get_async_database_url
from app.models.models import (  # noqa: F401
    Category,
    Customer,
    Employee,
    InventoryItem,
    MenuItem,
    Order,
    OrderItem,
    Subcategory,
)
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine


async def init_db():
    """Create all database tables."""
    database_url = get_async_database_url()
    engine = create_async_engine(database_url, echo=False)
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        print("Database tables created successfully!")
    
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(init_db())
