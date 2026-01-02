import asyncio
import os

from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")


async def check_db():
    if not DATABASE_URL:
        print("[ERROR] DATABASE_URL not found in environment variables")
        return

    print(
        f"Checking database connection to: {DATABASE_URL.split('@')[-1] if '@' in DATABASE_URL else DATABASE_URL}"
    )

    try:
        engine = create_async_engine(DATABASE_URL)
        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT 1"))
            print("[OK] Database connection successful!")

            # List tables
            # This query works for PostgreSQL
            if "postgresql" in DATABASE_URL:
                result = await conn.execute(
                    text(
                        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
                    )
                )
                tables = result.fetchall()
                print("\nTables in database:")
                for table in tables:
                    print(f"  - {table[0]}")
            else:
                # Fallback for SQLite
                result = await conn.execute(
                    text("SELECT name FROM sqlite_master WHERE type='table'")
                )
                tables = result.fetchall()
                print("\nTables in database:")
                for table in tables:
                    print(f"  - {table[0]}")

        await engine.dispose()

    except Exception as e:
        print(f"[ERROR] Database connection failed: {e}")


if __name__ == "__main__":
    asyncio.run(check_db())
