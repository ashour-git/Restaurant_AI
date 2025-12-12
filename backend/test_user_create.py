import asyncio
import sys

sys.path.insert(0, ".")


async def test_register():
    from app.core.security import get_password_hash
    from app.models import Employee
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
    from sqlalchemy.orm import sessionmaker

    # Create engine
    engine = create_async_engine("sqlite+aiosqlite:///restaurant.db")
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        try:
            # Create new user
            new_user = Employee(
                email="test2@restaurant.com",
                password_hash=get_password_hash("Test1234!"),
                first_name="Test2",
                last_name="User2",
                role="staff",
                is_active=True,
            )
            session.add(new_user)
            await session.commit()
            await session.refresh(new_user)
            print(f"User created: {new_user.id}, {new_user.email}, {new_user.employee_id}")
        except Exception as e:
            print(f"Error: {type(e).__name__}: {e}")
            import traceback

            traceback.print_exc()


asyncio.run(test_register())
