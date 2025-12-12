"""
API dependencies for authentication and authorization.
"""

from typing import Annotated

from app.core.database import get_db
from app.core.security import decode_access_token
from app.models import Employee
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")
oauth2_scheme_optional = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login", auto_error=False)


async def get_current_user(
    token: str = Depends(oauth2_scheme), db: AsyncSession = Depends(get_db)
) -> Employee:
    """
    Get the current user from the authentication token.

    Args:
        token: The OAuth2 token.
        db: The database session.

    Returns:
        Employee: The current user.

    Raises:
        HTTPException: If the token is invalid or the user is not found.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    payload = decode_access_token(token)
    if payload is None:
        raise credentials_exception
    email: str | None = payload.get("sub")
    if email is None:
        raise credentials_exception

    result = await db.execute(select(Employee).where(Employee.email == email))
    user = result.scalar_one_or_none()

    if user is None:
        raise credentials_exception
    return user


async def get_current_user_optional(
    token: str | None = Depends(oauth2_scheme_optional), db: AsyncSession = Depends(get_db)
) -> Employee | None:
    """
    Get the current user optionally (for public routes that benefit from auth).
    """
    if token is None:
        return None

    payload = decode_access_token(token)
    if payload is None:
        return None

    email: str | None = payload.get("sub")
    if email is None:
        return None

    result = await db.execute(select(Employee).where(Employee.email == email))
    return result.scalar_one_or_none()


async def get_current_active_user(
    current_user: Employee = Depends(get_current_user),
) -> Employee:
    """
    Get the current active user.

    Args:
        current_user: The current user.

    Returns:
        Employee: The current active user.

    Raises:
        HTTPException: If the user is inactive.
    """
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


def require_role(allowed_roles: list[str]):
    """
    Dependency factory for role-based access control.

    Args:
        allowed_roles: List of roles that can access the endpoint.

    Returns:
        A dependency that validates user role.
    """

    async def role_checker(
        current_user: Employee = Depends(get_current_active_user),
    ) -> Employee:
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required roles: {allowed_roles}",
            )
        return current_user

    return role_checker


# Role-based dependencies
RequireAdmin = Annotated[Employee, Depends(require_role(["admin"]))]
RequireManager = Annotated[Employee, Depends(require_role(["admin", "manager"]))]
RequireStaff = Annotated[Employee, Depends(require_role(["admin", "manager", "staff"]))]
