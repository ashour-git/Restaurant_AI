"""
Customers API routes.

Provides CRUD operations for customer management.
"""

from app.core.database import get_db
from app.models import Customer, LoyaltyTier, Employee
from app.schemas import CustomerCreate, CustomerResponse, CustomerUpdate
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.api.dependencies import get_current_active_user

router = APIRouter(prefix="/customers", tags=["Customers"])


@router.get("", response_model=list[CustomerResponse])
async def get_customers(
    search: str | None = None,
    loyalty_tier: str | None = None,
    is_active: bool | None = None,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    current_user: Employee = Depends(get_current_active_user),
) -> list[CustomerResponse]:
    """
    Get customers with filtering and pagination.

    Args:
        search: Search in name, email, or phone.
        loyalty_tier: Filter by loyalty tier.
        is_active: Filter by active status.
        skip: Number of items to skip.
        limit: Maximum number of items to return.
        db: Database session.

    Returns:
        List of customers.
    """
    query = select(Customer).order_by(Customer.last_name, Customer.first_name)

    if search:
        search_filter = f"%{search}%"
        query = query.where(
            (Customer.first_name.ilike(search_filter))
            | (Customer.last_name.ilike(search_filter))
            | (Customer.email.ilike(search_filter))
            | (Customer.phone.ilike(search_filter))
        )

    if loyalty_tier:
        query = query.where(Customer.loyalty_tier == LoyaltyTier(loyalty_tier))

    if is_active is not None:
        query = query.where(Customer.is_active == is_active)

    query = query.offset(skip).limit(limit)

    result = await db.execute(query)
    return result.scalars().all()


@router.post("", response_model=CustomerResponse, status_code=status.HTTP_201_CREATED)
async def create_customer(
    customer: CustomerCreate,
    db: AsyncSession = Depends(get_db),
    current_user: Employee = Depends(get_current_active_user),
) -> CustomerResponse:
    """
    Create a new customer.

    Args:
        customer: Customer data.
        db: Database session.

    Returns:
        Created customer.
    """
    # Check if email already exists
    if customer.email:
        result = await db.execute(select(Customer).where(Customer.email == customer.email))
        if result.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered",
            )

    db_customer = Customer(**customer.model_dump())
    db.add(db_customer)
    await db.flush()
    await db.refresh(db_customer)
    return db_customer


@router.get("/{customer_id}", response_model=CustomerResponse)
async def get_customer(
    customer_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: Employee = Depends(get_current_active_user),
) -> CustomerResponse:
    """
    Get a customer by ID.

    Args:
        customer_id: Customer ID.
        db: Database session.

    Returns:
        Customer.
    """
    result = await db.execute(select(Customer).where(Customer.id == customer_id))
    customer = result.scalar_one_or_none()

    if not customer:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Customer with id {customer_id} not found",
        )

    return customer


@router.put("/{customer_id}", response_model=CustomerResponse)
async def update_customer(
    customer_id: int,
    customer_update: CustomerUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: Employee = Depends(get_current_active_user),
) -> CustomerResponse:
    """
    Update a customer.

    Args:
        customer_id: Customer ID.
        customer_update: Update data.
        db: Database session.

    Returns:
        Updated customer.
    """
    result = await db.execute(select(Customer).where(Customer.id == customer_id))
    customer = result.scalar_one_or_none()

    if not customer:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Customer with id {customer_id} not found",
        )

    update_data = customer_update.model_dump(exclude_unset=True)

    # Update loyalty tier based on points
    if "loyalty_points" in update_data:
        points = update_data["loyalty_points"]
        if points >= 2000:
            customer.loyalty_tier = LoyaltyTier.PLATINUM
        elif points >= 1000:
            customer.loyalty_tier = LoyaltyTier.GOLD
        elif points >= 500:
            customer.loyalty_tier = LoyaltyTier.SILVER
        else:
            customer.loyalty_tier = LoyaltyTier.BRONZE

    for field, value in update_data.items():
        setattr(customer, field, value)

    await db.flush()
    await db.refresh(customer)
    return customer


@router.delete("/{customer_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_customer(
    customer_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: Employee = Depends(get_current_active_user),
) -> None:
    """
    Soft delete a customer (set inactive).

    Args:
        customer_id: Customer ID.
        db: Database session.
    """
    result = await db.execute(select(Customer).where(Customer.id == customer_id))
    customer = result.scalar_one_or_none()

    if not customer:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Customer with id {customer_id} not found",
        )

    customer.is_active = False
    await db.flush()


@router.post("/{customer_id}/points", response_model=CustomerResponse)
async def add_loyalty_points(
    customer_id: int,
    points: int = Query(..., ge=1),
    db: AsyncSession = Depends(get_db),
    current_user: Employee = Depends(get_current_active_user),
) -> CustomerResponse:
    """
    Add loyalty points to a customer.

    Args:
        customer_id: Customer ID.
        points: Points to add.
        db: Database session.

    Returns:
        Updated customer.
    """
    result = await db.execute(select(Customer).where(Customer.id == customer_id))
    customer = result.scalar_one_or_none()

    if not customer:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Customer with id {customer_id} not found",
        )

    customer.loyalty_points += points

    # Update tier based on new points
    if customer.loyalty_points >= 2000:
        customer.loyalty_tier = LoyaltyTier.PLATINUM
    elif customer.loyalty_points >= 1000:
        customer.loyalty_tier = LoyaltyTier.GOLD
    elif customer.loyalty_points >= 500:
        customer.loyalty_tier = LoyaltyTier.SILVER

    await db.flush()
    await db.refresh(customer)
    return customer
