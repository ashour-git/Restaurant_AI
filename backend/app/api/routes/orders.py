"""
Orders API routes.

Provides CRUD operations for orders and order management.
"""

from datetime import datetime, timezone
from decimal import Decimal

from app.core.database import get_db
from app.models import (
    Customer,
    MenuItem,
    Order,
    OrderItem,
    OrderStatus,
    OrderType,
    Payment,
    PaymentMethod,
    PaymentStatus,
    Employee,
)
from app.schemas import (
    OrderCreate,
    OrderResponse,
    OrderSummary,
    OrderUpdate,
    PaymentCreate,
    PaymentResponse,
)
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from app.api.dependencies import get_current_active_user

router = APIRouter(prefix="/orders", tags=["Orders"])

# Tax rate (8.5%)
TAX_RATE = Decimal("0.085")


@router.get("", response_model=list[OrderSummary])
async def get_orders(
    status_filter: str | None = Query(None, alias="status"),
    order_type: str | None = None,
    customer_id: int | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    current_user: Employee = Depends(get_current_active_user),
) -> list[OrderSummary]:
    """
    Get orders with filtering and pagination.

    Args:
        status_filter: Filter by order status.
        order_type: Filter by order type.
        customer_id: Filter by customer.
        date_from: Filter orders from this date.
        date_to: Filter orders until this date.
        skip: Number of items to skip.
        limit: Maximum number of items to return.
        db: Database session.

    Returns:
        List of order summaries.
    """
    query = select(Order).order_by(Order.created_at.desc())

    if status_filter:
        query = query.where(Order.status == status_filter)

    if order_type:
        query = query.where(Order.order_type == order_type)

    if customer_id:
        query = query.where(Order.customer_id == customer_id)

    if date_from:
        query = query.where(Order.created_at >= datetime.fromisoformat(date_from))

    if date_to:
        query = query.where(Order.created_at <= datetime.fromisoformat(date_to))

    query = query.offset(skip).limit(limit)

    result = await db.execute(query)
    return result.scalars().all()


@router.post("", response_model=OrderResponse, status_code=status.HTTP_201_CREATED)
async def create_order(
    order_data: OrderCreate,
    db: AsyncSession = Depends(get_db),
    current_user: Employee = Depends(get_current_active_user),
) -> OrderResponse:
    """
    Create a new order.

    Args:
        order_data: Order data including items.
        db: Database session.

    Returns:
        Created order with items.
    """
    # Validate customer if provided
    if order_data.customer_id:
        customer_result = await db.execute(
            select(Customer).where(Customer.id == order_data.customer_id)
        )
        if not customer_result.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Customer with id {order_data.customer_id} not found",
            )

    # Validate and get menu items
    item_ids = [item.menu_item_id for item in order_data.items]
    items_result = await db.execute(select(MenuItem).where(MenuItem.id.in_(item_ids)))
    menu_items = {item.id: item for item in items_result.scalars().all()}

    if len(menu_items) != len(item_ids):
        missing_ids = set(item_ids) - set(menu_items.keys())
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Menu items not found: {missing_ids}",
        )

    # Create order
    order = Order(
        customer_id=order_data.customer_id,
        table_number=order_data.table_number,
        order_type=OrderType(order_data.order_type),
        notes=order_data.notes,
        status=OrderStatus.PENDING,
    )
    db.add(order)
    await db.flush()

    # Create order items and calculate totals
    subtotal = Decimal("0.00")

    for item_data in order_data.items:
        menu_item = menu_items[item_data.menu_item_id]
        unit_price = menu_item.price
        total_price = unit_price * item_data.quantity
        subtotal += total_price

        order_item = OrderItem(
            order_id=order.id,
            menu_item_id=item_data.menu_item_id,
            quantity=item_data.quantity,
            unit_price=unit_price,
            total_price=total_price,
            special_instructions=item_data.special_instructions,
            modifiers=item_data.modifiers,
        )
        db.add(order_item)

    # Calculate order totals
    order.subtotal = subtotal
    order.tax_amount = (subtotal * TAX_RATE).quantize(Decimal("0.01"))
    order.total = order.subtotal + order.tax_amount

    await db.flush()

    # Reload with relationships
    result = await db.execute(
        select(Order)
        .options(selectinload(Order.items).selectinload(OrderItem.menu_item))
        .options(selectinload(Order.customer))
        .where(Order.id == order.id)
    )

    return result.scalar_one()


@router.get("/{order_id}", response_model=OrderResponse)
async def get_order(
    order_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: Employee = Depends(get_current_active_user),
) -> OrderResponse:
    """
    Get an order by ID.

    Args:
        order_id: Order ID.
        db: Database session.

    Returns:
        Order with items.
    """
    result = await db.execute(
        select(Order)
        .options(selectinload(Order.items).selectinload(OrderItem.menu_item))
        .options(selectinload(Order.customer))
        .where(Order.id == order_id)
    )
    order = result.scalar_one_or_none()

    if not order:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Order with id {order_id} not found",
        )

    return order


@router.put("/{order_id}", response_model=OrderResponse)
async def update_order(
    order_id: int,
    order_update: OrderUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: Employee = Depends(get_current_active_user),
) -> OrderResponse:
    """
    Update an order.

    Args:
        order_id: Order ID.
        order_update: Update data.
        db: Database session.

    Returns:
        Updated order.
    """
    result = await db.execute(select(Order).where(Order.id == order_id))
    order = result.scalar_one_or_none()

    if not order:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Order with id {order_id} not found",
        )

    update_data = order_update.model_dump(exclude_unset=True)

    # Handle status change
    if "status" in update_data:
        new_status = OrderStatus(update_data["status"])
        update_data["status"] = new_status

        if new_status == OrderStatus.COMPLETED:
            update_data["completed_at"] = datetime.now(timezone.utc)

    # Recalculate total if discount or tip changed
    if "discount_amount" in update_data or "tip_amount" in update_data:
        discount = update_data.get("discount_amount", order.discount_amount)
        tip = update_data.get("tip_amount", order.tip_amount)
        update_data["total"] = order.subtotal + order.tax_amount - discount + tip

    for field, value in update_data.items():
        setattr(order, field, value)

    await db.flush()

    # Reload with relationships
    result = await db.execute(
        select(Order)
        .options(selectinload(Order.items).selectinload(OrderItem.menu_item))
        .options(selectinload(Order.customer))
        .where(Order.id == order_id)
    )

    return result.scalar_one()


@router.post("/{order_id}/cancel", response_model=OrderResponse)
async def cancel_order(
    order_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: Employee = Depends(get_current_active_user),
) -> OrderResponse:
    """
    Cancel an order.

    Args:
        order_id: Order ID.
        db: Database session.

    Returns:
        Cancelled order.
    """
    result = await db.execute(select(Order).where(Order.id == order_id))
    order = result.scalar_one_or_none()

    if not order:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Order with id {order_id} not found",
        )

    if order.status in [OrderStatus.COMPLETED, OrderStatus.CANCELLED]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot cancel order with status {order.status.value}",
        )

    order.status = OrderStatus.CANCELLED
    await db.flush()

    # Reload with relationships
    result = await db.execute(
        select(Order).options(selectinload(Order.items)).where(Order.id == order_id)
    )

    return result.scalar_one()


# ============================================================================
# PAYMENTS
# ============================================================================


@router.post(
    "/{order_id}/payment", response_model=PaymentResponse, status_code=status.HTTP_201_CREATED
)
async def create_payment(
    order_id: int,
    payment_data: PaymentCreate,
    db: AsyncSession = Depends(get_db),
    current_user: Employee = Depends(get_current_active_user),
) -> PaymentResponse:
    """
    Process a payment for an order.

    Args:
        order_id: Order ID.
        payment_data: Payment data.
        db: Database session.

    Returns:
        Created payment.
    """
    # Get order
    result = await db.execute(select(Order).where(Order.id == order_id))
    order = result.scalar_one_or_none()

    if not order:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Order with id {order_id} not found",
        )

    # Check if payment already exists
    payment_result = await db.execute(select(Payment).where(Payment.order_id == order_id))
    if payment_result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Payment already exists for this order",
        )

    # Create payment
    payment = Payment(
        order_id=order_id,
        payment_method=PaymentMethod(payment_data.payment_method),
        amount=payment_data.amount,
        tip_amount=payment_data.tip_amount,
        status=PaymentStatus.COMPLETED,  # Simplified - assume payment succeeds
        completed_at=datetime.now(timezone.utc),
    )
    db.add(payment)

    # Update order with tip and complete it
    order.tip_amount = payment_data.tip_amount
    order.total = (
        order.subtotal + order.tax_amount - order.discount_amount + payment_data.tip_amount
    )
    order.status = OrderStatus.COMPLETED
    order.completed_at = datetime.now(timezone.utc)

    await db.flush()
    await db.refresh(payment)

    return payment


@router.get("/{order_id}/payment", response_model=PaymentResponse)
async def get_payment(
    order_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: Employee = Depends(get_current_active_user),
) -> PaymentResponse:
    """
    Get payment for an order.

    Args:
        order_id: Order ID.
        db: Database session.

    Returns:
        Payment details.
    """
    result = await db.execute(select(Payment).where(Payment.order_id == order_id))
    payment = result.scalar_one_or_none()

    if not payment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Payment not found for order {order_id}",
        )

    return payment
