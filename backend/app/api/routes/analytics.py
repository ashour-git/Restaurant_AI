"""
Analytics API routes.

Provides endpoints for sales analytics, reports, and dashboards.
"""

from datetime import datetime, timedelta
from decimal import Decimal

from app.core.database import get_db
from app.models import MenuItem, Order, OrderItem, OrderStatus, Employee
from app.schemas import DailySalesData, ItemSalesData, SalesSummary, TopSellingItem
from fastapi import APIRouter, Depends, Query
from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from app.api.dependencies import get_current_active_user

router = APIRouter(prefix="/analytics", tags=["Analytics"])


@router.get("/sales/summary", response_model=SalesSummary)
async def get_sales_summary(
    period: str = Query("today", regex="^(today|week|month|year)$"),
    db: AsyncSession = Depends(get_db),
    current_user: Employee = Depends(get_current_active_user),
) -> SalesSummary:
    """
    Get sales summary for a given period.

    Args:
        period: Time period (today, week, month, year).
        db: Database session.

    Returns:
        Sales summary with key metrics.
    """
    now = datetime.now()

    if period == "today":
        start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
    elif period == "week":
        start_date = now - timedelta(days=7)
    elif period == "month":
        start_date = now - timedelta(days=30)
    else:  # year
        start_date = now - timedelta(days=365)

    # Get completed orders in period
    orders_query = select(Order).where(
        Order.status == OrderStatus.COMPLETED,
        Order.completed_at >= start_date,
    )

    result = await db.execute(orders_query)
    orders = result.scalars().all()

    if not orders:
        return SalesSummary(
            period=period,
            total_revenue=Decimal("0.00"),
            total_orders=0,
            total_items_sold=0,
            avg_order_value=Decimal("0.00"),
            top_category="N/A",
            top_item="N/A",
        )

    total_revenue = sum(order.total for order in orders)
    total_orders = len(orders)
    avg_order_value = total_revenue / total_orders

    # Get total items sold
    order_ids = [order.id for order in orders]
    items_result = await db.execute(
        select(func.sum(OrderItem.quantity)).where(OrderItem.order_id.in_(order_ids))
    )
    total_items_sold = items_result.scalar() or 0

    # Get top category
    category_result = await db.execute(
        select(
            MenuItem.subcategory_id,
            func.sum(OrderItem.quantity).label("qty"),
        )
        .join(OrderItem, OrderItem.menu_item_id == MenuItem.id)
        .where(OrderItem.order_id.in_(order_ids))
        .group_by(MenuItem.subcategory_id)
        .order_by(desc("qty"))
        .limit(1)
    )
    top_category_row = category_result.first()
    top_category = f"Category {top_category_row[0]}" if top_category_row else "N/A"

    # Get top item
    item_result = await db.execute(
        select(
            MenuItem.name,
            func.sum(OrderItem.quantity).label("qty"),
        )
        .join(OrderItem, OrderItem.menu_item_id == MenuItem.id)
        .where(OrderItem.order_id.in_(order_ids))
        .group_by(MenuItem.id, MenuItem.name)
        .order_by(desc("qty"))
        .limit(1)
    )
    top_item_row = item_result.first()
    top_item = top_item_row[0] if top_item_row else "N/A"

    return SalesSummary(
        period=period,
        total_revenue=total_revenue,
        total_orders=total_orders,
        total_items_sold=total_items_sold,
        avg_order_value=avg_order_value.quantize(Decimal("0.01")),
        top_category=top_category,
        top_item=top_item,
    )


@router.get("/sales/daily", response_model=list[DailySalesData])
async def get_daily_sales(
    days: int = Query(30, ge=1, le=365),
    db: AsyncSession = Depends(get_db),
    current_user: Employee = Depends(get_current_active_user),
) -> list[DailySalesData]:
    """
    Get daily sales data for the specified number of days.

    Args:
        days: Number of days to retrieve.
        db: Database session.

    Returns:
        List of daily sales data.
    """
    start_date = datetime.now() - timedelta(days=days)

    result = await db.execute(
        select(
            func.date(Order.completed_at).label("date"),
            func.sum(Order.total).label("total_revenue"),
            func.count(Order.id).label("total_orders"),
        )
        .where(
            Order.status == OrderStatus.COMPLETED,
            Order.completed_at >= start_date,
        )
        .group_by(func.date(Order.completed_at))
        .order_by(func.date(Order.completed_at))
    )

    daily_data = []
    for row in result.all():
        # Get items sold for this day
        items_result = await db.execute(
            select(func.sum(OrderItem.quantity))
            .join(Order, Order.id == OrderItem.order_id)
            .where(
                Order.status == OrderStatus.COMPLETED,
                func.date(Order.completed_at) == row.date,
            )
        )
        total_items = items_result.scalar() or 0

        daily_data.append(
            DailySalesData(
                date=str(row.date),
                total_revenue=row.total_revenue or Decimal("0.00"),
                total_orders=row.total_orders,
                total_items_sold=total_items,
                avg_order_value=(
                    (row.total_revenue / row.total_orders).quantize(Decimal("0.01"))
                    if row.total_orders > 0
                    else Decimal("0.00")
                ),
            )
        )

    return daily_data


@router.get("/items/top-selling", response_model=list[TopSellingItem])
async def get_top_selling_items(
    days: int = Query(30, ge=1, le=365),
    limit: int = Query(10, ge=1, le=50),
    db: AsyncSession = Depends(get_db),
    current_user: Employee = Depends(get_current_active_user),
) -> list[TopSellingItem]:
    """
    Get top-selling menu items.

    Args:
        days: Number of days to analyze.
        limit: Maximum number of items to return.
        db: Database session.

    Returns:
        List of top-selling items.
    """
    start_date = datetime.now() - timedelta(days=days)

    result = await db.execute(
        select(
            MenuItem.id,
            MenuItem.name,
            func.sum(OrderItem.quantity).label("quantity_sold"),
            func.sum(OrderItem.total_price).label("revenue"),
        )
        .join(OrderItem, OrderItem.menu_item_id == MenuItem.id)
        .join(Order, Order.id == OrderItem.order_id)
        .where(
            Order.status == OrderStatus.COMPLETED,
            Order.completed_at >= start_date,
        )
        .group_by(MenuItem.id, MenuItem.name)
        .order_by(desc("quantity_sold"))
        .limit(limit)
    )

    top_items = []
    for rank, row in enumerate(result.all(), 1):
        top_items.append(
            TopSellingItem(
                rank=rank,
                item_id=row.id,
                item_name=row.name,
                quantity_sold=row.quantity_sold,
                revenue=row.revenue,
            )
        )

    return top_items


@router.get("/items/by-category", response_model=list[ItemSalesData])
async def get_sales_by_category(
    days: int = Query(30, ge=1, le=365),
    db: AsyncSession = Depends(get_db),
    current_user: Employee = Depends(get_current_active_user),
) -> list[ItemSalesData]:
    """
    Get sales data grouped by item category.

    Args:
        days: Number of days to analyze.
        db: Database session.

    Returns:
        List of item sales data.
    """
    start_date = datetime.now() - timedelta(days=days)

    result = await db.execute(
        select(
            MenuItem.id,
            MenuItem.name,
            MenuItem.subcategory_id,
            func.sum(OrderItem.quantity).label("quantity_sold"),
            func.sum(OrderItem.total_price).label("revenue"),
        )
        .join(OrderItem, OrderItem.menu_item_id == MenuItem.id)
        .join(Order, Order.id == OrderItem.order_id)
        .where(
            Order.status == OrderStatus.COMPLETED,
            Order.completed_at >= start_date,
        )
        .group_by(MenuItem.id, MenuItem.name, MenuItem.subcategory_id)
        .order_by(MenuItem.subcategory_id, desc("quantity_sold"))
    )

    sales_data = []
    for row in result.all():
        sales_data.append(
            ItemSalesData(
                item_id=row.id,
                item_name=row.name,
                category=f"Subcategory {row.subcategory_id}",
                quantity_sold=row.quantity_sold,
                revenue=row.revenue,
            )
        )

    return sales_data
