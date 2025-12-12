"""
Inventory API routes.

Provides CRUD operations for inventory management.
"""

from datetime import datetime, timezone

from app.core.database import get_db
from app.models import InventoryItem, Supplier, Employee
from app.schemas import InventoryItemCreate, InventoryItemResponse, InventoryItemUpdate
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from app.api.dependencies import get_current_active_user

router = APIRouter(prefix="/inventory", tags=["Inventory"])


@router.get("", response_model=list[InventoryItemResponse])
async def get_inventory_items(
    category: str | None = None,
    supplier_id: int | None = None,
    low_stock: bool | None = None,
    is_active: bool | None = None,
    search: str | None = None,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    current_user: Employee = Depends(get_current_active_user),
) -> list[InventoryItemResponse]:
    """
    Get inventory items with filtering and pagination.

    Args:
        category: Filter by category.
        supplier_id: Filter by supplier.
        low_stock: Filter items below reorder level.
        is_active: Filter by active status.
        search: Search in name.
        skip: Number of items to skip.
        limit: Maximum number of items to return.
        db: Database session.

    Returns:
        List of inventory items.
    """
    query = select(InventoryItem).order_by(InventoryItem.name)

    if category:
        query = query.where(InventoryItem.category == category)

    if supplier_id:
        query = query.where(InventoryItem.supplier_id == supplier_id)

    if low_stock:
        query = query.where(InventoryItem.quantity_on_hand <= InventoryItem.reorder_level)

    if is_active is not None:
        query = query.where(InventoryItem.is_active == is_active)

    if search:
        query = query.where(InventoryItem.name.ilike(f"%{search}%"))

    query = query.offset(skip).limit(limit)

    result = await db.execute(query)
    items = result.scalars().all()

    # Add computed field
    response_items = []
    for item in items:
        item_dict = {
            "id": item.id,
            "sku": item.sku,
            "name": item.name,
            "category": item.category,
            "quantity_on_hand": item.quantity_on_hand,
            "unit": item.unit,
            "unit_cost": item.unit_cost,
            "reorder_level": item.reorder_level,
            "reorder_quantity": item.reorder_quantity,
            "supplier_id": item.supplier_id,
            "storage_location": item.storage_location,
            "expiry_date": item.expiry_date,
            "last_restocked": item.last_restocked,
            "is_active": item.is_active,
            "created_at": item.created_at,
            "updated_at": item.updated_at,
            "needs_reorder": item.quantity_on_hand <= item.reorder_level,
        }
        response_items.append(InventoryItemResponse(**item_dict))

    return response_items


@router.post("", response_model=InventoryItemResponse, status_code=status.HTTP_201_CREATED)
async def create_inventory_item(
    item: InventoryItemCreate,
    db: AsyncSession = Depends(get_db),
    current_user: Employee = Depends(get_current_active_user),
) -> InventoryItemResponse:
    """
    Create a new inventory item.

    Args:
        item: Inventory item data.
        db: Database session.

    Returns:
        Created inventory item.
    """
    # Verify supplier if provided
    if item.supplier_id:
        result = await db.execute(select(Supplier).where(Supplier.id == item.supplier_id))
        if not result.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Supplier with id {item.supplier_id} not found",
            )

    # Generate SKU if not provided
    item_data = item.model_dump()
    if not item_data.get("sku"):
        count_result = await db.execute(select(func.count(InventoryItem.id)))
        count = count_result.scalar() or 0
        item_data["sku"] = f"INV_{count + 1:04d}"

    db_item = InventoryItem(**item_data)
    db.add(db_item)
    await db.flush()
    await db.refresh(db_item)

    return InventoryItemResponse(
        **{k: getattr(db_item, k) for k in item_data.keys() if hasattr(db_item, k)},
        id=db_item.id,
        created_at=db_item.created_at,
        updated_at=db_item.updated_at,
        needs_reorder=db_item.needs_reorder,
    )


@router.get("/low-stock", response_model=list[InventoryItemResponse])
async def get_low_stock_items(
    db: AsyncSession = Depends(get_db),
    current_user: Employee = Depends(get_current_active_user),
) -> list[InventoryItemResponse]:
    """
    Get all items below reorder level.

    Args:
        db: Database session.

    Returns:
        List of low-stock items.
    """
    query = (
        select(InventoryItem)
        .where(
            InventoryItem.quantity_on_hand <= InventoryItem.reorder_level,
            InventoryItem.is_active == True,
        )
        .order_by(InventoryItem.category, InventoryItem.name)
    )

    result = await db.execute(query)
    items = result.scalars().all()

    return [
        InventoryItemResponse(
            id=item.id,
            sku=item.sku,
            name=item.name,
            category=item.category,
            quantity_on_hand=item.quantity_on_hand,
            unit=item.unit,
            unit_cost=item.unit_cost,
            reorder_level=item.reorder_level,
            reorder_quantity=item.reorder_quantity,
            supplier_id=item.supplier_id,
            storage_location=item.storage_location,
            expiry_date=item.expiry_date,
            last_restocked=item.last_restocked,
            is_active=item.is_active,
            created_at=item.created_at,
            updated_at=item.updated_at,
            needs_reorder=True,
        )
        for item in items
    ]


@router.get("/{item_id}", response_model=InventoryItemResponse)
async def get_inventory_item(
    item_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: Employee = Depends(get_current_active_user),
) -> InventoryItemResponse:
    """
    Get an inventory item by ID.

    Args:
        item_id: Inventory item ID.
        db: Database session.

    Returns:
        Inventory item.
    """
    result = await db.execute(select(InventoryItem).where(InventoryItem.id == item_id))
    item = result.scalar_one_or_none()

    if not item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Inventory item with id {item_id} not found",
        )

    return InventoryItemResponse(
        id=item.id,
        sku=item.sku,
        name=item.name,
        category=item.category,
        quantity_on_hand=item.quantity_on_hand,
        unit=item.unit,
        unit_cost=item.unit_cost,
        reorder_level=item.reorder_level,
        reorder_quantity=item.reorder_quantity,
        supplier_id=item.supplier_id,
        storage_location=item.storage_location,
        expiry_date=item.expiry_date,
        last_restocked=item.last_restocked,
        is_active=item.is_active,
        created_at=item.created_at,
        updated_at=item.updated_at,
        needs_reorder=item.needs_reorder,
    )


@router.put("/{item_id}", response_model=InventoryItemResponse)
async def update_inventory_item(
    item_id: int,
    item_update: InventoryItemUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: Employee = Depends(get_current_active_user),
) -> InventoryItemResponse:
    """
    Update an inventory item.

    Args:
        item_id: Inventory item ID.
        item_update: Update data.
        db: Database session.

    Returns:
        Updated inventory item.
    """
    result = await db.execute(select(InventoryItem).where(InventoryItem.id == item_id))
    item = result.scalar_one_or_none()

    if not item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Inventory item with id {item_id} not found",
        )

    update_data = item_update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(item, field, value)

    await db.flush()
    await db.refresh(item)

    return InventoryItemResponse(
        id=item.id,
        sku=item.sku,
        name=item.name,
        category=item.category,
        quantity_on_hand=item.quantity_on_hand,
        unit=item.unit,
        unit_cost=item.unit_cost,
        reorder_level=item.reorder_level,
        reorder_quantity=item.reorder_quantity,
        supplier_id=item.supplier_id,
        storage_location=item.storage_location,
        expiry_date=item.expiry_date,
        last_restocked=item.last_restocked,
        is_active=item.is_active,
        created_at=item.created_at,
        updated_at=item.updated_at,
        needs_reorder=item.needs_reorder,
    )


@router.post("/{item_id}/restock", response_model=InventoryItemResponse)
async def restock_item(
    item_id: int,
    quantity: float = Query(..., gt=0),
    db: AsyncSession = Depends(get_db),
    current_user: Employee = Depends(get_current_active_user),
) -> InventoryItemResponse:
    """
    Add stock to an inventory item.

    Args:
        item_id: Inventory item ID.
        quantity: Quantity to add.
        db: Database session.

    Returns:
        Updated inventory item.
    """
    result = await db.execute(select(InventoryItem).where(InventoryItem.id == item_id))
    item = result.scalar_one_or_none()

    if not item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Inventory item with id {item_id} not found",
        )

    item.quantity_on_hand += quantity
    item.last_restocked = datetime.now(timezone.utc)

    await db.flush()
    await db.refresh(item)

    return InventoryItemResponse(
        id=item.id,
        sku=item.sku,
        name=item.name,
        category=item.category,
        quantity_on_hand=item.quantity_on_hand,
        unit=item.unit,
        unit_cost=item.unit_cost,
        reorder_level=item.reorder_level,
        reorder_quantity=item.reorder_quantity,
        supplier_id=item.supplier_id,
        storage_location=item.storage_location,
        expiry_date=item.expiry_date,
        last_restocked=item.last_restocked,
        is_active=item.is_active,
        created_at=item.created_at,
        updated_at=item.updated_at,
        needs_reorder=item.needs_reorder,
    )
