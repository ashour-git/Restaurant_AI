"""
Menu API routes.

Provides CRUD operations for categories, subcategories, and menu items.
"""

from app.core.database import get_db
from app.models import Category, MenuItem, Subcategory
from app.schemas import (
    CategoryCreate,
    CategoryResponse,
    CategoryUpdate,
    MenuItemCreate,
    MenuItemResponse,
    MenuItemUpdate,
    SubcategoryCreate,
    SubcategoryResponse,
)
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

router = APIRouter(prefix="/menu", tags=["Menu"])


# ============================================================================
# CATEGORIES
# ============================================================================


@router.get("/categories", response_model=list[CategoryResponse])
async def get_categories(
    is_active: bool | None = None,
    db: AsyncSession = Depends(get_db),
) -> list[CategoryResponse]:
    """
    Get all menu categories.

    Args:
        is_active: Filter by active status.
        db: Database session.

    Returns:
        List of categories.
    """
    query = select(Category).order_by(Category.display_order, Category.name)

    if is_active is not None:
        query = query.where(Category.is_active == is_active)

    result = await db.execute(query)
    return result.scalars().all()


@router.post("/categories", response_model=CategoryResponse, status_code=status.HTTP_201_CREATED)
async def create_category(
    category: CategoryCreate,
    db: AsyncSession = Depends(get_db),
) -> CategoryResponse:
    """
    Create a new category.

    Args:
        category: Category data.
        db: Database session.

    Returns:
        Created category.
    """
    db_category = Category(**category.model_dump())
    db.add(db_category)
    await db.flush()
    await db.refresh(db_category)
    return db_category


@router.get("/categories/{category_id}", response_model=CategoryResponse)
async def get_category(
    category_id: int,
    db: AsyncSession = Depends(get_db),
) -> CategoryResponse:
    """
    Get a category by ID.

    Args:
        category_id: Category ID.
        db: Database session.

    Returns:
        Category.

    Raises:
        HTTPException: If category not found.
    """
    result = await db.execute(select(Category).where(Category.id == category_id))
    category = result.scalar_one_or_none()

    if not category:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Category with id {category_id} not found",
        )

    return category


@router.put("/categories/{category_id}", response_model=CategoryResponse)
async def update_category(
    category_id: int,
    category_update: CategoryUpdate,
    db: AsyncSession = Depends(get_db),
) -> CategoryResponse:
    """
    Update a category.

    Args:
        category_id: Category ID.
        category_update: Update data.
        db: Database session.

    Returns:
        Updated category.
    """
    result = await db.execute(select(Category).where(Category.id == category_id))
    category = result.scalar_one_or_none()

    if not category:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Category with id {category_id} not found",
        )

    update_data = category_update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(category, field, value)

    await db.flush()
    await db.refresh(category)
    return category


@router.delete("/categories/{category_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_category(
    category_id: int,
    db: AsyncSession = Depends(get_db),
) -> None:
    """
    Delete a category.

    Args:
        category_id: Category ID.
        db: Database session.
    """
    result = await db.execute(select(Category).where(Category.id == category_id))
    category = result.scalar_one_or_none()

    if not category:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Category with id {category_id} not found",
        )

    await db.delete(category)


# ============================================================================
# SUBCATEGORIES
# ============================================================================


@router.get("/subcategories", response_model=list[SubcategoryResponse])
async def get_subcategories(
    category_id: int | None = None,
    is_active: bool | None = None,
    db: AsyncSession = Depends(get_db),
) -> list[SubcategoryResponse]:
    """
    Get all subcategories.

    Args:
        category_id: Filter by category.
        is_active: Filter by active status.
        db: Database session.

    Returns:
        List of subcategories.
    """
    query = select(Subcategory).order_by(Subcategory.display_order, Subcategory.name)

    if category_id is not None:
        query = query.where(Subcategory.category_id == category_id)

    if is_active is not None:
        query = query.where(Subcategory.is_active == is_active)

    result = await db.execute(query)
    return result.scalars().all()


@router.post(
    "/subcategories", response_model=SubcategoryResponse, status_code=status.HTTP_201_CREATED
)
async def create_subcategory(
    subcategory: SubcategoryCreate,
    db: AsyncSession = Depends(get_db),
) -> SubcategoryResponse:
    """
    Create a new subcategory.

    Args:
        subcategory: Subcategory data.
        db: Database session.

    Returns:
        Created subcategory.
    """
    # Verify category exists
    result = await db.execute(select(Category).where(Category.id == subcategory.category_id))
    if not result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Category with id {subcategory.category_id} not found",
        )

    db_subcategory = Subcategory(**subcategory.model_dump())
    db.add(db_subcategory)
    await db.flush()
    await db.refresh(db_subcategory)
    return db_subcategory


# ============================================================================
# MENU ITEMS
# ============================================================================


@router.get("/items", response_model=list[MenuItemResponse])
async def get_menu_items(
    subcategory_id: int | None = None,
    category_id: int | None = None,
    is_active: bool | None = None,
    is_vegetarian: bool | None = None,
    is_vegan: bool | None = None,
    is_gluten_free: bool | None = None,
    search: str | None = Query(None, min_length=1),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
) -> list[MenuItemResponse]:
    """
    Get menu items with filtering and pagination.

    Args:
        subcategory_id: Filter by subcategory.
        category_id: Filter by category.
        is_active: Filter by active status.
        is_vegetarian: Filter by vegetarian.
        is_vegan: Filter by vegan.
        is_gluten_free: Filter by gluten-free.
        search: Search in name and description.
        skip: Number of items to skip.
        limit: Maximum number of items to return.
        db: Database session.

    Returns:
        List of menu items.
    """
    query = select(MenuItem).options(selectinload(MenuItem.subcategory))

    if subcategory_id is not None:
        query = query.where(MenuItem.subcategory_id == subcategory_id)

    if category_id is not None:
        query = query.join(Subcategory).where(Subcategory.category_id == category_id)

    if is_active is not None:
        query = query.where(MenuItem.is_active == is_active)

    if is_vegetarian is not None:
        query = query.where(MenuItem.is_vegetarian == is_vegetarian)

    if is_vegan is not None:
        query = query.where(MenuItem.is_vegan == is_vegan)

    if is_gluten_free is not None:
        query = query.where(MenuItem.is_gluten_free == is_gluten_free)

    if search:
        search_filter = f"%{search}%"
        query = query.where(
            (MenuItem.name.ilike(search_filter)) | (MenuItem.description.ilike(search_filter))
        )

    query = query.order_by(MenuItem.name).offset(skip).limit(limit)

    result = await db.execute(query)
    return result.scalars().all()


@router.post("/items", response_model=MenuItemResponse, status_code=status.HTTP_201_CREATED)
async def create_menu_item(
    item: MenuItemCreate,
    db: AsyncSession = Depends(get_db),
) -> MenuItemResponse:
    """
    Create a new menu item.

    Args:
        item: Menu item data.
        db: Database session.

    Returns:
        Created menu item.
    """
    # Verify subcategory exists
    result = await db.execute(select(Subcategory).where(Subcategory.id == item.subcategory_id))
    if not result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Subcategory with id {item.subcategory_id} not found",
        )

    # Generate SKU if not provided
    item_data = item.model_dump()
    if not item_data.get("sku"):
        count_result = await db.execute(select(func.count(MenuItem.id)))
        count = count_result.scalar() or 0
        item_data["sku"] = f"ITEM_{count + 1:04d}"

    db_item = MenuItem(**item_data)
    db.add(db_item)
    await db.flush()
    await db.refresh(db_item)
    return db_item


@router.get("/items/{item_id}", response_model=MenuItemResponse)
async def get_menu_item(
    item_id: int,
    db: AsyncSession = Depends(get_db),
) -> MenuItemResponse:
    """
    Get a menu item by ID.

    Args:
        item_id: Menu item ID.
        db: Database session.

    Returns:
        Menu item.
    """
    result = await db.execute(
        select(MenuItem).options(selectinload(MenuItem.subcategory)).where(MenuItem.id == item_id)
    )
    item = result.scalar_one_or_none()

    if not item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Menu item with id {item_id} not found",
        )

    return item


@router.put("/items/{item_id}", response_model=MenuItemResponse)
async def update_menu_item(
    item_id: int,
    item_update: MenuItemUpdate,
    db: AsyncSession = Depends(get_db),
) -> MenuItemResponse:
    """
    Update a menu item.

    Args:
        item_id: Menu item ID.
        item_update: Update data.
        db: Database session.

    Returns:
        Updated menu item.
    """
    result = await db.execute(select(MenuItem).where(MenuItem.id == item_id))
    item = result.scalar_one_or_none()

    if not item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Menu item with id {item_id} not found",
        )

    update_data = item_update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(item, field, value)

    await db.flush()
    await db.refresh(item)
    return item


@router.delete("/items/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_menu_item(
    item_id: int,
    db: AsyncSession = Depends(get_db),
) -> None:
    """
    Delete a menu item.

    Args:
        item_id: Menu item ID.
        db: Database session.
    """
    result = await db.execute(select(MenuItem).where(MenuItem.id == item_id))
    item = result.scalar_one_or_none()

    if not item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Menu item with id {item_id} not found",
        )

    await db.delete(item)
