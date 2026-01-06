"""
Pydantic schemas for API request/response validation.

This module defines all data transfer objects (DTOs) used by the API.
"""

from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, EmailStr, Field

# ============================================================================
# BASE SCHEMAS
# ============================================================================


class BaseSchema(BaseModel):
    """Base schema with common configuration."""

    model_config = ConfigDict(
        from_attributes=True,
        str_strip_whitespace=True,
    )


class TimestampMixin(BaseModel):
    """Mixin for timestamp fields."""

    created_at: datetime
    updated_at: datetime | None = None


# ============================================================================
# CATEGORY SCHEMAS
# ============================================================================


class CategoryBase(BaseSchema):
    """Base category schema."""

    name: str = Field(..., min_length=1, max_length=100)
    description: str | None = None
    display_order: int = Field(default=0, ge=0)
    is_active: bool = True


class CategoryCreate(CategoryBase):
    """Schema for creating a category."""

    pass


class CategoryUpdate(BaseSchema):
    """Schema for updating a category."""

    name: str | None = Field(None, min_length=1, max_length=100)
    description: str | None = None
    display_order: int | None = Field(None, ge=0)
    is_active: bool | None = None


class CategoryResponse(CategoryBase, TimestampMixin):
    """Schema for category response."""

    id: int


# ============================================================================
# SUBCATEGORY SCHEMAS
# ============================================================================


class SubcategoryBase(BaseSchema):
    """Base subcategory schema."""

    name: str = Field(..., min_length=1, max_length=100)
    description: str | None = None
    display_order: int = Field(default=0, ge=0)
    is_active: bool = True


class SubcategoryCreate(SubcategoryBase):
    """Schema for creating a subcategory."""

    category_id: int


class SubcategoryUpdate(BaseSchema):
    """Schema for updating a subcategory."""

    name: str | None = Field(None, min_length=1, max_length=100)
    description: str | None = None
    display_order: int | None = Field(None, ge=0)
    is_active: bool | None = None


class SubcategoryResponse(SubcategoryBase, TimestampMixin):
    """Schema for subcategory response."""

    id: int
    category_id: int


# ============================================================================
# MENU ITEM SCHEMAS
# ============================================================================


class MenuItemBase(BaseSchema):
    """Base menu item schema."""

    name: str = Field(..., min_length=1, max_length=200)
    description: str | None = None
    cost: Decimal = Field(..., ge=0, decimal_places=2)
    price: Decimal = Field(..., ge=0, decimal_places=2)
    prep_time_minutes: int = Field(default=15, ge=0)
    calories: int | None = Field(None, ge=0)
    is_vegetarian: bool = False
    is_vegan: bool = False
    is_gluten_free: bool = False
    allergens: list[str] | None = None
    image_url: str | None = None
    is_active: bool = True
    is_featured: bool = False


class MenuItemCreate(MenuItemBase):
    """Schema for creating a menu item."""

    subcategory_id: int
    sku: str | None = None


class MenuItemUpdate(BaseSchema):
    """Schema for updating a menu item."""

    name: str | None = Field(None, min_length=1, max_length=200)
    description: str | None = None
    cost: Decimal | None = Field(None, ge=0, decimal_places=2)
    price: Decimal | None = Field(None, ge=0, decimal_places=2)
    prep_time_minutes: int | None = Field(None, ge=0)
    calories: int | None = Field(None, ge=0)
    is_vegetarian: bool | None = None
    is_vegan: bool | None = None
    is_gluten_free: bool | None = None
    allergens: list[str] | None = None
    image_url: str | None = None
    is_active: bool | None = None
    is_featured: bool | None = None


class MenuItemResponse(MenuItemBase, TimestampMixin):
    """Schema for menu item response."""

    id: int
    sku: str
    subcategory_id: int
    category: str | None = None  # Category name for frontend display
    subcategory_name: str | None = None  # Subcategory name for display


class MenuItemWithCategory(MenuItemResponse):
    """Menu item with category info."""

    subcategory: SubcategoryResponse


# ============================================================================
# CUSTOMER SCHEMAS
# ============================================================================


class CustomerBase(BaseSchema):
    """Base customer schema."""

    first_name: str = Field(..., min_length=1, max_length=100)
    last_name: str = Field(..., min_length=1, max_length=100)
    email: EmailStr | None = None
    phone: str | None = Field(None, max_length=20)
    dietary_preferences: list[str] | None = None
    notes: str | None = None


class CustomerCreate(CustomerBase):
    """Schema for creating a customer."""

    pass


class CustomerUpdate(BaseSchema):
    """Schema for updating a customer."""

    first_name: str | None = Field(None, min_length=1, max_length=100)
    last_name: str | None = Field(None, min_length=1, max_length=100)
    email: EmailStr | None = None
    phone: str | None = Field(None, max_length=20)
    dietary_preferences: list[str] | None = None
    notes: str | None = None
    loyalty_points: int | None = Field(None, ge=0)


class CustomerResponse(CustomerBase, TimestampMixin):
    """Schema for customer response."""

    id: int
    external_id: str
    loyalty_tier: str
    loyalty_points: int
    is_active: bool


# ============================================================================
# ORDER SCHEMAS
# ============================================================================


class OrderItemCreate(BaseSchema):
    """Schema for creating an order item."""

    menu_item_id: int
    quantity: int = Field(default=1, ge=1)
    special_instructions: str | None = None
    modifiers: dict | None = None


class OrderItemResponse(BaseSchema):
    """Schema for order item response."""

    id: int
    menu_item_id: int
    quantity: int
    unit_price: Decimal
    total_price: Decimal
    special_instructions: str | None = None
    modifiers: dict | None = None
    created_at: datetime

    # Include menu item details
    menu_item: MenuItemResponse | None = None


class OrderCreate(BaseSchema):
    """Schema for creating an order."""

    customer_id: int | None = None
    table_number: int | None = None
    order_type: str = "dine_in"
    items: list[OrderItemCreate]
    notes: str | None = None


class OrderUpdate(BaseSchema):
    """Schema for updating an order."""

    status: str | None = None
    table_number: int | None = None
    discount_amount: Decimal | None = Field(None, ge=0)
    tip_amount: Decimal | None = Field(None, ge=0)
    notes: str | None = None


class OrderResponse(BaseSchema, TimestampMixin):
    """Schema for order response."""

    id: int
    order_number: str
    customer_id: int | None = None
    employee_id: int | None = None
    table_number: int | None = None
    order_type: str
    status: str
    subtotal: Decimal
    tax_amount: Decimal
    discount_amount: Decimal
    tip_amount: Decimal
    total: Decimal
    notes: str | None = None
    completed_at: datetime | None = None

    # Include related data
    items: list[OrderItemResponse] = []
    customer: CustomerResponse | None = None


class OrderSummary(BaseSchema):
    """Schema for order summary (list view)."""

    id: int
    order_number: str
    order_type: str
    status: str
    total: Decimal
    table_number: int | None = None
    created_at: datetime


# ============================================================================
# PAYMENT SCHEMAS
# ============================================================================


class PaymentCreate(BaseSchema):
    """Schema for creating a payment."""

    order_id: int
    payment_method: str
    amount: Decimal = Field(..., ge=0)
    tip_amount: Decimal = Field(default=Decimal("0.00"), ge=0)


class PaymentResponse(BaseSchema):
    """Schema for payment response."""

    id: int
    order_id: int
    payment_method: str
    status: str
    amount: Decimal
    tip_amount: Decimal
    transaction_id: str | None = None
    created_at: datetime
    completed_at: datetime | None = None


# ============================================================================
# EMPLOYEE SCHEMAS
# ============================================================================


class EmployeeBase(BaseSchema):
    """Base employee schema."""

    first_name: str = Field(..., min_length=1, max_length=100)
    last_name: str = Field(..., min_length=1, max_length=100)
    email: EmailStr
    role: str = Field(default="server", max_length=50)
    phone: str | None = Field(None, max_length=20)


class EmployeeCreate(EmployeeBase):
    """Schema for creating an employee."""

    password: str = Field(..., min_length=8)


class EmployeeUpdate(BaseSchema):
    """Schema for updating an employee."""

    first_name: str | None = Field(None, min_length=1, max_length=100)
    last_name: str | None = Field(None, min_length=1, max_length=100)
    email: EmailStr | None = None
    role: str | None = Field(None, max_length=50)
    phone: str | None = Field(None, max_length=20)
    is_active: bool | None = None


class EmployeeResponse(EmployeeBase, TimestampMixin):
    """Schema for employee response."""

    id: int
    employee_id: str
    is_active: bool
    last_login: datetime | None = None


# ============================================================================
# INVENTORY SCHEMAS
# ============================================================================


class InventoryItemBase(BaseSchema):
    """Base inventory item schema."""

    name: str = Field(..., min_length=1, max_length=200)
    category: str = Field(..., min_length=1, max_length=100)
    unit: str = Field(..., min_length=1, max_length=20)
    unit_cost: Decimal = Field(..., ge=0)
    reorder_level: Decimal = Field(default=Decimal("0"), ge=0)
    reorder_quantity: Decimal = Field(default=Decimal("0"), ge=0)
    storage_location: str | None = None


class InventoryItemCreate(InventoryItemBase):
    """Schema for creating an inventory item."""

    sku: str | None = None
    supplier_id: int | None = None
    quantity_on_hand: Decimal = Field(default=Decimal("0"), ge=0)


class InventoryItemUpdate(BaseSchema):
    """Schema for updating an inventory item."""

    name: str | None = Field(None, min_length=1, max_length=200)
    category: str | None = Field(None, min_length=1, max_length=100)
    unit: str | None = Field(None, min_length=1, max_length=20)
    unit_cost: Decimal | None = Field(None, ge=0)
    quantity_on_hand: Decimal | None = Field(None, ge=0)
    reorder_level: Decimal | None = Field(None, ge=0)
    reorder_quantity: Decimal | None = Field(None, ge=0)
    supplier_id: int | None = None
    storage_location: str | None = None
    expiry_date: datetime | None = None
    is_active: bool | None = None


class InventoryItemResponse(InventoryItemBase, TimestampMixin):
    """Schema for inventory item response."""

    id: int
    sku: str
    quantity_on_hand: Decimal
    supplier_id: int | None = None
    expiry_date: datetime | None = None
    last_restocked: datetime | None = None
    is_active: bool
    needs_reorder: bool = False


# ============================================================================
# AUTH SCHEMAS
# ============================================================================


class LoginRequest(BaseSchema):
    """Schema for login request."""

    email: EmailStr
    password: str


class TokenResponse(BaseSchema):
    """Schema for token response."""

    access_token: str
    token_type: str = "bearer"
    expires_in: int


# ============================================================================
# ANALYTICS SCHEMAS
# ============================================================================


class DailySalesData(BaseSchema):
    """Schema for daily sales analytics."""

    date: str
    total_revenue: Decimal
    total_orders: int
    total_items_sold: int
    avg_order_value: Decimal


class ItemSalesData(BaseSchema):
    """Schema for item sales analytics."""

    item_id: int
    item_name: str
    category: str
    quantity_sold: int
    revenue: Decimal


class TopSellingItem(BaseSchema):
    """Schema for top-selling item."""

    rank: int
    item_id: int
    item_name: str
    quantity_sold: int
    revenue: Decimal


class SalesSummary(BaseSchema):
    """Schema for sales summary."""

    period: str
    total_revenue: Decimal
    total_orders: int
    total_items_sold: int
    avg_order_value: Decimal
    top_category: str
    top_item: str


# ============================================================================
# PAGINATION
# ============================================================================


class PaginationParams(BaseSchema):
    """Schema for pagination parameters."""

    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=20, ge=1, le=100)


class PaginatedResponse(BaseSchema):
    """Schema for paginated response."""

    items: list
    total: int
    page: int
    page_size: int
    total_pages: int
