"""API routes package initialization."""

from app.api.routes import analytics, customers, inventory, menu, orders, auth
from fastapi import APIRouter

api_router = APIRouter()

# Include all routers
api_router.include_router(auth.router)
api_router.include_router(menu.router)
api_router.include_router(orders.router)
api_router.include_router(customers.router)
api_router.include_router(inventory.router)
api_router.include_router(analytics.router)
