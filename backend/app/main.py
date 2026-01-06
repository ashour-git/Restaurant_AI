"""
Restaurant SaaS API - Main Application.

This is the main FastAPI application that serves as the backend for the
Restaurant SaaS platform, providing POS, menu management, analytics,
and AI-powered features.
"""

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

# Load environment variables from .env file FIRST
from dotenv import load_dotenv

load_dotenv()

# Add parent directory to path for ML module imports
parent_dir = Path(__file__).resolve().parent.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from app.api.routes import api_router
from app.api.routes.ml import initialize_all_models
from app.api.routes.ml import router as ml_router
from app.core.config import settings
from app.core.middleware import (
    RateLimitMiddleware,
    RequestIdMiddleware,
    SecurityHeadersMiddleware,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class TimingMiddleware(BaseHTTPMiddleware):
    """Add request timing headers for performance monitoring."""

    async def dispatch(self, request: Request, call_next):
        start_time = time.perf_counter()
        response = await call_next(request)
        process_time = (time.perf_counter() - start_time) * 1000
        response.headers["X-Process-Time-Ms"] = f"{process_time:.2f}"
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.

    Handles startup and shutdown events for the application.
    """
    # Startup
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"Debug mode: {settings.DEBUG}")

    # Initialize database tables
    logger.info("Initializing database...")
    try:
        from app.core.database import Base, engine, async_session_maker
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created/verified successfully")
        
        # Auto-seed database if empty
        await seed_initial_data(async_session_maker)
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")

    # Initialize ML models
    logger.info("Initializing ML models...")
    initialize_all_models(groq_api_key=os.environ.get("GROQ_API_KEY"))

    yield

    # Shutdown
    logger.info("Shutting down application...")


async def seed_initial_data(async_session_maker):
    """Seed database with initial data if empty."""
    from sqlalchemy import select
    from app.models.models import Category, MenuItem, Subcategory
    from decimal import Decimal
    
    async with async_session_maker() as session:
        # Check if data exists
        result = await session.execute(select(Category).limit(1))
        if result.scalar():
            logger.info("Database already has data, skipping seed")
            return
        
        logger.info("Seeding database with initial menu data...")
        
        # Create categories and menu items
        categories_data = [
            {"name": "Appetizers", "description": "Start your meal right", "display_order": 1},
            {"name": "Main Courses", "description": "Hearty main dishes", "display_order": 2},
            {"name": "Desserts", "description": "Sweet endings", "display_order": 3},
            {"name": "Beverages", "description": "Refreshing drinks", "display_order": 4},
        ]
        
        categories = {}
        for cat_data in categories_data:
            category = Category(**cat_data)
            session.add(category)
            await session.flush()
            categories[cat_data["name"]] = category
            
            # Add a default subcategory
            subcategory = Subcategory(
                name="General",
                category_id=category.id,
                display_order=1
            )
            session.add(subcategory)
            await session.flush()
            
            # Add menu items
            menu_items = get_menu_items_for_category(cat_data["name"], subcategory.id)
            for item_data in menu_items:
                item = MenuItem(**item_data)
                session.add(item)
        
        await session.commit()
        logger.info("Database seeded successfully with menu items!")


def get_menu_items_for_category(category_name: str, subcategory_id: int) -> list:
    """Get menu items for a category."""
    from decimal import Decimal
    
    items = {
        "Appetizers": [
            {"name": "Caesar Salad", "description": "Fresh romaine lettuce with caesar dressing", "price": Decimal("12.99"), "cost": Decimal("4.50"), "subcategory_id": subcategory_id, "is_available": True, "is_vegetarian": True},
            {"name": "Soup of the Day", "description": "Chef's daily soup selection", "price": Decimal("8.99"), "cost": Decimal("2.50"), "subcategory_id": subcategory_id, "is_available": True, "is_vegetarian": True},
            {"name": "Chicken Wings", "description": "Crispy wings with your choice of sauce", "price": Decimal("14.99"), "cost": Decimal("5.00"), "subcategory_id": subcategory_id, "is_available": True},
            {"name": "Mozzarella Sticks", "description": "Golden fried mozzarella with marinara", "price": Decimal("10.99"), "cost": Decimal("3.50"), "subcategory_id": subcategory_id, "is_available": True, "is_vegetarian": True},
        ],
        "Main Courses": [
            {"name": "Classic Burger", "description": "Angus beef patty with all the fixings", "price": Decimal("16.99"), "cost": Decimal("6.00"), "subcategory_id": subcategory_id, "is_available": True},
            {"name": "Grilled Salmon", "description": "Atlantic salmon with lemon butter sauce", "price": Decimal("24.99"), "cost": Decimal("10.00"), "subcategory_id": subcategory_id, "is_available": True, "is_gluten_free": True},
            {"name": "Ribeye Steak", "description": "12oz prime ribeye cooked to perfection", "price": Decimal("32.99"), "cost": Decimal("14.00"), "subcategory_id": subcategory_id, "is_available": True, "is_gluten_free": True},
            {"name": "Pasta Carbonara", "description": "Creamy pasta with bacon and parmesan", "price": Decimal("18.99"), "cost": Decimal("5.50"), "subcategory_id": subcategory_id, "is_available": True},
            {"name": "Veggie Burger", "description": "Plant-based patty with fresh vegetables", "price": Decimal("15.99"), "cost": Decimal("5.00"), "subcategory_id": subcategory_id, "is_available": True, "is_vegetarian": True, "is_vegan": True},
            {"name": "Fish & Chips", "description": "Beer-battered cod with crispy fries", "price": Decimal("19.99"), "cost": Decimal("7.00"), "subcategory_id": subcategory_id, "is_available": True},
        ],
        "Desserts": [
            {"name": "Chocolate Lava Cake", "description": "Warm chocolate cake with molten center", "price": Decimal("9.99"), "cost": Decimal("3.00"), "subcategory_id": subcategory_id, "is_available": True, "is_vegetarian": True},
            {"name": "New York Cheesecake", "description": "Classic creamy cheesecake", "price": Decimal("8.99"), "cost": Decimal("2.50"), "subcategory_id": subcategory_id, "is_available": True, "is_vegetarian": True},
            {"name": "Ice Cream Sundae", "description": "Three scoops with toppings", "price": Decimal("7.99"), "cost": Decimal("2.00"), "subcategory_id": subcategory_id, "is_available": True, "is_vegetarian": True, "is_gluten_free": True},
            {"name": "Apple Pie", "description": "Warm apple pie with vanilla ice cream", "price": Decimal("8.99"), "cost": Decimal("2.50"), "subcategory_id": subcategory_id, "is_available": True, "is_vegetarian": True},
        ],
        "Beverages": [
            {"name": "Fresh Lemonade", "description": "House-made lemonade", "price": Decimal("4.99"), "cost": Decimal("1.00"), "subcategory_id": subcategory_id, "is_available": True, "is_vegetarian": True, "is_vegan": True, "is_gluten_free": True},
            {"name": "Iced Tea", "description": "Freshly brewed iced tea", "price": Decimal("3.99"), "cost": Decimal("0.75"), "subcategory_id": subcategory_id, "is_available": True, "is_vegetarian": True, "is_vegan": True, "is_gluten_free": True},
            {"name": "Coffee", "description": "Premium roasted coffee", "price": Decimal("3.49"), "cost": Decimal("0.50"), "subcategory_id": subcategory_id, "is_available": True, "is_vegetarian": True, "is_vegan": True, "is_gluten_free": True},
            {"name": "Soft Drinks", "description": "Coca-Cola, Sprite, or Fanta", "price": Decimal("2.99"), "cost": Decimal("0.50"), "subcategory_id": subcategory_id, "is_available": True, "is_vegetarian": True, "is_vegan": True, "is_gluten_free": True},
            {"name": "Craft Beer", "description": "Selection of local craft beers", "price": Decimal("7.99"), "cost": Decimal("3.00"), "subcategory_id": subcategory_id, "is_available": True, "is_vegetarian": True, "is_vegan": True, "is_gluten_free": True},
        ],
    }
    return items.get(category_name, [])


def create_application() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        FastAPI: Configured application instance.
    """
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="Restaurant SaaS API with POS, menu management, analytics, and AI features",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add GZip compression for responses > 500 bytes
    app.add_middleware(GZipMiddleware, minimum_size=500)

    # Add security headers
    app.add_middleware(SecurityHeadersMiddleware)

    # Add request ID for tracing
    app.add_middleware(RequestIdMiddleware)

    # Add rate limiting (100 requests per minute)
    app.add_middleware(RateLimitMiddleware, requests_per_minute=100)

    # Add timing middleware for performance monitoring
    app.add_middleware(TimingMiddleware)

    # Include API router
    app.include_router(api_router, prefix=settings.API_V1_PREFIX)

    # Include ML router
    app.include_router(ml_router, prefix=settings.API_V1_PREFIX)

    # Health check endpoint
    @app.get("/health", tags=["Health"])
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "app": settings.APP_NAME,
            "version": settings.APP_VERSION,
        }

    @app.get("/", tags=["Root"])
    async def root():
        """Root endpoint with API information."""
        return {
            "message": f"Welcome to {settings.APP_NAME}",
            "version": settings.APP_VERSION,
            "docs": "/docs",
            "health": "/health",
        }

    return app


# Create application instance
app = create_application()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
    )
