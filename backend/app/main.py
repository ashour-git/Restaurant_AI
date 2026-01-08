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
import threading
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

    # Initialize ML models in background thread (non-blocking)
    # This allows the server to start accepting requests immediately
    # ML endpoints return 503 if models aren't ready yet
    def init_ml_background():
        logger.info("Starting background ML model initialization...")
        try:
            initialize_all_models(groq_api_key=os.environ.get("GROQ_API_KEY"))
            logger.info("Background ML initialization complete")
        except Exception as e:
            logger.error(f"Background ML initialization failed: {e}")
    
    ml_thread = threading.Thread(target=init_ml_background, daemon=True)
    ml_thread.start()
    logger.info("Server ready - ML models loading in background")

    yield

    # Shutdown
    logger.info("Shutting down application...")


async def seed_initial_data(async_session_maker):
    """Seed database with initial data if empty."""
    from sqlalchemy import select
    from app.models.models import Category, MenuItem, Subcategory, Customer, Order, OrderItem, OrderStatus, OrderType, LoyaltyTier
    from decimal import Decimal
    from datetime import datetime, timedelta
    import random
    
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
        all_menu_items = []
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
                await session.flush()
                all_menu_items.append(item)
        
        # Create sample customers
        customers_data = [
            {"first_name": "John", "last_name": "Smith", "email": "john.smith@email.com", "phone": "+1-555-0101", "loyalty_tier": LoyaltyTier.GOLD, "loyalty_points": 2500},
            {"first_name": "Sarah", "last_name": "Johnson", "email": "sarah.j@email.com", "phone": "+1-555-0102", "loyalty_tier": LoyaltyTier.SILVER, "loyalty_points": 1200},
            {"first_name": "Michael", "last_name": "Brown", "email": "m.brown@email.com", "phone": "+1-555-0103", "loyalty_tier": LoyaltyTier.PLATINUM, "loyalty_points": 5000},
            {"first_name": "Emily", "last_name": "Davis", "email": "emily.d@email.com", "phone": "+1-555-0104", "loyalty_tier": LoyaltyTier.BRONZE, "loyalty_points": 350},
            {"first_name": "David", "last_name": "Wilson", "email": "d.wilson@email.com", "phone": "+1-555-0105", "loyalty_tier": LoyaltyTier.GOLD, "loyalty_points": 3100},
        ]
        
        customers = []
        for cust_data in customers_data:
            customer = Customer(**cust_data)
            session.add(customer)
            await session.flush()
            customers.append(customer)
        
        # Create sample orders for the past week
        order_types = [OrderType.DINE_IN, OrderType.TAKEOUT, OrderType.DELIVERY]
        
        for days_ago in range(7):
            # 3-8 orders per day
            num_orders = random.randint(3, 8)
            for _ in range(num_orders):
                order_date = datetime.utcnow() - timedelta(days=days_ago, hours=random.randint(10, 21), minutes=random.randint(0, 59))
                customer = random.choice(customers)
                
                order = Order(
                    customer_id=customer.id,
                    order_type=random.choice(order_types),
                    status=OrderStatus.COMPLETED,
                    table_number=random.randint(1, 12) if random.random() > 0.3 else None,
                    subtotal=Decimal("0"),
                    tax_amount=Decimal("0"),
                    total=Decimal("0"),
                    notes="",
                    created_at=order_date,
                    updated_at=order_date,
                    completed_at=order_date,
                )
                session.add(order)
                await session.flush()
                
                # Add 1-4 items per order
                num_items = random.randint(1, 4)
                order_items = random.sample(all_menu_items, min(num_items, len(all_menu_items)))
                subtotal = Decimal("0")
                
                for menu_item in order_items:
                    quantity = random.randint(1, 2)
                    item_total = menu_item.price * quantity
                    subtotal += item_total
                    
                    order_item = OrderItem(
                        order_id=order.id,
                        menu_item_id=menu_item.id,
                        quantity=quantity,
                        unit_price=menu_item.price,
                        total_price=item_total,
                    )
                    session.add(order_item)
                
                # Update order totals
                tax = subtotal * Decimal("0.08")
                order.subtotal = subtotal
                order.tax_amount = tax
                order.total = subtotal + tax
        
        await session.commit()
        logger.info("Database seeded successfully with menu items, customers, and orders!")


def get_menu_items_for_category(category_name: str, subcategory_id: int) -> list:
    """Get menu items for a category with professional images."""
    from decimal import Decimal
    import uuid
    
    def make_sku(prefix: str) -> str:
        return f"{prefix}-{uuid.uuid4().hex[:8].upper()}"
    
    # Professional food images from Unsplash
    items = {
        "Appetizers": [
            {"sku": make_sku("APP"), "name": "Caesar Salad", "description": "Fresh romaine lettuce with caesar dressing, parmesan, and croutons", "price": Decimal("12.99"), "cost": Decimal("4.50"), "subcategory_id": subcategory_id, "is_active": True, "is_vegetarian": True, "image_url": "https://images.unsplash.com/photo-1550304943-4f24f54ddde9?w=400&h=300&fit=crop"},
            {"sku": make_sku("APP"), "name": "Soup of the Day", "description": "Chef's daily soup selection served with artisan bread", "price": Decimal("8.99"), "cost": Decimal("2.50"), "subcategory_id": subcategory_id, "is_active": True, "is_vegetarian": True, "image_url": "https://images.unsplash.com/photo-1547592166-23ac45744acd?w=400&h=300&fit=crop"},
            {"sku": make_sku("APP"), "name": "Chicken Wings", "description": "Crispy wings with your choice of buffalo, BBQ, or honey garlic sauce", "price": Decimal("14.99"), "cost": Decimal("5.00"), "subcategory_id": subcategory_id, "is_active": True, "image_url": "https://images.unsplash.com/photo-1608039829572-9b5bba1dc5b9?w=400&h=300&fit=crop"},
            {"sku": make_sku("APP"), "name": "Mozzarella Sticks", "description": "Golden fried mozzarella served with marinara sauce", "price": Decimal("10.99"), "cost": Decimal("3.50"), "subcategory_id": subcategory_id, "is_active": True, "is_vegetarian": True, "image_url": "https://images.unsplash.com/photo-1531749668029-2db88e27a9d5?w=400&h=300&fit=crop"},
        ],
        "Main Courses": [
            {"sku": make_sku("MAIN"), "name": "Classic Burger", "description": "Angus beef patty with lettuce, tomato, onion, and special sauce", "price": Decimal("16.99"), "cost": Decimal("6.00"), "subcategory_id": subcategory_id, "is_active": True, "image_url": "https://images.unsplash.com/photo-1568901346375-23c9450c58cd?w=400&h=300&fit=crop"},
            {"sku": make_sku("MAIN"), "name": "Grilled Salmon", "description": "Atlantic salmon with lemon butter sauce and seasonal vegetables", "price": Decimal("24.99"), "cost": Decimal("10.00"), "subcategory_id": subcategory_id, "is_active": True, "is_gluten_free": True, "image_url": "https://images.unsplash.com/photo-1467003909585-2f8a72700288?w=400&h=300&fit=crop"},
            {"sku": make_sku("MAIN"), "name": "Ribeye Steak", "description": "12oz prime ribeye cooked to perfection with garlic butter", "price": Decimal("32.99"), "cost": Decimal("14.00"), "subcategory_id": subcategory_id, "is_active": True, "is_gluten_free": True, "image_url": "https://images.unsplash.com/photo-1600891964092-4316c288032e?w=400&h=300&fit=crop"},
            {"sku": make_sku("MAIN"), "name": "Pasta Carbonara", "description": "Creamy pasta with crispy bacon, egg, and parmesan cheese", "price": Decimal("18.99"), "cost": Decimal("5.50"), "subcategory_id": subcategory_id, "is_active": True, "image_url": "https://images.unsplash.com/photo-1612874742237-6526221588e3?w=400&h=300&fit=crop"},
            {"sku": make_sku("MAIN"), "name": "Veggie Burger", "description": "Plant-based patty with avocado, sprouts, and chipotle mayo", "price": Decimal("15.99"), "cost": Decimal("5.00"), "subcategory_id": subcategory_id, "is_active": True, "is_vegetarian": True, "is_vegan": True, "image_url": "https://images.unsplash.com/photo-1520072959219-c595dc870360?w=400&h=300&fit=crop"},
            {"sku": make_sku("MAIN"), "name": "Fish & Chips", "description": "Beer-battered cod with crispy fries and tartar sauce", "price": Decimal("19.99"), "cost": Decimal("7.00"), "subcategory_id": subcategory_id, "is_active": True, "image_url": "https://images.unsplash.com/photo-1579208030886-b937da0925dc?w=400&h=300&fit=crop"},
        ],
        "Desserts": [
            {"sku": make_sku("DES"), "name": "Chocolate Lava Cake", "description": "Warm chocolate cake with molten center and vanilla ice cream", "price": Decimal("9.99"), "cost": Decimal("3.00"), "subcategory_id": subcategory_id, "is_active": True, "is_vegetarian": True, "image_url": "https://images.unsplash.com/photo-1624353365286-3f8d62daad51?w=400&h=300&fit=crop"},
            {"sku": make_sku("DES"), "name": "New York Cheesecake", "description": "Classic creamy cheesecake with berry compote", "price": Decimal("8.99"), "cost": Decimal("2.50"), "subcategory_id": subcategory_id, "is_active": True, "is_vegetarian": True, "image_url": "https://images.unsplash.com/photo-1533134242443-d4fd215305ad?w=400&h=300&fit=crop"},
            {"sku": make_sku("DES"), "name": "Ice Cream Sundae", "description": "Three scoops with chocolate sauce, whipped cream, and cherry", "price": Decimal("7.99"), "cost": Decimal("2.00"), "subcategory_id": subcategory_id, "is_active": True, "is_vegetarian": True, "is_gluten_free": True, "image_url": "https://images.unsplash.com/photo-1563805042-7684c019e1cb?w=400&h=300&fit=crop"},
            {"sku": make_sku("DES"), "name": "Apple Pie", "description": "Warm apple pie with cinnamon and vanilla ice cream", "price": Decimal("8.99"), "cost": Decimal("2.50"), "subcategory_id": subcategory_id, "is_active": True, "is_vegetarian": True, "image_url": "https://images.unsplash.com/photo-1568571780765-9276ac8b75a2?w=400&h=300&fit=crop"},
        ],
        "Beverages": [
            {"sku": make_sku("BEV"), "name": "Fresh Lemonade", "description": "House-made lemonade with fresh mint", "price": Decimal("4.99"), "cost": Decimal("1.00"), "subcategory_id": subcategory_id, "is_active": True, "is_vegetarian": True, "is_vegan": True, "is_gluten_free": True, "image_url": "https://images.unsplash.com/photo-1621263764928-df1444c5e859?w=400&h=300&fit=crop"},
            {"sku": make_sku("BEV"), "name": "Iced Tea", "description": "Freshly brewed iced tea with lemon", "price": Decimal("3.99"), "cost": Decimal("0.75"), "subcategory_id": subcategory_id, "is_active": True, "is_vegetarian": True, "is_vegan": True, "is_gluten_free": True, "image_url": "https://images.unsplash.com/photo-1556679343-c7306c1976bc?w=400&h=300&fit=crop"},
            {"sku": make_sku("BEV"), "name": "Coffee", "description": "Premium roasted Arabica coffee", "price": Decimal("3.49"), "cost": Decimal("0.50"), "subcategory_id": subcategory_id, "is_active": True, "is_vegetarian": True, "is_vegan": True, "is_gluten_free": True, "image_url": "https://images.unsplash.com/photo-1509042239860-f550ce710b93?w=400&h=300&fit=crop"},
            {"sku": make_sku("BEV"), "name": "Soft Drinks", "description": "Coca-Cola, Sprite, or Fanta", "price": Decimal("2.99"), "cost": Decimal("0.50"), "subcategory_id": subcategory_id, "is_active": True, "is_vegetarian": True, "is_vegan": True, "is_gluten_free": True, "image_url": "https://images.unsplash.com/photo-1581636625402-29b2a704ef13?w=400&h=300&fit=crop"},
            {"sku": make_sku("BEV"), "name": "Craft Beer", "description": "Selection of local craft beers on tap", "price": Decimal("7.99"), "cost": Decimal("3.00"), "subcategory_id": subcategory_id, "is_active": True, "is_vegetarian": True, "is_vegan": True, "is_gluten_free": True, "image_url": "https://images.unsplash.com/photo-1535958636474-b021ee887b13?w=400&h=300&fit=crop"},
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

    # Health check endpoint - always returns healthy for Fly.io
    # ML model status is separate from app health
    @app.get("/health", tags=["Health"])
    async def health_check():
        """Health check endpoint."""
        from app.api.routes.ml import _demand_forecaster, _recommender, _assistant
        
        return {
            "status": "healthy",
            "app": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "ml_models": {
                "demand_forecaster": "ready" if _demand_forecaster else "loading",
                "recommender": "ready" if _recommender else "loading",
                "assistant": "ready" if _assistant else "loading",
            }
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
