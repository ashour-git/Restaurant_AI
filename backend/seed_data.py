"""
Seed database with sample data for the Restaurant SaaS platform.
Run this script to populate the database with realistic test data.
"""

import asyncio
import random
from datetime import datetime, timedelta
from decimal import Decimal

from app.core.database import Base, async_session_maker, engine
from app.models.models import (
    Category,
    Customer,
    Employee,
    InventoryItem,
    LoyaltyTier,
    MenuItem,
    Order,
    OrderItem,
    OrderStatus,
    OrderType,
    Payment,
    PaymentMethod,
    PaymentStatus,
    Subcategory,
)


async def seed_database():
    """Seed the database with sample data."""

    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with async_session_maker() as session:
        # Check if data already exists
        from sqlalchemy import select
        existing = await session.execute(select(Category).limit(1))
        if existing.scalar():
            print("Database already has data. Skipping seed.")
            return

        print("Seeding database with sample data...")

        # ========== CATEGORIES & SUBCATEGORIES ==========
        categories_data = [
            {
                "name": "Appetizers",
                "description": "Start your meal with our delicious appetizers",
                "display_order": 1,
                "subcategories": [
                    {"name": "Soups", "display_order": 1},
                    {"name": "Salads", "display_order": 2},
                    {"name": "Starters", "display_order": 3},
                ]
            },
            {
                "name": "Main Courses",
                "description": "Hearty main dishes for every appetite",
                "display_order": 2,
                "subcategories": [
                    {"name": "Burgers", "display_order": 1},
                    {"name": "Steaks", "display_order": 2},
                    {"name": "Seafood", "display_order": 3},
                    {"name": "Pasta", "display_order": 4},
                    {"name": "Sandwiches", "display_order": 5},
                ]
            },
            {
                "name": "Sides",
                "description": "Perfect accompaniments to your meal",
                "display_order": 3,
                "subcategories": [
                    {"name": "Potatoes", "display_order": 1},
                    {"name": "Vegetables", "display_order": 2},
                ]
            },
            {
                "name": "Desserts",
                "description": "Sweet endings to your perfect meal",
                "display_order": 4,
                "subcategories": [
                    {"name": "Cakes", "display_order": 1},
                    {"name": "Ice Cream", "display_order": 2},
                    {"name": "Pies", "display_order": 3},
                ]
            },
            {
                "name": "Beverages",
                "description": "Refreshing drinks and beverages",
                "display_order": 5,
                "subcategories": [
                    {"name": "Soft Drinks", "display_order": 1},
                    {"name": "Hot Drinks", "display_order": 2},
                    {"name": "Alcoholic", "display_order": 3},
                ]
            },
        ]

        subcategory_map = {}
        for cat_data in categories_data:
            subcats = cat_data.pop("subcategories")
            category = Category(**cat_data)
            session.add(category)
            await session.flush()

            for subcat_data in subcats:
                subcat = Subcategory(category_id=category.id, **subcat_data)
                session.add(subcat)
                await session.flush()
                subcategory_map[f"{cat_data['name']}_{subcat_data['name']}"] = subcat.id

        print(f"Created {len(categories_data)} categories with subcategories")

        # ========== MENU ITEMS ==========
        menu_items_data = [
            # Appetizers - Soups
            {"sku": "APP-SOUP-001", "subcategory_key": "Appetizers_Soups", "name": "French Onion Soup", "description": "Classic caramelized onion soup with melted gruyère cheese", "cost": 3.50, "price": 8.99, "prep_time_minutes": 10, "calories": 320, "is_vegetarian": True},
            {"sku": "APP-SOUP-002", "subcategory_key": "Appetizers_Soups", "name": "Lobster Bisque", "description": "Creamy lobster soup with sherry and fresh herbs", "cost": 5.50, "price": 12.99, "prep_time_minutes": 10, "calories": 380},
            {"sku": "APP-SOUP-003", "subcategory_key": "Appetizers_Soups", "name": "Tomato Basil Soup", "description": "Creamy tomato soup with fresh basil", "cost": 2.50, "price": 7.49, "prep_time_minutes": 8, "calories": 240, "is_vegetarian": True, "is_vegan": True, "is_gluten_free": True},

            # Appetizers - Salads
            {"sku": "APP-SAL-001", "subcategory_key": "Appetizers_Salads", "name": "Caesar Salad", "description": "Romaine lettuce, parmesan, croutons, caesar dressing", "cost": 4.00, "price": 11.99, "prep_time_minutes": 8, "calories": 380},
            {"sku": "APP-SAL-002", "subcategory_key": "Appetizers_Salads", "name": "Greek Salad", "description": "Cucumber, tomato, olives, feta cheese, olive oil", "cost": 4.50, "price": 12.49, "prep_time_minutes": 8, "calories": 320, "is_vegetarian": True, "is_gluten_free": True},
            {"sku": "APP-SAL-003", "subcategory_key": "Appetizers_Salads", "name": "Garden Salad", "description": "Mixed greens with house vinaigrette", "cost": 3.00, "price": 8.99, "prep_time_minutes": 5, "calories": 180, "is_vegetarian": True, "is_vegan": True, "is_gluten_free": True},

            # Appetizers - Starters
            {"sku": "APP-STA-001", "subcategory_key": "Appetizers_Starters", "name": "Mozzarella Sticks", "description": "Crispy fried mozzarella with marinara sauce", "cost": 3.50, "price": 9.99, "prep_time_minutes": 8, "calories": 450, "is_vegetarian": True},
            {"sku": "APP-STA-002", "subcategory_key": "Appetizers_Starters", "name": "Buffalo Wings", "description": "Crispy chicken wings with buffalo sauce", "cost": 5.00, "price": 13.99, "prep_time_minutes": 15, "calories": 580, "is_gluten_free": True},
            {"sku": "APP-STA-003", "subcategory_key": "Appetizers_Starters", "name": "Calamari Fritti", "description": "Crispy fried calamari with aioli", "cost": 6.00, "price": 14.99, "prep_time_minutes": 12, "calories": 420},
            {"sku": "APP-STA-004", "subcategory_key": "Appetizers_Starters", "name": "Spinach Artichoke Dip", "description": "Creamy spinach dip with tortilla chips", "cost": 4.00, "price": 11.99, "prep_time_minutes": 10, "calories": 520, "is_vegetarian": True},

            # Main - Burgers
            {"sku": "MAIN-BUR-001", "subcategory_key": "Main Courses_Burgers", "name": "Classic Burger", "description": "Angus beef patty with lettuce, tomato, onion", "cost": 5.50, "price": 14.99, "prep_time_minutes": 15, "calories": 720, "is_featured": True},
            {"sku": "MAIN-BUR-002", "subcategory_key": "Main Courses_Burgers", "name": "Bacon Cheeseburger", "description": "Classic burger with bacon and cheddar", "cost": 6.50, "price": 16.99, "prep_time_minutes": 15, "calories": 890, "is_featured": True},
            {"sku": "MAIN-BUR-003", "subcategory_key": "Main Courses_Burgers", "name": "Mushroom Swiss Burger", "description": "Beef patty with mushrooms and swiss cheese", "cost": 6.00, "price": 15.99, "prep_time_minutes": 15, "calories": 780},
            {"sku": "MAIN-BUR-004", "subcategory_key": "Main Courses_Burgers", "name": "Veggie Burger", "description": "House-made black bean patty", "cost": 4.50, "price": 13.99, "prep_time_minutes": 12, "calories": 520, "is_vegetarian": True, "is_vegan": True},
            {"sku": "MAIN-BUR-005", "subcategory_key": "Main Courses_Burgers", "name": "Turkey Burger", "description": "Lean turkey patty with avocado", "cost": 5.00, "price": 14.49, "prep_time_minutes": 15, "calories": 580},

            # Main - Steaks
            {"sku": "MAIN-STK-001", "subcategory_key": "Main Courses_Steaks", "name": "Ribeye Steak", "description": "12oz USDA Prime ribeye, grilled to perfection", "cost": 18.00, "price": 38.99, "prep_time_minutes": 25, "calories": 850, "is_gluten_free": True, "is_featured": True},
            {"sku": "MAIN-STK-002", "subcategory_key": "Main Courses_Steaks", "name": "Filet Mignon", "description": "8oz center-cut tenderloin", "cost": 20.00, "price": 42.99, "prep_time_minutes": 25, "calories": 620, "is_gluten_free": True, "is_featured": True},
            {"sku": "MAIN-STK-003", "subcategory_key": "Main Courses_Steaks", "name": "New York Strip", "description": "14oz NY strip with herb butter", "cost": 16.00, "price": 34.99, "prep_time_minutes": 25, "calories": 780, "is_gluten_free": True},
            {"sku": "MAIN-STK-004", "subcategory_key": "Main Courses_Steaks", "name": "Sirloin Steak", "description": "10oz sirloin with peppercorn sauce", "cost": 12.00, "price": 26.99, "prep_time_minutes": 20, "calories": 680, "is_gluten_free": True},

            # Main - Seafood
            {"sku": "MAIN-SEA-001", "subcategory_key": "Main Courses_Seafood", "name": "Grilled Salmon", "description": "Atlantic salmon with lemon dill sauce", "cost": 12.00, "price": 24.99, "prep_time_minutes": 18, "calories": 520, "is_gluten_free": True, "is_featured": True},
            {"sku": "MAIN-SEA-002", "subcategory_key": "Main Courses_Seafood", "name": "Fish and Chips", "description": "Beer-battered cod with fries", "cost": 8.00, "price": 17.99, "prep_time_minutes": 15, "calories": 850},
            {"sku": "MAIN-SEA-003", "subcategory_key": "Main Courses_Seafood", "name": "Shrimp Scampi", "description": "Garlic butter shrimp over linguine", "cost": 10.00, "price": 22.99, "prep_time_minutes": 15, "calories": 680},
            {"sku": "MAIN-SEA-004", "subcategory_key": "Main Courses_Seafood", "name": "Lobster Tail", "description": "8oz Maine lobster tail with drawn butter", "cost": 25.00, "price": 49.99, "prep_time_minutes": 20, "calories": 420, "is_gluten_free": True},
            {"sku": "MAIN-SEA-005", "subcategory_key": "Main Courses_Seafood", "name": "Crab Cakes", "description": "Maryland-style crab cakes with remoulade", "cost": 14.00, "price": 28.99, "prep_time_minutes": 18, "calories": 480},

            # Main - Pasta
            {"sku": "MAIN-PAS-001", "subcategory_key": "Main Courses_Pasta", "name": "Spaghetti Bolognese", "description": "Classic meat sauce over spaghetti", "cost": 6.00, "price": 15.99, "prep_time_minutes": 12, "calories": 720},
            {"sku": "MAIN-PAS-002", "subcategory_key": "Main Courses_Pasta", "name": "Fettuccine Alfredo", "description": "Creamy parmesan sauce over fettuccine", "cost": 5.50, "price": 14.99, "prep_time_minutes": 12, "calories": 880, "is_vegetarian": True},
            {"sku": "MAIN-PAS-003", "subcategory_key": "Main Courses_Pasta", "name": "Chicken Parmesan", "description": "Breaded chicken with marinara and mozzarella", "cost": 8.00, "price": 18.99, "prep_time_minutes": 18, "calories": 920, "is_featured": True},
            {"sku": "MAIN-PAS-004", "subcategory_key": "Main Courses_Pasta", "name": "Lasagna", "description": "Layers of pasta, meat, and cheese", "cost": 7.00, "price": 16.99, "prep_time_minutes": 15, "calories": 850},
            {"sku": "MAIN-PAS-005", "subcategory_key": "Main Courses_Pasta", "name": "Vegetable Primavera", "description": "Seasonal vegetables in garlic olive oil", "cost": 5.00, "price": 13.99, "prep_time_minutes": 12, "calories": 480, "is_vegetarian": True, "is_vegan": True},

            # Main - Sandwiches
            {"sku": "MAIN-SAN-001", "subcategory_key": "Main Courses_Sandwiches", "name": "Club Sandwich", "description": "Triple-decker with turkey, bacon, lettuce, tomato", "cost": 5.50, "price": 13.99, "prep_time_minutes": 10, "calories": 680},
            {"sku": "MAIN-SAN-002", "subcategory_key": "Main Courses_Sandwiches", "name": "Philly Cheesesteak", "description": "Sliced beef with peppers, onions, cheese", "cost": 6.50, "price": 15.99, "prep_time_minutes": 12, "calories": 780},
            {"sku": "MAIN-SAN-003", "subcategory_key": "Main Courses_Sandwiches", "name": "BLT", "description": "Bacon, lettuce, tomato on toasted bread", "cost": 4.00, "price": 10.99, "prep_time_minutes": 8, "calories": 520},
            {"sku": "MAIN-SAN-004", "subcategory_key": "Main Courses_Sandwiches", "name": "Reuben", "description": "Corned beef, sauerkraut, swiss, thousand island", "cost": 6.00, "price": 14.99, "prep_time_minutes": 10, "calories": 720},
            {"sku": "MAIN-SAN-005", "subcategory_key": "Main Courses_Sandwiches", "name": "Grilled Chicken Sandwich", "description": "Grilled chicken breast with avocado", "cost": 5.00, "price": 13.49, "prep_time_minutes": 12, "calories": 580},

            # Sides - Potatoes
            {"sku": "SIDE-POT-001", "subcategory_key": "Sides_Potatoes", "name": "French Fries", "description": "Crispy golden fries", "cost": 2.00, "price": 5.99, "prep_time_minutes": 8, "calories": 380, "is_vegetarian": True, "is_vegan": True, "is_gluten_free": True},
            {"sku": "SIDE-POT-002", "subcategory_key": "Sides_Potatoes", "name": "Sweet Potato Fries", "description": "Crispy sweet potato fries", "cost": 2.50, "price": 6.99, "prep_time_minutes": 8, "calories": 320, "is_vegetarian": True, "is_vegan": True, "is_gluten_free": True},
            {"sku": "SIDE-POT-003", "subcategory_key": "Sides_Potatoes", "name": "Mashed Potatoes", "description": "Creamy garlic mashed potatoes", "cost": 2.00, "price": 5.49, "prep_time_minutes": 5, "calories": 280, "is_vegetarian": True, "is_gluten_free": True},
            {"sku": "SIDE-POT-004", "subcategory_key": "Sides_Potatoes", "name": "Baked Potato", "description": "Loaded with butter and sour cream", "cost": 2.50, "price": 6.49, "prep_time_minutes": 5, "calories": 350, "is_vegetarian": True, "is_gluten_free": True},
            {"sku": "SIDE-POT-005", "subcategory_key": "Sides_Potatoes", "name": "Onion Rings", "description": "Beer-battered onion rings", "cost": 2.50, "price": 6.99, "prep_time_minutes": 8, "calories": 420, "is_vegetarian": True},

            # Sides - Vegetables
            {"sku": "SIDE-VEG-001", "subcategory_key": "Sides_Vegetables", "name": "Steamed Broccoli", "description": "Fresh steamed broccoli", "cost": 1.50, "price": 4.99, "prep_time_minutes": 6, "calories": 55, "is_vegetarian": True, "is_vegan": True, "is_gluten_free": True},
            {"sku": "SIDE-VEG-002", "subcategory_key": "Sides_Vegetables", "name": "Grilled Asparagus", "description": "Grilled asparagus with lemon", "cost": 2.00, "price": 5.99, "prep_time_minutes": 8, "calories": 80, "is_vegetarian": True, "is_vegan": True, "is_gluten_free": True},
            {"sku": "SIDE-VEG-003", "subcategory_key": "Sides_Vegetables", "name": "Coleslaw", "description": "Creamy house-made coleslaw", "cost": 1.00, "price": 3.99, "prep_time_minutes": 2, "calories": 180, "is_vegetarian": True, "is_gluten_free": True},
            {"sku": "SIDE-VEG-004", "subcategory_key": "Sides_Vegetables", "name": "Rice Pilaf", "description": "Herb butter rice pilaf", "cost": 1.50, "price": 4.49, "prep_time_minutes": 5, "calories": 220, "is_vegetarian": True, "is_vegan": True, "is_gluten_free": True},

            # Desserts - Cakes
            {"sku": "DES-CAK-001", "subcategory_key": "Desserts_Cakes", "name": "Chocolate Lava Cake", "description": "Warm chocolate cake with molten center", "cost": 3.50, "price": 8.99, "prep_time_minutes": 12, "calories": 580, "is_vegetarian": True, "is_featured": True},
            {"sku": "DES-CAK-002", "subcategory_key": "Desserts_Cakes", "name": "New York Cheesecake", "description": "Creamy cheesecake with berry compote", "cost": 3.00, "price": 7.99, "prep_time_minutes": 5, "calories": 450, "is_vegetarian": True},
            {"sku": "DES-CAK-003", "subcategory_key": "Desserts_Cakes", "name": "Tiramisu", "description": "Classic Italian coffee-flavored dessert", "cost": 3.50, "price": 8.49, "prep_time_minutes": 5, "calories": 420, "is_vegetarian": True},
            {"sku": "DES-CAK-004", "subcategory_key": "Desserts_Cakes", "name": "Carrot Cake", "description": "Moist carrot cake with cream cheese frosting", "cost": 3.00, "price": 7.49, "prep_time_minutes": 5, "calories": 480, "is_vegetarian": True},

            # Desserts - Ice Cream
            {"sku": "DES-ICE-001", "subcategory_key": "Desserts_Ice Cream", "name": "Vanilla Ice Cream", "description": "Three scoops of premium vanilla", "cost": 2.00, "price": 5.99, "prep_time_minutes": 3, "calories": 380, "is_vegetarian": True, "is_gluten_free": True},
            {"sku": "DES-ICE-002", "subcategory_key": "Desserts_Ice Cream", "name": "Chocolate Sundae", "description": "Ice cream, hot fudge, whipped cream", "cost": 2.50, "price": 6.99, "prep_time_minutes": 5, "calories": 520, "is_vegetarian": True, "is_gluten_free": True},
            {"sku": "DES-ICE-003", "subcategory_key": "Desserts_Ice Cream", "name": "Brownie Sundae", "description": "Warm brownie with ice cream", "cost": 3.00, "price": 7.99, "prep_time_minutes": 8, "calories": 680, "is_vegetarian": True},

            # Desserts - Pies
            {"sku": "DES-PIE-001", "subcategory_key": "Desserts_Pies", "name": "Apple Pie", "description": "Warm apple pie with vanilla ice cream", "cost": 2.50, "price": 6.99, "prep_time_minutes": 8, "calories": 420, "is_vegetarian": True},
            {"sku": "DES-PIE-002", "subcategory_key": "Desserts_Pies", "name": "Pecan Pie", "description": "Southern-style pecan pie", "cost": 3.00, "price": 7.49, "prep_time_minutes": 5, "calories": 520, "is_vegetarian": True},
            {"sku": "DES-PIE-003", "subcategory_key": "Desserts_Pies", "name": "Key Lime Pie", "description": "Tangy key lime pie with whipped cream", "cost": 3.00, "price": 7.49, "prep_time_minutes": 5, "calories": 380, "is_vegetarian": True},

            # Beverages - Soft Drinks
            {"sku": "BEV-SOF-001", "subcategory_key": "Beverages_Soft Drinks", "name": "Coca-Cola", "description": "Classic Coca-Cola", "cost": 0.50, "price": 2.99, "prep_time_minutes": 1, "calories": 140, "is_vegetarian": True, "is_vegan": True, "is_gluten_free": True},
            {"sku": "BEV-SOF-002", "subcategory_key": "Beverages_Soft Drinks", "name": "Sprite", "description": "Lemon-lime soda", "cost": 0.50, "price": 2.99, "prep_time_minutes": 1, "calories": 140, "is_vegetarian": True, "is_vegan": True, "is_gluten_free": True},
            {"sku": "BEV-SOF-003", "subcategory_key": "Beverages_Soft Drinks", "name": "Iced Tea", "description": "Fresh brewed iced tea", "cost": 0.30, "price": 2.49, "prep_time_minutes": 1, "calories": 0, "is_vegetarian": True, "is_vegan": True, "is_gluten_free": True},
            {"sku": "BEV-SOF-004", "subcategory_key": "Beverages_Soft Drinks", "name": "Lemonade", "description": "Fresh-squeezed lemonade", "cost": 0.50, "price": 3.49, "prep_time_minutes": 2, "calories": 120, "is_vegetarian": True, "is_vegan": True, "is_gluten_free": True},

            # Beverages - Hot Drinks
            {"sku": "BEV-HOT-001", "subcategory_key": "Beverages_Hot Drinks", "name": "Coffee", "description": "Fresh brewed coffee", "cost": 0.30, "price": 2.99, "prep_time_minutes": 2, "calories": 5, "is_vegetarian": True, "is_vegan": True, "is_gluten_free": True},
            {"sku": "BEV-HOT-002", "subcategory_key": "Beverages_Hot Drinks", "name": "Cappuccino", "description": "Espresso with steamed milk foam", "cost": 0.80, "price": 4.49, "prep_time_minutes": 3, "calories": 80, "is_vegetarian": True, "is_gluten_free": True},
            {"sku": "BEV-HOT-003", "subcategory_key": "Beverages_Hot Drinks", "name": "Hot Tea", "description": "Selection of premium teas", "cost": 0.25, "price": 2.49, "prep_time_minutes": 2, "calories": 0, "is_vegetarian": True, "is_vegan": True, "is_gluten_free": True},

            # Beverages - Alcoholic
            {"sku": "BEV-ALC-001", "subcategory_key": "Beverages_Alcoholic", "name": "Draft Beer", "description": "Selection of local craft beers", "cost": 2.00, "price": 6.99, "prep_time_minutes": 1, "calories": 180, "is_vegetarian": True, "is_vegan": True},
            {"sku": "BEV-ALC-002", "subcategory_key": "Beverages_Alcoholic", "name": "House Red Wine", "description": "Glass of house cabernet", "cost": 3.00, "price": 8.99, "prep_time_minutes": 1, "calories": 125, "is_vegetarian": True, "is_vegan": True, "is_gluten_free": True},
            {"sku": "BEV-ALC-003", "subcategory_key": "Beverages_Alcoholic", "name": "House White Wine", "description": "Glass of house chardonnay", "cost": 3.00, "price": 8.99, "prep_time_minutes": 1, "calories": 120, "is_vegetarian": True, "is_vegan": True, "is_gluten_free": True},
            {"sku": "BEV-ALC-004", "subcategory_key": "Beverages_Alcoholic", "name": "Margarita", "description": "Classic lime margarita", "cost": 2.50, "price": 9.99, "prep_time_minutes": 3, "calories": 280, "is_vegetarian": True, "is_vegan": True, "is_gluten_free": True},
            {"sku": "BEV-ALC-005", "subcategory_key": "Beverages_Alcoholic", "name": "Mojito", "description": "Rum, mint, lime, soda", "cost": 2.50, "price": 9.99, "prep_time_minutes": 3, "calories": 220, "is_vegetarian": True, "is_vegan": True, "is_gluten_free": True},
        ]

        menu_item_ids = []
        for item_data in menu_items_data:
            subcat_key = item_data.pop("subcategory_key")
            item = MenuItem(
                subcategory_id=subcategory_map[subcat_key],
                cost=Decimal(str(item_data.pop("cost"))),
                price=Decimal(str(item_data.pop("price"))),
                **item_data
            )
            session.add(item)
            await session.flush()
            menu_item_ids.append(item.id)

        print(f"Created {len(menu_items_data)} menu items")

        # ========== EMPLOYEES ==========
        employees_data = [
            {"first_name": "John", "last_name": "Smith", "email": "john.smith@restaurant.com", "phone": "555-0101", "role": "manager"},
            {"first_name": "Sarah", "last_name": "Johnson", "email": "sarah.johnson@restaurant.com", "phone": "555-0102", "role": "server"},
            {"first_name": "Mike", "last_name": "Williams", "email": "mike.williams@restaurant.com", "phone": "555-0103", "role": "chef"},
            {"first_name": "Emily", "last_name": "Brown", "email": "emily.brown@restaurant.com", "phone": "555-0104", "role": "server"},
            {"first_name": "David", "last_name": "Davis", "email": "david.davis@restaurant.com", "phone": "555-0105", "role": "bartender"},
            {"first_name": "Lisa", "last_name": "Miller", "email": "lisa.miller@restaurant.com", "phone": "555-0106", "role": "host"},
            {"first_name": "James", "last_name": "Wilson", "email": "james.wilson@restaurant.com", "phone": "555-0107", "role": "chef"},
            {"first_name": "Amanda", "last_name": "Taylor", "email": "amanda.taylor@restaurant.com", "phone": "555-0108", "role": "server"},
        ]

        employee_ids = []
        for emp_data in employees_data:
            emp = Employee(password_hash="hashed_password_placeholder", **emp_data)
            session.add(emp)
            await session.flush()
            employee_ids.append(emp.id)

        print(f"Created {len(employees_data)} employees")

        # ========== CUSTOMERS ==========
        customers_data = [
            {"first_name": "Alice", "last_name": "Anderson", "email": "alice@email.com", "phone": "555-1001", "loyalty_tier": LoyaltyTier.GOLD, "loyalty_points": 2500},
            {"first_name": "Bob", "last_name": "Baker", "email": "bob@email.com", "phone": "555-1002", "loyalty_tier": LoyaltyTier.SILVER, "loyalty_points": 1200},
            {"first_name": "Carol", "last_name": "Clark", "email": "carol@email.com", "phone": "555-1003", "loyalty_tier": LoyaltyTier.PLATINUM, "loyalty_points": 5000},
            {"first_name": "Daniel", "last_name": "Davis", "email": "daniel@email.com", "phone": "555-1004", "loyalty_tier": LoyaltyTier.BRONZE, "loyalty_points": 300},
            {"first_name": "Emma", "last_name": "Evans", "email": "emma@email.com", "phone": "555-1005", "loyalty_tier": LoyaltyTier.SILVER, "loyalty_points": 850},
            {"first_name": "Frank", "last_name": "Foster", "email": "frank@email.com", "phone": "555-1006", "loyalty_tier": LoyaltyTier.GOLD, "loyalty_points": 2100},
            {"first_name": "Grace", "last_name": "Green", "email": "grace@email.com", "phone": "555-1007", "loyalty_tier": LoyaltyTier.BRONZE, "loyalty_points": 150},
            {"first_name": "Henry", "last_name": "Harris", "email": "henry@email.com", "phone": "555-1008", "loyalty_tier": LoyaltyTier.SILVER, "loyalty_points": 920},
            {"first_name": "Ivy", "last_name": "Irving", "email": "ivy@email.com", "phone": "555-1009", "loyalty_tier": LoyaltyTier.GOLD, "loyalty_points": 1850},
            {"first_name": "Jack", "last_name": "Jackson", "email": "jack@email.com", "phone": "555-1010", "loyalty_tier": LoyaltyTier.BRONZE, "loyalty_points": 400},
            {"first_name": "Karen", "last_name": "King", "email": "karen@email.com", "phone": "555-1011", "loyalty_tier": LoyaltyTier.PLATINUM, "loyalty_points": 6200},
            {"first_name": "Leo", "last_name": "Lewis", "email": "leo@email.com", "phone": "555-1012", "loyalty_tier": LoyaltyTier.SILVER, "loyalty_points": 780},
            {"first_name": "Mia", "last_name": "Moore", "email": "mia@email.com", "phone": "555-1013", "loyalty_tier": LoyaltyTier.GOLD, "loyalty_points": 2300},
            {"first_name": "Noah", "last_name": "Nelson", "email": "noah@email.com", "phone": "555-1014", "loyalty_tier": LoyaltyTier.BRONZE, "loyalty_points": 220},
            {"first_name": "Olivia", "last_name": "Owen", "email": "olivia@email.com", "phone": "555-1015", "loyalty_tier": LoyaltyTier.SILVER, "loyalty_points": 1100},
        ]

        customer_ids = []
        for cust_data in customers_data:
            customer = Customer(**cust_data)
            session.add(customer)
            await session.flush()
            customer_ids.append(customer.id)

        print(f"Created {len(customers_data)} customers")

        # ========== INVENTORY ITEMS ==========
        inventory_data = [
            {"name": "Ground Beef", "sku": "INV-MEAT-001", "category": "Meat", "unit": "lb", "quantity_on_hand": 50, "reorder_level": 20, "unit_cost": 5.99},
            {"name": "Chicken Breast", "sku": "INV-MEAT-002", "category": "Meat", "unit": "lb", "quantity_on_hand": 40, "reorder_level": 15, "unit_cost": 4.99},
            {"name": "Salmon Fillet", "sku": "INV-FISH-001", "category": "Seafood", "unit": "lb", "quantity_on_hand": 25, "reorder_level": 10, "unit_cost": 12.99},
            {"name": "Shrimp", "sku": "INV-FISH-002", "category": "Seafood", "unit": "lb", "quantity_on_hand": 20, "reorder_level": 8, "unit_cost": 14.99},
            {"name": "Lobster Tail", "sku": "INV-FISH-003", "category": "Seafood", "unit": "each", "quantity_on_hand": 15, "reorder_level": 5, "unit_cost": 18.99},
            {"name": "Lettuce", "sku": "INV-VEG-001", "category": "Produce", "unit": "head", "quantity_on_hand": 30, "reorder_level": 15, "unit_cost": 1.49},
            {"name": "Tomatoes", "sku": "INV-VEG-002", "category": "Produce", "unit": "lb", "quantity_on_hand": 25, "reorder_level": 10, "unit_cost": 2.49},
            {"name": "Onions", "sku": "INV-VEG-003", "category": "Produce", "unit": "lb", "quantity_on_hand": 40, "reorder_level": 15, "unit_cost": 0.99},
            {"name": "Potatoes", "sku": "INV-VEG-004", "category": "Produce", "unit": "lb", "quantity_on_hand": 60, "reorder_level": 25, "unit_cost": 0.79},
            {"name": "Mushrooms", "sku": "INV-VEG-005", "category": "Produce", "unit": "lb", "quantity_on_hand": 15, "reorder_level": 8, "unit_cost": 4.99},
            {"name": "Cheddar Cheese", "sku": "INV-DAI-001", "category": "Dairy", "unit": "lb", "quantity_on_hand": 20, "reorder_level": 10, "unit_cost": 6.99},
            {"name": "Mozzarella", "sku": "INV-DAI-002", "category": "Dairy", "unit": "lb", "quantity_on_hand": 18, "reorder_level": 8, "unit_cost": 5.99},
            {"name": "Butter", "sku": "INV-DAI-003", "category": "Dairy", "unit": "lb", "quantity_on_hand": 25, "reorder_level": 10, "unit_cost": 4.49},
            {"name": "Heavy Cream", "sku": "INV-DAI-004", "category": "Dairy", "unit": "quart", "quantity_on_hand": 12, "reorder_level": 6, "unit_cost": 3.99},
            {"name": "Eggs", "sku": "INV-DAI-005", "category": "Dairy", "unit": "dozen", "quantity_on_hand": 20, "reorder_level": 10, "unit_cost": 3.49},
            {"name": "Burger Buns", "sku": "INV-BRD-001", "category": "Bakery", "unit": "pack", "quantity_on_hand": 30, "reorder_level": 15, "unit_cost": 2.99},
            {"name": "Bread Loaves", "sku": "INV-BRD-002", "category": "Bakery", "unit": "loaf", "quantity_on_hand": 15, "reorder_level": 8, "unit_cost": 2.49},
            {"name": "Pasta", "sku": "INV-DRY-001", "category": "Dry Goods", "unit": "lb", "quantity_on_hand": 40, "reorder_level": 20, "unit_cost": 1.49},
            {"name": "Rice", "sku": "INV-DRY-002", "category": "Dry Goods", "unit": "lb", "quantity_on_hand": 35, "reorder_level": 15, "unit_cost": 1.29},
            {"name": "Olive Oil", "sku": "INV-DRY-003", "category": "Dry Goods", "unit": "liter", "quantity_on_hand": 10, "reorder_level": 5, "unit_cost": 8.99},
            {"name": "Coca-Cola", "sku": "INV-BEV-001", "category": "Beverages", "unit": "case", "quantity_on_hand": 15, "reorder_level": 8, "unit_cost": 18.99},
            {"name": "Coffee Beans", "sku": "INV-BEV-002", "category": "Beverages", "unit": "lb", "quantity_on_hand": 20, "reorder_level": 10, "unit_cost": 12.99},
            {"name": "Wine (Red)", "sku": "INV-BEV-003", "category": "Beverages", "unit": "bottle", "quantity_on_hand": 25, "reorder_level": 12, "unit_cost": 15.99},
            {"name": "Wine (White)", "sku": "INV-BEV-004", "category": "Beverages", "unit": "bottle", "quantity_on_hand": 20, "reorder_level": 10, "unit_cost": 14.99},
            {"name": "Draft Beer Keg", "sku": "INV-BEV-005", "category": "Beverages", "unit": "keg", "quantity_on_hand": 5, "reorder_level": 3, "unit_cost": 89.99},
        ]

        for inv_data in inventory_data:
            inv = InventoryItem(**inv_data)
            session.add(inv)

        print(f"Created {len(inventory_data)} inventory items")

        # ========== ORDERS (Last 30 days) ==========
        order_types = list(OrderType)
        order_statuses = [OrderStatus.COMPLETED, OrderStatus.COMPLETED, OrderStatus.COMPLETED,
                         OrderStatus.PREPARING, OrderStatus.READY, OrderStatus.PENDING]

        orders_to_create = []
        order_items_to_create = []
        payments_to_create = []

        # Generate orders for last 30 days
        now = datetime.now()
        for days_ago in range(30):
            date = now - timedelta(days=days_ago)
            # More orders on weekends
            is_weekend = date.weekday() >= 5
            num_orders = random.randint(15, 25) if is_weekend else random.randint(8, 15)

            for _ in range(num_orders):
                # Random time during operating hours (11 AM - 10 PM)
                hour = random.randint(11, 21)
                minute = random.randint(0, 59)
                order_time = date.replace(hour=hour, minute=minute, second=0, microsecond=0)

                # Random customer (some orders without customer)
                customer_id = random.choice(customer_ids) if random.random() > 0.3 else None
                employee_id = random.choice(employee_ids)
                order_type = random.choice(order_types)

                # Status based on recency
                if days_ago == 0 and hour >= datetime.now().hour - 2:
                    status = random.choice([OrderStatus.PENDING, OrderStatus.PREPARING, OrderStatus.READY])
                else:
                    status = OrderStatus.COMPLETED

                table_number = random.randint(1, 20) if order_type == OrderType.DINE_IN else None

                order = Order(
                    customer_id=customer_id,
                    employee_id=employee_id,
                    table_number=table_number,
                    order_type=order_type,
                    status=status,
                    subtotal=Decimal("0"),
                    tax_amount=Decimal("0"),
                    discount_amount=Decimal("0"),
                    tip_amount=Decimal(str(random.choice([0, 2, 3, 5, 8, 10]))) if status == OrderStatus.COMPLETED else Decimal("0"),
                    total=Decimal("0"),
                    created_at=order_time,
                    completed_at=order_time + timedelta(minutes=random.randint(20, 45)) if status == OrderStatus.COMPLETED else None,
                )
                orders_to_create.append(order)

        # Add all orders first
        session.add_all(orders_to_create)
        await session.flush()

        # Now create order items for each order
        for order in orders_to_create:
            num_items = random.randint(2, 5)
            selected_items = random.sample(menu_item_ids, num_items)
            subtotal = Decimal("0")

            for item_id in selected_items:
                # Get the menu item price
                from sqlalchemy import select
                result = await session.execute(select(MenuItem).where(MenuItem.id == item_id))
                menu_item = result.scalar()

                quantity = random.randint(1, 3)
                unit_price = menu_item.price
                item_total = unit_price * quantity
                subtotal += item_total

                order_item = OrderItem(
                    order_id=order.id,
                    menu_item_id=item_id,
                    quantity=quantity,
                    unit_price=unit_price,
                    total_price=item_total,
                    special_instructions=random.choice([None, "No onions", "Extra sauce", "Medium rare", None, None]),
                )
                order_items_to_create.append(order_item)

            # Update order totals
            tax_rate = Decimal("0.0825")  # 8.25% tax
            order.subtotal = subtotal
            order.tax_amount = (subtotal * tax_rate).quantize(Decimal("0.01"))
            order.total = order.subtotal + order.tax_amount - order.discount_amount + order.tip_amount

            # Create payment for completed orders
            if order.status == OrderStatus.COMPLETED:
                payment = Payment(
                    order_id=order.id,
                    amount=order.total,
                    payment_method=random.choice(list(PaymentMethod)),
                    status=PaymentStatus.COMPLETED,
                    completed_at=order.completed_at,
                )
                payments_to_create.append(payment)

        session.add_all(order_items_to_create)
        session.add_all(payments_to_create)

        print(f"Created {len(orders_to_create)} orders with {len(order_items_to_create)} order items")
        print(f"Created {len(payments_to_create)} payments")

        await session.commit()
        print("\n[SUCCESS] Database seeding completed successfully!")

        # Print summary
        print("\n[SUMMARY] Data Summary:")
        print(f"   Categories: {len(categories_data)}")
        print(f"   Menu Items: {len(menu_items_data)}")
        print(f"   Employees: {len(employees_data)}")
        print(f"   Customers: {len(customers_data)}")
        print(f"   Inventory Items: {len(inventory_data)}")
        print(f"   Orders: {len(orders_to_create)}")


if __name__ == "__main__":
    asyncio.run(seed_database())
