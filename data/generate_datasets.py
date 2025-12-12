"""
Dataset Generation Script for Restaurant SaaS Platform.

This script generates synthetic datasets for the Restaurant SaaS platform,
including transactions, menu items, customers, inventory, and co-occurrence data.

Author: Restaurant SaaS Team
Date: December 2025
"""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)


# Configuration
DATA_DIR = Path(__file__).parent
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Ensure directories exist
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# MENU ITEMS GENERATION
# ============================================================================


def generate_menu_items() -> pd.DataFrame:
    """
    Generate synthetic menu items dataset.

    Returns:
        pd.DataFrame: Menu items with fields:
            - item_id: Unique identifier
            - name: Item name
            - description: Item description
            - category: Food category
            - subcategory: Food subcategory
            - cost: Cost to prepare
            - price: Selling price
            - prep_time_minutes: Preparation time
            - is_vegetarian: Boolean flag
            - is_vegan: Boolean flag
            - is_gluten_free: Boolean flag
            - calories: Estimated calories
            - allergens: List of allergens
            - is_active: Whether item is currently available
            - created_at: Creation timestamp

    Example:
        >>> df = generate_menu_items()
        >>> df.head()
    """

    menu_categories = {
        "Appetizers": {
            "Soups": [
                (
                    "Tomato Basil Soup",
                    "Creamy tomato soup with fresh basil",
                    2.50,
                    7.99,
                    10,
                    True,
                    True,
                    True,
                    180,
                ),
                (
                    "French Onion Soup",
                    "Classic French onion soup with gruyere",
                    3.00,
                    9.99,
                    15,
                    True,
                    False,
                    False,
                    320,
                ),
                (
                    "Clam Chowder",
                    "New England style creamy clam chowder",
                    4.00,
                    11.99,
                    12,
                    False,
                    False,
                    False,
                    380,
                ),
                ("Minestrone", "Italian vegetable soup", 2.00, 7.49, 10, True, True, True, 150),
                (
                    "Chicken Noodle Soup",
                    "Homemade chicken noodle soup",
                    2.50,
                    8.49,
                    12,
                    False,
                    False,
                    False,
                    220,
                ),
            ],
            "Salads": [
                (
                    "Caesar Salad",
                    "Romaine lettuce with caesar dressing and croutons",
                    3.00,
                    10.99,
                    8,
                    True,
                    False,
                    False,
                    280,
                ),
                (
                    "Greek Salad",
                    "Mixed greens with feta, olives, and tomatoes",
                    3.50,
                    11.99,
                    8,
                    True,
                    False,
                    True,
                    250,
                ),
                (
                    "Caprese Salad",
                    "Fresh mozzarella, tomatoes, and basil",
                    4.00,
                    12.99,
                    5,
                    True,
                    False,
                    True,
                    320,
                ),
                (
                    "Cobb Salad",
                    "Mixed greens with chicken, bacon, eggs, avocado",
                    5.00,
                    14.99,
                    10,
                    False,
                    False,
                    True,
                    450,
                ),
                (
                    "Asian Sesame Salad",
                    "Crispy wontons with sesame ginger dressing",
                    3.50,
                    11.49,
                    8,
                    True,
                    False,
                    False,
                    290,
                ),
            ],
            "Small Plates": [
                (
                    "Bruschetta",
                    "Toasted bread with tomato basil topping",
                    2.50,
                    8.99,
                    8,
                    True,
                    True,
                    False,
                    180,
                ),
                (
                    "Mozzarella Sticks",
                    "Fried mozzarella with marinara",
                    3.00,
                    9.99,
                    10,
                    True,
                    False,
                    False,
                    420,
                ),
                (
                    "Chicken Wings",
                    "Crispy wings with choice of sauce",
                    4.50,
                    13.99,
                    15,
                    False,
                    False,
                    True,
                    580,
                ),
                (
                    "Calamari",
                    "Fried calamari with aioli",
                    5.00,
                    14.99,
                    12,
                    False,
                    False,
                    False,
                    350,
                ),
                (
                    "Spinach Artichoke Dip",
                    "Creamy dip with tortilla chips",
                    3.50,
                    11.99,
                    10,
                    True,
                    False,
                    False,
                    420,
                ),
                (
                    "Hummus Platter",
                    "House-made hummus with pita bread",
                    3.00,
                    10.99,
                    5,
                    True,
                    True,
                    False,
                    280,
                ),
            ],
        },
        "Main Courses": {
            "Burgers": [
                (
                    "Classic Burger",
                    "Angus beef patty with lettuce, tomato, onion",
                    4.00,
                    14.99,
                    15,
                    False,
                    False,
                    False,
                    650,
                ),
                (
                    "Bacon Cheeseburger",
                    "Classic burger with bacon and cheddar",
                    5.00,
                    16.99,
                    15,
                    False,
                    False,
                    False,
                    780,
                ),
                (
                    "Mushroom Swiss Burger",
                    "Beef patty with mushrooms and swiss cheese",
                    4.50,
                    15.99,
                    15,
                    False,
                    False,
                    False,
                    700,
                ),
                (
                    "Veggie Burger",
                    "House-made black bean patty",
                    3.50,
                    13.99,
                    15,
                    True,
                    True,
                    False,
                    450,
                ),
                (
                    "Turkey Burger",
                    "Lean turkey patty with avocado",
                    4.00,
                    14.49,
                    15,
                    False,
                    False,
                    False,
                    520,
                ),
            ],
            "Sandwiches": [
                (
                    "Club Sandwich",
                    "Triple-decker with turkey, bacon, lettuce",
                    4.50,
                    13.99,
                    12,
                    False,
                    False,
                    False,
                    580,
                ),
                (
                    "Philly Cheesesteak",
                    "Sliced beef with peppers and onions",
                    5.00,
                    15.99,
                    15,
                    False,
                    False,
                    False,
                    720,
                ),
                (
                    "Grilled Chicken Sandwich",
                    "Marinated chicken breast with pesto",
                    4.00,
                    13.49,
                    12,
                    False,
                    False,
                    False,
                    480,
                ),
                (
                    "BLT",
                    "Bacon, lettuce, tomato on sourdough",
                    3.50,
                    11.99,
                    10,
                    False,
                    False,
                    False,
                    420,
                ),
                (
                    "Reuben",
                    "Corned beef with sauerkraut and swiss",
                    5.00,
                    14.99,
                    12,
                    False,
                    False,
                    False,
                    680,
                ),
            ],
            "Pasta": [
                (
                    "Spaghetti Bolognese",
                    "Classic meat sauce over spaghetti",
                    4.00,
                    15.99,
                    20,
                    False,
                    False,
                    False,
                    680,
                ),
                (
                    "Fettuccine Alfredo",
                    "Creamy parmesan sauce with fettuccine",
                    3.50,
                    14.99,
                    18,
                    True,
                    False,
                    False,
                    720,
                ),
                (
                    "Chicken Parmesan",
                    "Breaded chicken with marinara over pasta",
                    5.50,
                    18.99,
                    25,
                    False,
                    False,
                    False,
                    850,
                ),
                (
                    "Shrimp Scampi",
                    "Garlic butter shrimp over linguine",
                    7.00,
                    21.99,
                    20,
                    False,
                    False,
                    False,
                    620,
                ),
                (
                    "Vegetable Primavera",
                    "Seasonal vegetables with olive oil and herbs",
                    3.00,
                    14.49,
                    18,
                    True,
                    True,
                    False,
                    420,
                ),
                (
                    "Lasagna",
                    "Layered pasta with meat sauce and ricotta",
                    4.50,
                    16.99,
                    25,
                    False,
                    False,
                    False,
                    780,
                ),
            ],
            "Steaks": [
                (
                    "Ribeye Steak",
                    "12oz ribeye grilled to perfection",
                    12.00,
                    34.99,
                    25,
                    False,
                    False,
                    True,
                    850,
                ),
                ("Filet Mignon", "8oz center-cut filet", 15.00, 39.99, 25, False, False, True, 580),
                (
                    "New York Strip",
                    "14oz NY strip steak",
                    11.00,
                    32.99,
                    25,
                    False,
                    False,
                    True,
                    780,
                ),
                ("Sirloin Steak", "10oz top sirloin", 8.00, 26.99, 20, False, False, True, 620),
                (
                    "Surf and Turf",
                    "Filet mignon with lobster tail",
                    22.00,
                    54.99,
                    30,
                    False,
                    False,
                    True,
                    920,
                ),
            ],
            "Seafood": [
                (
                    "Grilled Salmon",
                    "Atlantic salmon with lemon dill sauce",
                    8.00,
                    24.99,
                    20,
                    False,
                    False,
                    True,
                    480,
                ),
                (
                    "Fish and Chips",
                    "Beer-battered cod with fries",
                    5.00,
                    16.99,
                    18,
                    False,
                    False,
                    False,
                    780,
                ),
                (
                    "Shrimp Basket",
                    "Fried shrimp with cocktail sauce",
                    6.00,
                    17.99,
                    15,
                    False,
                    False,
                    False,
                    520,
                ),
                (
                    "Lobster Roll",
                    "Maine lobster on a buttered roll",
                    12.00,
                    28.99,
                    12,
                    False,
                    False,
                    False,
                    480,
                ),
                (
                    "Crab Cakes",
                    "Pan-seared crab cakes with remoulade",
                    10.00,
                    26.99,
                    18,
                    False,
                    False,
                    True,
                    380,
                ),
            ],
            "Chicken": [
                (
                    "Grilled Chicken Breast",
                    "Herb-marinated chicken with vegetables",
                    4.00,
                    16.99,
                    18,
                    False,
                    False,
                    True,
                    380,
                ),
                (
                    "Chicken Marsala",
                    "Pan-seared chicken with mushroom wine sauce",
                    5.00,
                    18.99,
                    22,
                    False,
                    False,
                    True,
                    480,
                ),
                (
                    "Chicken Piccata",
                    "Chicken with lemon caper butter sauce",
                    4.50,
                    17.99,
                    20,
                    False,
                    False,
                    True,
                    420,
                ),
                (
                    "Fried Chicken",
                    "Southern-style fried chicken",
                    4.00,
                    15.99,
                    25,
                    False,
                    False,
                    False,
                    680,
                ),
                (
                    "Teriyaki Chicken",
                    "Grilled chicken with teriyaki glaze",
                    4.00,
                    16.49,
                    18,
                    False,
                    False,
                    True,
                    450,
                ),
            ],
        },
        "Sides": {
            "Starches": [
                ("French Fries", "Crispy golden fries", 1.00, 4.99, 8, True, True, True, 320),
                (
                    "Sweet Potato Fries",
                    "Seasoned sweet potato fries",
                    1.50,
                    5.99,
                    10,
                    True,
                    True,
                    True,
                    280,
                ),
                (
                    "Mashed Potatoes",
                    "Creamy garlic mashed potatoes",
                    1.00,
                    4.49,
                    8,
                    True,
                    False,
                    True,
                    220,
                ),
                ("Baked Potato", "Loaded baked potato", 1.00, 4.99, 12, True, False, True, 350),
                ("Rice Pilaf", "Seasoned rice with herbs", 0.75, 3.99, 8, True, True, True, 180),
                (
                    "Onion Rings",
                    "Beer-battered onion rings",
                    1.50,
                    5.99,
                    10,
                    True,
                    False,
                    False,
                    380,
                ),
            ],
            "Vegetables": [
                ("Steamed Broccoli", "Fresh steamed broccoli", 0.75, 3.99, 6, True, True, True, 55),
                (
                    "Grilled Asparagus",
                    "Seasoned grilled asparagus",
                    1.00,
                    4.99,
                    8,
                    True,
                    True,
                    True,
                    40,
                ),
                ("Sauteed Spinach", "Garlic sauteed spinach", 0.75, 3.99, 5, True, True, True, 45),
                ("Coleslaw", "Creamy coleslaw", 0.50, 2.99, 3, True, False, True, 180),
                (
                    "Corn on the Cob",
                    "Buttered corn on the cob",
                    0.75,
                    3.49,
                    8,
                    True,
                    False,
                    True,
                    130,
                ),
            ],
        },
        "Desserts": {
            "Cakes": [
                (
                    "Chocolate Lava Cake",
                    "Warm chocolate cake with molten center",
                    3.00,
                    8.99,
                    12,
                    True,
                    False,
                    False,
                    480,
                ),
                (
                    "New York Cheesecake",
                    "Classic creamy cheesecake",
                    2.50,
                    7.99,
                    5,
                    True,
                    False,
                    False,
                    420,
                ),
                (
                    "Carrot Cake",
                    "Spiced carrot cake with cream cheese frosting",
                    2.50,
                    7.49,
                    5,
                    True,
                    False,
                    False,
                    380,
                ),
                (
                    "Tiramisu",
                    "Italian coffee-flavored dessert",
                    3.00,
                    8.49,
                    5,
                    True,
                    False,
                    False,
                    350,
                ),
            ],
            "Ice Cream": [
                (
                    "Vanilla Ice Cream",
                    "Two scoops of vanilla ice cream",
                    1.50,
                    4.99,
                    3,
                    True,
                    False,
                    True,
                    280,
                ),
                (
                    "Chocolate Sundae",
                    "Ice cream with hot fudge and whipped cream",
                    2.00,
                    6.99,
                    5,
                    True,
                    False,
                    True,
                    420,
                ),
                (
                    "Brownie Sundae",
                    "Warm brownie with ice cream",
                    2.50,
                    7.99,
                    8,
                    True,
                    False,
                    False,
                    580,
                ),
            ],
            "Pies": [
                (
                    "Apple Pie",
                    "Warm apple pie with vanilla ice cream",
                    2.00,
                    6.99,
                    5,
                    True,
                    False,
                    False,
                    350,
                ),
                ("Key Lime Pie", "Tangy key lime pie", 2.00, 6.49, 5, True, False, False, 320),
                ("Pecan Pie", "Rich pecan pie", 2.50, 7.49, 5, True, False, False, 450),
            ],
        },
        "Beverages": {
            "Soft Drinks": [
                ("Coca-Cola", "Classic Coca-Cola", 0.30, 2.99, 1, True, True, True, 140),
                ("Sprite", "Lemon-lime soda", 0.30, 2.99, 1, True, True, True, 140),
                ("Iced Tea", "Freshly brewed iced tea", 0.25, 2.49, 1, True, True, True, 0),
                ("Lemonade", "Fresh-squeezed lemonade", 0.50, 3.49, 2, True, True, True, 120),
                ("Coffee", "Freshly brewed coffee", 0.25, 2.99, 2, True, True, True, 5),
            ],
            "Alcoholic": [
                ("House Red Wine", "Glass of house red wine", 3.00, 8.99, 1, True, True, True, 125),
                (
                    "House White Wine",
                    "Glass of house white wine",
                    3.00,
                    8.99,
                    1,
                    True,
                    True,
                    True,
                    120,
                ),
                ("Draft Beer", "Pint of draft beer", 2.00, 5.99, 1, True, True, True, 150),
                ("Margarita", "Classic lime margarita", 3.00, 10.99, 5, True, True, True, 280),
                ("Mojito", "Rum with mint and lime", 3.50, 11.99, 5, True, True, True, 220),
            ],
        },
    }

    items = []
    item_id = 1

    for category, subcategories in menu_categories.items():
        for subcategory, menu_items in subcategories.items():
            for item in menu_items:
                name, description, cost, price, prep_time, is_veg, is_vegan, is_gf, calories = item

                # Generate allergens based on item characteristics
                allergens = []
                if not is_gf:
                    allergens.append("gluten")
                if "cheese" in name.lower() or "cream" in description.lower() or not is_vegan:
                    if "milk" not in allergens and "dairy" not in description.lower():
                        allergens.append("dairy")
                if "shrimp" in name.lower() or "lobster" in name.lower() or "crab" in name.lower():
                    allergens.append("shellfish")
                if "fish" in name.lower() or "salmon" in name.lower() or "cod" in name.lower():
                    allergens.append("fish")

                items.append(
                    {
                        "item_id": f"ITEM_{item_id:04d}",
                        "name": name,
                        "description": description,
                        "category": category,
                        "subcategory": subcategory,
                        "cost": round(cost, 2),
                        "price": round(price, 2),
                        "prep_time_minutes": prep_time,
                        "is_vegetarian": is_veg,
                        "is_vegan": is_vegan,
                        "is_gluten_free": is_gf,
                        "calories": calories,
                        "allergens": json.dumps(allergens),
                        "is_active": True,
                        "created_at": (
                            datetime(2024, 1, 1) + timedelta(days=random.randint(0, 365))
                        ).isoformat(),
                    }
                )
                item_id += 1

    df = pd.DataFrame(items)
    df.to_csv(RAW_DIR / "menu_items.csv", index=False)
    print(f"Generated {len(df)} menu items -> {RAW_DIR / 'menu_items.csv'}")
    return df


# ============================================================================
# CUSTOMERS GENERATION
# ============================================================================


def generate_customers(num_customers: int = 5000) -> pd.DataFrame:
    """
    Generate synthetic customer profiles.

    Args:
        num_customers: Number of customers to generate.

    Returns:
        pd.DataFrame: Customer profiles with fields:
            - customer_id: Unique identifier
            - first_name: Customer first name
            - last_name: Customer last name
            - email: Customer email
            - phone: Customer phone
            - loyalty_tier: Bronze/Silver/Gold/Platinum
            - loyalty_points: Accumulated points
            - total_orders: Historical order count
            - total_spent: Total amount spent
            - average_order_value: Average order value
            - preferred_category: Most ordered category
            - dietary_preferences: JSON list of preferences
            - created_at: Registration date
            - last_order_date: Most recent order

    Example:
        >>> df = generate_customers(1000)
        >>> df.head()
    """

    first_names = [
        "James",
        "Mary",
        "John",
        "Patricia",
        "Robert",
        "Jennifer",
        "Michael",
        "Linda",
        "William",
        "Elizabeth",
        "David",
        "Barbara",
        "Richard",
        "Susan",
        "Joseph",
        "Jessica",
        "Thomas",
        "Sarah",
        "Charles",
        "Karen",
        "Christopher",
        "Lisa",
        "Daniel",
        "Nancy",
        "Matthew",
        "Betty",
        "Anthony",
        "Margaret",
        "Mark",
        "Sandra",
        "Donald",
        "Ashley",
        "Steven",
        "Kimberly",
        "Paul",
        "Emily",
        "Andrew",
        "Donna",
        "Joshua",
        "Michelle",
        "Kevin",
        "Dorothy",
        "Brian",
        "Carol",
        "George",
        "Amanda",
        "Timothy",
        "Melissa",
        "Ronald",
        "Deborah",
        "Edward",
        "Stephanie",
        "Jason",
        "Rebecca",
        "Jeffrey",
        "Sharon",
        "Ryan",
        "Laura",
        "Jacob",
        "Cynthia",
        "Gary",
        "Kathleen",
        "Nicholas",
        "Amy",
    ]

    last_names = [
        "Smith",
        "Johnson",
        "Williams",
        "Brown",
        "Jones",
        "Garcia",
        "Miller",
        "Davis",
        "Rodriguez",
        "Martinez",
        "Hernandez",
        "Lopez",
        "Gonzalez",
        "Wilson",
        "Anderson",
        "Thomas",
        "Taylor",
        "Moore",
        "Jackson",
        "Martin",
        "Lee",
        "Perez",
        "Thompson",
        "White",
        "Harris",
        "Sanchez",
        "Clark",
        "Ramirez",
        "Lewis",
        "Robinson",
        "Walker",
        "Young",
        "Allen",
        "King",
        "Wright",
        "Scott",
        "Torres",
        "Nguyen",
        "Hill",
        "Flores",
        "Green",
        "Adams",
        "Nelson",
        "Baker",
        "Hall",
        "Rivera",
        "Campbell",
        "Mitchell",
    ]

    dietary_options = [
        "vegetarian",
        "vegan",
        "gluten-free",
        "dairy-free",
        "nut-free",
        "halal",
        "kosher",
    ]
    categories = ["Appetizers", "Main Courses", "Sides", "Desserts", "Beverages"]

    customers = []

    for i in range(1, num_customers + 1):
        first_name = random.choice(first_names)
        last_name = random.choice(last_names)

        # Generate realistic order patterns
        total_orders = max(1, int(np.random.exponential(scale=15)))
        total_spent = round(total_orders * np.random.uniform(15, 50), 2)
        avg_order_value = round(total_spent / total_orders, 2)

        # Determine loyalty tier based on spending
        if total_spent >= 2000:
            loyalty_tier = "Platinum"
            loyalty_points = int(total_spent * 2)
        elif total_spent >= 1000:
            loyalty_tier = "Gold"
            loyalty_points = int(total_spent * 1.5)
        elif total_spent >= 500:
            loyalty_tier = "Silver"
            loyalty_points = int(total_spent * 1.2)
        else:
            loyalty_tier = "Bronze"
            loyalty_points = int(total_spent)

        # Random dietary preferences (30% have preferences)
        if random.random() < 0.3:
            num_prefs = random.randint(1, 3)
            dietary_prefs = random.sample(dietary_options, num_prefs)
        else:
            dietary_prefs = []

        created_at = datetime(2023, 1, 1) + timedelta(days=random.randint(0, 700))
        last_order = created_at + timedelta(
            days=random.randint(0, (datetime(2025, 12, 1) - created_at).days)
        )

        customers.append(
            {
                "customer_id": f"CUST_{i:06d}",
                "first_name": first_name,
                "last_name": last_name,
                "email": f"{first_name.lower()}.{last_name.lower()}{i}@email.com",
                "phone": f"+1-{random.randint(200, 999)}-{random.randint(100, 999)}-{random.randint(1000, 9999)}",
                "loyalty_tier": loyalty_tier,
                "loyalty_points": loyalty_points,
                "total_orders": total_orders,
                "total_spent": total_spent,
                "average_order_value": avg_order_value,
                "preferred_category": random.choice(categories),
                "dietary_preferences": json.dumps(dietary_prefs),
                "created_at": created_at.isoformat(),
                "last_order_date": last_order.isoformat(),
            }
        )

    df = pd.DataFrame(customers)
    df.to_csv(RAW_DIR / "customers.csv", index=False)
    print(f"Generated {len(df)} customers -> {RAW_DIR / 'customers.csv'}")
    return df


# ============================================================================
# TRANSACTIONS GENERATION
# ============================================================================


def generate_transactions(
    menu_df: pd.DataFrame,
    customers_df: pd.DataFrame,
    num_transactions: int = 100000,
    start_date: datetime = datetime(2024, 1, 1),
    end_date: datetime = datetime(2025, 12, 1),
) -> pd.DataFrame:
    """
    Generate synthetic POS transactions.

    Args:
        menu_df: Menu items DataFrame.
        customers_df: Customers DataFrame.
        num_transactions: Number of transactions to generate.
        start_date: Start date for transactions.
        end_date: End date for transactions.

    Returns:
        pd.DataFrame: Transactions with fields:
            - transaction_id: Unique identifier
            - order_id: Order identifier (groups items in same order)
            - customer_id: Customer reference (nullable for walk-ins)
            - item_id: Menu item reference
            - item_name: Item name (denormalized)
            - category: Item category
            - quantity: Number of items
            - unit_price: Price per item
            - total_price: quantity * unit_price
            - discount_amount: Applied discount
            - tax_amount: Calculated tax
            - timestamp: Transaction datetime
            - payment_method: Cash/Credit/Debit/Mobile
            - order_type: Dine-in/Takeout/Delivery
            - server_id: Server/employee reference
            - table_number: For dine-in orders

    Example:
        >>> df = generate_transactions(menu_df, customers_df, 10000)
        >>> df.head()
    """

    item_ids = menu_df["item_id"].tolist()
    item_data = menu_df.set_index("item_id").to_dict("index")
    customer_ids = customers_df["customer_id"].tolist()

    payment_methods = ["Credit", "Debit", "Cash", "Mobile"]
    payment_weights = [0.45, 0.25, 0.15, 0.15]

    order_types = ["Dine-in", "Takeout", "Delivery"]
    order_weights = [0.55, 0.25, 0.20]

    transactions = []

    # Generate orders (each order has 1-8 items)
    num_orders = num_transactions // 3  # Average 3 items per order

    date_range = (end_date - start_date).days

    for order_idx in range(num_orders):
        order_id = f"ORD_{order_idx + 1:08d}"

        # Customer (70% have customer ID, 30% walk-ins)
        customer_id = random.choice(customer_ids) if random.random() < 0.7 else None

        # Random timestamp with realistic hour distribution
        order_date = start_date + timedelta(days=random.randint(0, date_range))

        # Peak hours: 11-14 (lunch), 17-21 (dinner)
        hour_weights = (
            [0.01] * 6
            + [0.02, 0.03, 0.04, 0.05]
            + [0.08, 0.10, 0.12, 0.10]
            + [0.05, 0.04, 0.03, 0.08, 0.10, 0.12, 0.10, 0.08]
            + [0.04, 0.02]
        )
        hour = random.choices(range(24), weights=hour_weights)[0]
        minute = random.randint(0, 59)
        second = random.randint(0, 59)

        timestamp = order_date.replace(hour=hour, minute=minute, second=second)

        # Seasonality adjustments (more orders on weekends, holidays)
        day_of_week = timestamp.weekday()
        is_weekend = day_of_week >= 5

        # Order details
        order_type = random.choices(order_types, weights=order_weights)[0]
        payment_method = random.choices(payment_methods, weights=payment_weights)[0]
        server_id = f"EMP_{random.randint(1, 20):03d}" if order_type == "Dine-in" else None
        table_number = random.randint(1, 30) if order_type == "Dine-in" else None

        # Generate items for this order (realistic meal patterns)
        num_items = random.choices(
            [1, 2, 3, 4, 5, 6, 7, 8], weights=[0.10, 0.25, 0.30, 0.20, 0.08, 0.04, 0.02, 0.01]
        )[0]

        if is_weekend:
            num_items = min(8, num_items + 1)  # More items on weekends

        selected_items = random.sample(item_ids, min(num_items, len(item_ids)))

        for item_id in selected_items:
            item_info = item_data[item_id]
            quantity = random.choices([1, 2, 3, 4], weights=[0.7, 0.2, 0.08, 0.02])[0]
            unit_price = item_info["price"]
            total_price = round(unit_price * quantity, 2)

            # Random discount (10% of orders have discounts)
            discount_amount = 0.0
            if random.random() < 0.1:
                discount_pct = random.choice([0.05, 0.10, 0.15, 0.20])
                discount_amount = round(total_price * discount_pct, 2)

            # Tax calculation (8.5% tax rate)
            tax_amount = round((total_price - discount_amount) * 0.085, 2)

            transactions.append(
                {
                    "transaction_id": f"TXN_{len(transactions) + 1:010d}",
                    "order_id": order_id,
                    "customer_id": customer_id,
                    "item_id": item_id,
                    "item_name": item_info["name"],
                    "category": item_info["category"],
                    "quantity": quantity,
                    "unit_price": unit_price,
                    "total_price": total_price,
                    "discount_amount": discount_amount,
                    "tax_amount": tax_amount,
                    "timestamp": timestamp.isoformat(),
                    "payment_method": payment_method,
                    "order_type": order_type,
                    "server_id": server_id,
                    "table_number": table_number,
                }
            )

    df = pd.DataFrame(transactions)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df.to_csv(RAW_DIR / "transactions.csv", index=False)
    print(f"Generated {len(df)} transactions -> {RAW_DIR / 'transactions.csv'}")
    return df


# ============================================================================
# INVENTORY GENERATION
# ============================================================================


def generate_inventory(menu_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate synthetic inventory dataset.

    Args:
        menu_df: Menu items DataFrame for reference.

    Returns:
        pd.DataFrame: Inventory with fields:
            - inventory_id: Unique identifier
            - ingredient_name: Name of ingredient
            - category: Ingredient category
            - quantity_on_hand: Current quantity
            - unit: Unit of measurement
            - unit_cost: Cost per unit
            - reorder_level: Minimum quantity before reorder
            - reorder_quantity: Amount to order
            - supplier_id: Supplier reference
            - last_restocked: Last restock date
            - expiry_date: Expiration date (if applicable)
            - storage_location: Where item is stored

    Example:
        >>> df = generate_inventory(menu_df)
        >>> df.head()
    """

    ingredients = {
        "Proteins": [
            ("Ground Beef", "lb", 5.99, 50, 100, "Refrigerator A"),
            ("Chicken Breast", "lb", 4.99, 40, 80, "Refrigerator A"),
            ("Ribeye Steak", "lb", 15.99, 20, 40, "Refrigerator B"),
            ("Filet Mignon", "lb", 22.99, 15, 30, "Refrigerator B"),
            ("NY Strip Steak", "lb", 14.99, 20, 40, "Refrigerator B"),
            ("Sirloin Steak", "lb", 10.99, 25, 50, "Refrigerator B"),
            ("Salmon Fillet", "lb", 12.99, 25, 50, "Refrigerator C"),
            ("Cod Fillet", "lb", 8.99, 20, 40, "Refrigerator C"),
            ("Shrimp", "lb", 11.99, 30, 60, "Freezer A"),
            ("Lobster Tail", "each", 18.99, 10, 20, "Freezer A"),
            ("Crab Meat", "lb", 24.99, 8, 15, "Freezer A"),
            ("Bacon", "lb", 6.99, 35, 70, "Refrigerator A"),
            ("Turkey Breast", "lb", 5.99, 25, 50, "Refrigerator A"),
            ("Corned Beef", "lb", 9.99, 15, 30, "Refrigerator A"),
            ("Calamari", "lb", 8.99, 15, 30, "Freezer A"),
            ("Clams", "lb", 7.99, 10, 20, "Refrigerator C"),
        ],
        "Dairy": [
            ("Butter", "lb", 4.99, 30, 60, "Refrigerator D"),
            ("Heavy Cream", "qt", 3.99, 20, 40, "Refrigerator D"),
            ("Milk", "gal", 3.49, 15, 30, "Refrigerator D"),
            ("Cheddar Cheese", "lb", 5.99, 25, 50, "Refrigerator D"),
            ("Swiss Cheese", "lb", 6.99, 20, 40, "Refrigerator D"),
            ("Mozzarella", "lb", 5.49, 30, 60, "Refrigerator D"),
            ("Parmesan", "lb", 9.99, 15, 30, "Refrigerator D"),
            ("Feta Cheese", "lb", 7.99, 15, 30, "Refrigerator D"),
            ("Cream Cheese", "lb", 3.99, 20, 40, "Refrigerator D"),
            ("Gruyere Cheese", "lb", 12.99, 10, 20, "Refrigerator D"),
            ("Ricotta", "lb", 4.49, 15, 30, "Refrigerator D"),
            ("Eggs", "dozen", 3.99, 40, 80, "Refrigerator D"),
            ("Sour Cream", "pt", 2.49, 20, 40, "Refrigerator D"),
        ],
        "Produce": [
            ("Lettuce Romaine", "head", 1.99, 40, 80, "Walk-in Cooler"),
            ("Tomatoes", "lb", 2.49, 50, 100, "Walk-in Cooler"),
            ("Onions", "lb", 0.99, 60, 120, "Dry Storage"),
            ("Potatoes", "lb", 0.79, 100, 200, "Dry Storage"),
            ("Sweet Potatoes", "lb", 1.29, 40, 80, "Dry Storage"),
            ("Mushrooms", "lb", 3.99, 25, 50, "Walk-in Cooler"),
            ("Bell Peppers", "each", 0.99, 40, 80, "Walk-in Cooler"),
            ("Broccoli", "lb", 2.49, 30, 60, "Walk-in Cooler"),
            ("Asparagus", "bunch", 3.99, 20, 40, "Walk-in Cooler"),
            ("Spinach", "lb", 3.49, 25, 50, "Walk-in Cooler"),
            ("Carrots", "lb", 1.49, 40, 80, "Walk-in Cooler"),
            ("Corn", "ear", 0.49, 50, 100, "Walk-in Cooler"),
            ("Lemons", "each", 0.39, 60, 120, "Walk-in Cooler"),
            ("Limes", "each", 0.29, 60, 120, "Walk-in Cooler"),
            ("Avocado", "each", 1.49, 30, 60, "Walk-in Cooler"),
            ("Garlic", "head", 0.49, 50, 100, "Dry Storage"),
            ("Fresh Basil", "bunch", 2.49, 15, 30, "Walk-in Cooler"),
            ("Fresh Mint", "bunch", 2.49, 10, 20, "Walk-in Cooler"),
            ("Olives", "jar", 4.99, 20, 40, "Dry Storage"),
            ("Artichoke Hearts", "can", 3.99, 15, 30, "Dry Storage"),
        ],
        "Dry Goods": [
            ("All-Purpose Flour", "lb", 0.59, 80, 150, "Dry Storage"),
            ("Sugar", "lb", 0.69, 60, 120, "Dry Storage"),
            ("Olive Oil", "liter", 8.99, 20, 40, "Dry Storage"),
            ("Vegetable Oil", "gal", 5.99, 15, 30, "Dry Storage"),
            ("Spaghetti Pasta", "lb", 1.49, 50, 100, "Dry Storage"),
            ("Fettuccine Pasta", "lb", 1.49, 40, 80, "Dry Storage"),
            ("Linguine Pasta", "lb", 1.49, 40, 80, "Dry Storage"),
            ("Lasagna Noodles", "box", 2.99, 20, 40, "Dry Storage"),
            ("Rice", "lb", 1.29, 60, 120, "Dry Storage"),
            ("Breadcrumbs", "lb", 2.49, 30, 60, "Dry Storage"),
            ("Panko Breadcrumbs", "lb", 3.49, 25, 50, "Dry Storage"),
            ("Chicken Broth", "qt", 2.99, 30, 60, "Dry Storage"),
            ("Beef Broth", "qt", 3.49, 25, 50, "Dry Storage"),
            ("Canned Tomatoes", "can", 1.99, 50, 100, "Dry Storage"),
            ("Tomato Paste", "can", 0.99, 40, 80, "Dry Storage"),
            ("Marinara Sauce", "jar", 3.99, 30, 60, "Dry Storage"),
            ("Worcestershire Sauce", "bottle", 3.99, 15, 30, "Dry Storage"),
            ("Soy Sauce", "bottle", 4.99, 15, 30, "Dry Storage"),
            ("Tortilla Chips", "bag", 2.99, 25, 50, "Dry Storage"),
            ("Pita Bread", "pack", 2.99, 20, 40, "Dry Storage"),
            ("Hamburger Buns", "pack", 2.99, 40, 80, "Dry Storage"),
            ("Sourdough Bread", "loaf", 3.99, 20, 40, "Dry Storage"),
        ],
        "Beverages": [
            ("Coca-Cola Syrup", "box", 49.99, 8, 15, "Beverage Storage"),
            ("Sprite Syrup", "box", 49.99, 6, 12, "Beverage Storage"),
            ("Coffee Beans", "lb", 12.99, 20, 40, "Dry Storage"),
            ("Black Tea", "lb", 8.99, 10, 20, "Dry Storage"),
            ("Lemon Juice", "gal", 7.99, 10, 20, "Refrigerator D"),
            ("House Red Wine", "bottle", 8.99, 40, 80, "Wine Storage"),
            ("House White Wine", "bottle", 8.99, 40, 80, "Wine Storage"),
            ("Draft Beer Keg", "keg", 89.99, 5, 10, "Beverage Storage"),
            ("Tequila", "bottle", 29.99, 10, 20, "Bar Storage"),
            ("Rum", "bottle", 24.99, 10, 20, "Bar Storage"),
            ("Triple Sec", "bottle", 12.99, 8, 15, "Bar Storage"),
        ],
        "Frozen": [
            ("Vanilla Ice Cream", "gal", 12.99, 10, 20, "Freezer B"),
            ("Chocolate Ice Cream", "gal", 12.99, 8, 15, "Freezer B"),
            ("Frozen French Fries", "lb", 2.49, 100, 200, "Freezer C"),
            ("Frozen Sweet Potato Fries", "lb", 3.49, 50, 100, "Freezer C"),
            ("Frozen Onion Rings", "lb", 3.99, 40, 80, "Freezer C"),
            ("Frozen Pie Shells", "each", 2.99, 20, 40, "Freezer B"),
            ("Frozen Brownies", "dozen", 8.99, 10, 20, "Freezer B"),
        ],
    }

    inventory_items = []
    inv_id = 1
    suppliers = [f"SUP_{i:03d}" for i in range(1, 11)]

    for category, items in ingredients.items():
        for item in items:
            name, unit, unit_cost, reorder_level, reorder_qty, location = item

            # Random current quantity (between 50% and 200% of reorder level)
            quantity = int(reorder_level * np.random.uniform(0.5, 2.0))

            # Last restocked (within last 30 days)
            last_restocked = datetime.now() - timedelta(days=random.randint(1, 30))

            # Expiry date (depends on category)
            if category in ["Proteins", "Dairy", "Produce"]:
                expiry_days = random.randint(3, 14)
            elif category == "Frozen":
                expiry_days = random.randint(60, 180)
            else:
                expiry_days = random.randint(90, 365)

            expiry_date = last_restocked + timedelta(days=expiry_days)

            inventory_items.append(
                {
                    "inventory_id": f"INV_{inv_id:04d}",
                    "ingredient_name": name,
                    "category": category,
                    "quantity_on_hand": quantity,
                    "unit": unit,
                    "unit_cost": unit_cost,
                    "reorder_level": reorder_level,
                    "reorder_quantity": reorder_qty,
                    "supplier_id": random.choice(suppliers),
                    "last_restocked": last_restocked.date().isoformat(),
                    "expiry_date": (
                        expiry_date.date().isoformat() if category != "Dry Goods" else None
                    ),
                    "storage_location": location,
                }
            )
            inv_id += 1

    df = pd.DataFrame(inventory_items)
    df.to_csv(RAW_DIR / "inventory.csv", index=False)
    print(f"Generated {len(df)} inventory items -> {RAW_DIR / 'inventory.csv'}")
    return df


# ============================================================================
# CO-OCCURRENCE GENERATION
# ============================================================================


def generate_cooccurrence(transactions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate item-to-item co-occurrence counts from transactions.

    This computes how often items appear together in the same order,
    which is useful for market basket analysis and recommendations.

    Args:
        transactions_df: Transactions DataFrame.

    Returns:
        pd.DataFrame: Co-occurrence counts with fields:
            - item_id_1: First item
            - item_id_2: Second item
            - item_name_1: First item name
            - item_name_2: Second item name
            - cooccurrence_count: Number of times items appear together
            - support: Proportion of orders containing both items
            - confidence_1_to_2: P(item_2 | item_1)
            - confidence_2_to_1: P(item_1 | item_2)
            - lift: Lift score for the association

    Example:
        >>> df = generate_cooccurrence(transactions_df)
        >>> df.head()
    """

    # Group by order and get unique items per order
    order_items = (
        transactions_df.groupby("order_id")
        .agg({"item_id": lambda x: list(set(x)), "item_name": lambda x: list(set(x))})
        .reset_index()
    )

    # Build co-occurrence counts
    from collections import defaultdict

    cooccurrence_counts = defaultdict(int)
    item_counts = defaultdict(int)

    total_orders = len(order_items)

    for _, row in order_items.iterrows():
        items = row["item_id"]

        # Count individual items
        for item in items:
            item_counts[item] += 1

        # Count pairs
        for i, item1 in enumerate(items):
            for item2 in items[i + 1 :]:
                pair = tuple(sorted([item1, item2]))
                cooccurrence_counts[pair] += 1

    # Build item name mapping
    item_name_map = (
        transactions_df.drop_duplicates("item_id").set_index("item_id")["item_name"].to_dict()
    )

    # Create co-occurrence DataFrame
    cooccurrence_data = []

    for (item1, item2), count in cooccurrence_counts.items():
        if count >= 5:  # Only keep pairs that appear at least 5 times
            support = count / total_orders
            conf_1_to_2 = count / item_counts[item1] if item_counts[item1] > 0 else 0
            conf_2_to_1 = count / item_counts[item2] if item_counts[item2] > 0 else 0

            # Lift calculation
            expected = (item_counts[item1] / total_orders) * (item_counts[item2] / total_orders)
            lift = support / expected if expected > 0 else 0

            cooccurrence_data.append(
                {
                    "item_id_1": item1,
                    "item_id_2": item2,
                    "item_name_1": item_name_map.get(item1, "Unknown"),
                    "item_name_2": item_name_map.get(item2, "Unknown"),
                    "cooccurrence_count": count,
                    "support": round(support, 6),
                    "confidence_1_to_2": round(conf_1_to_2, 4),
                    "confidence_2_to_1": round(conf_2_to_1, 4),
                    "lift": round(lift, 4),
                }
            )

    df = pd.DataFrame(cooccurrence_data)
    df = df.sort_values("cooccurrence_count", ascending=False).reset_index(drop=True)
    df.to_csv(RAW_DIR / "cooccurrence.csv", index=False)
    print(f"Generated {len(df)} co-occurrence pairs -> {RAW_DIR / 'cooccurrence.csv'}")
    return df


# ============================================================================
# PROCESSED DATA GENERATION
# ============================================================================


def generate_daily_aggregates(transactions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate daily aggregated metrics for time-series forecasting.

    Args:
        transactions_df: Transactions DataFrame.

    Returns:
        pd.DataFrame: Daily aggregates with fields:
            - date: Date
            - total_revenue: Total revenue for the day
            - total_orders: Number of orders
            - total_items_sold: Total items sold
            - avg_order_value: Average order value
            - unique_customers: Unique customers
            - day_of_week: Day of week (0=Monday)
            - is_weekend: Boolean
            - month: Month number
            - year: Year

    Example:
        >>> df = generate_daily_aggregates(transactions_df)
        >>> df.head()
    """

    transactions_df = transactions_df.copy()
    transactions_df["date"] = pd.to_datetime(transactions_df["timestamp"]).dt.date

    daily = (
        transactions_df.groupby("date")
        .agg(
            {
                "total_price": "sum",
                "order_id": "nunique",
                "quantity": "sum",
                "customer_id": "nunique",
            }
        )
        .reset_index()
    )

    daily.columns = [
        "date",
        "total_revenue",
        "total_orders",
        "total_items_sold",
        "unique_customers",
    ]
    daily["avg_order_value"] = (daily["total_revenue"] / daily["total_orders"]).round(2)

    daily["date"] = pd.to_datetime(daily["date"])
    daily["day_of_week"] = daily["date"].dt.dayofweek
    daily["is_weekend"] = daily["day_of_week"] >= 5
    daily["month"] = daily["date"].dt.month
    daily["year"] = daily["date"].dt.year
    daily["date"] = daily["date"].dt.date

    daily = daily.sort_values("date").reset_index(drop=True)
    daily.to_csv(PROCESSED_DIR / "daily_aggregates.csv", index=False)
    print(f"Generated {len(daily)} daily aggregates -> {PROCESSED_DIR / 'daily_aggregates.csv'}")
    return daily


def generate_item_daily_sales(transactions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate item-level daily sales for demand forecasting.

    Args:
        transactions_df: Transactions DataFrame.

    Returns:
        pd.DataFrame: Item-daily sales with fields:
            - date: Date
            - item_id: Item identifier
            - item_name: Item name
            - category: Item category
            - quantity_sold: Total quantity sold
            - revenue: Total revenue
            - num_orders: Number of orders containing this item

    Example:
        >>> df = generate_item_daily_sales(transactions_df)
        >>> df.head()
    """

    transactions_df = transactions_df.copy()
    transactions_df["date"] = pd.to_datetime(transactions_df["timestamp"]).dt.date

    item_daily = (
        transactions_df.groupby(["date", "item_id", "item_name", "category"])
        .agg(
            {
                "quantity": "sum",
                "total_price": "sum",
                "order_id": "nunique",
            }
        )
        .reset_index()
    )

    item_daily.columns = [
        "date",
        "item_id",
        "item_name",
        "category",
        "quantity_sold",
        "revenue",
        "num_orders",
    ]
    item_daily = item_daily.sort_values(["date", "item_id"]).reset_index(drop=True)

    item_daily.to_csv(PROCESSED_DIR / "item_daily_sales.csv", index=False)
    print(
        f"Generated {len(item_daily)} item-daily sales records -> {PROCESSED_DIR / 'item_daily_sales.csv'}"
    )
    return item_daily


def generate_customer_features(
    transactions_df: pd.DataFrame, customers_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Generate customer features for recommendation and segmentation.

    Args:
        transactions_df: Transactions DataFrame.
        customers_df: Customers DataFrame.

    Returns:
        pd.DataFrame: Customer features for ML models.

    Example:
        >>> df = generate_customer_features(transactions_df, customers_df)
        >>> df.head()
    """

    # Filter to transactions with customer IDs
    cust_txn = transactions_df[transactions_df["customer_id"].notna()].copy()
    cust_txn["timestamp"] = pd.to_datetime(cust_txn["timestamp"])

    # Aggregate per customer
    customer_agg = cust_txn.groupby("customer_id").agg(
        {
            "order_id": "nunique",
            "total_price": "sum",
            "quantity": "sum",
            "timestamp": ["min", "max"],
            "category": lambda x: x.mode().iloc[0] if len(x) > 0 else "Unknown",
        }
    )

    customer_agg.columns = [
        "num_orders",
        "total_spent",
        "total_items",
        "first_order_date",
        "last_order_date",
        "favorite_category",
    ]
    customer_agg = customer_agg.reset_index()

    customer_agg["avg_order_value"] = (
        customer_agg["total_spent"] / customer_agg["num_orders"]
    ).round(2)
    customer_agg["avg_items_per_order"] = (
        customer_agg["total_items"] / customer_agg["num_orders"]
    ).round(2)
    customer_agg["days_since_first_order"] = (
        datetime.now() - customer_agg["first_order_date"]
    ).dt.days
    customer_agg["days_since_last_order"] = (
        datetime.now() - customer_agg["last_order_date"]
    ).dt.days
    customer_agg["order_frequency"] = (
        customer_agg["num_orders"] / (customer_agg["days_since_first_order"] + 1) * 30
    ).round(
        2
    )  # Orders per month

    # Merge with customer base info
    merged = customers_df[["customer_id", "loyalty_tier", "dietary_preferences"]].merge(
        customer_agg, on="customer_id", how="left"
    )

    merged.to_csv(PROCESSED_DIR / "customer_features.csv", index=False)
    print(
        f"Generated {len(merged)} customer feature records -> {PROCESSED_DIR / 'customer_features.csv'}"
    )
    return merged


# ============================================================================
# MAIN EXECUTION
# ============================================================================


def main():
    """Generate all datasets for the Restaurant SaaS platform."""

    print("=" * 60)
    print("RESTAURANT SAAS - DATASET GENERATION")
    print("=" * 60)
    print()

    # Step 1: Generate menu items
    print("[1/7] Generating menu items...")
    menu_df = generate_menu_items()
    print()

    # Step 2: Generate customers
    print("[2/7] Generating customers...")
    customers_df = generate_customers(num_customers=5000)
    print()

    # Step 3: Generate transactions
    print("[3/7] Generating transactions...")
    transactions_df = generate_transactions(menu_df, customers_df, num_transactions=100000)
    print()

    # Step 4: Generate inventory
    print("[4/7] Generating inventory...")
    inventory_df = generate_inventory(menu_df)
    print()

    # Step 5: Generate co-occurrence
    print("[5/7] Generating co-occurrence matrix...")
    cooccurrence_df = generate_cooccurrence(transactions_df)
    print()

    # Step 6: Generate processed data
    print("[6/7] Generating daily aggregates...")
    daily_df = generate_daily_aggregates(transactions_df)
    print()

    print("[7/7] Generating item-daily sales and customer features...")
    item_daily_df = generate_item_daily_sales(transactions_df)
    customer_features_df = generate_customer_features(transactions_df, customers_df)
    print()

    # Summary
    print("=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"\nRaw datasets saved to: {RAW_DIR}")
    print(f"  - menu_items.csv: {len(menu_df)} items")
    print(f"  - customers.csv: {len(customers_df)} customers")
    print(f"  - transactions.csv: {len(transactions_df)} transactions")
    print(f"  - inventory.csv: {len(inventory_df)} inventory items")
    print(f"  - cooccurrence.csv: {len(cooccurrence_df)} item pairs")
    print(f"\nProcessed datasets saved to: {PROCESSED_DIR}")
    print(f"  - daily_aggregates.csv: {len(daily_df)} days")
    print(f"  - item_daily_sales.csv: {len(item_daily_df)} item-day records")
    print(f"  - customer_features.csv: {len(customer_features_df)} customer profiles")
    print()


if __name__ == "__main__":
    main()
