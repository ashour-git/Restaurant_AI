"""Add image URLs to menu items."""
import sqlite3
import random

# Food image URLs mapped by category/keywords
FOOD_IMAGES = {
    # Soups
    "soup": [
        "https://images.unsplash.com/photo-1547592166-23ac45744acd?w=400&h=300&fit=crop",
        "https://images.unsplash.com/photo-1603105037880-880cd4edfb0d?w=400&h=300&fit=crop",
    ],
    # Salads
    "salad": [
        "https://images.unsplash.com/photo-1512621776951-a57141f2eefd?w=400&h=300&fit=crop",
        "https://images.unsplash.com/photo-1540420773420-3366772f4999?w=400&h=300&fit=crop",
        "https://images.unsplash.com/photo-1607532941433-304659e8198a?w=400&h=300&fit=crop",
    ],
    # Burgers
    "burger": [
        "https://images.unsplash.com/photo-1568901346375-23c9450c58cd?w=400&h=300&fit=crop",
        "https://images.unsplash.com/photo-1550317138-10000687a72b?w=400&h=300&fit=crop",
        "https://images.unsplash.com/photo-1553979459-d2229ba7433b?w=400&h=300&fit=crop",
    ],
    # Steaks
    "steak": [
        "https://images.unsplash.com/photo-1600891964092-4316c288032e?w=400&h=300&fit=crop",
        "https://images.unsplash.com/photo-1558030006-450675393462?w=400&h=300&fit=crop",
    ],
    "filet": [
        "https://images.unsplash.com/photo-1588168333986-5078d3ae3976?w=400&h=300&fit=crop",
    ],
    # Seafood
    "salmon": [
        "https://images.unsplash.com/photo-1519708227418-c8fd9a32b7a2?w=400&h=300&fit=crop",
        "https://images.unsplash.com/photo-1467003909585-2f8a72700288?w=400&h=300&fit=crop",
    ],
    "fish": [
        "https://images.unsplash.com/photo-1580476262798-bddd9f4b7369?w=400&h=300&fit=crop",
        "https://images.unsplash.com/photo-1510130387422-82bed34b37e9?w=400&h=300&fit=crop",
    ],
    "shrimp": [
        "https://images.unsplash.com/photo-1565680018434-b513d5e5fd47?w=400&h=300&fit=crop",
    ],
    "lobster": [
        "https://images.unsplash.com/photo-1553247407-23251ce81f59?w=400&h=300&fit=crop",
    ],
    # Pasta
    "pasta": [
        "https://images.unsplash.com/photo-1621996346565-e3dbc646d9a9?w=400&h=300&fit=crop",
        "https://images.unsplash.com/photo-1563379926898-05f4575a45d8?w=400&h=300&fit=crop",
    ],
    "lasagna": [
        "https://images.unsplash.com/photo-1574894709920-11b28e7367e3?w=400&h=300&fit=crop",
    ],
    "spaghetti": [
        "https://images.unsplash.com/photo-1551892374-ecf8754cf8b0?w=400&h=300&fit=crop",
    ],
    # Sandwiches
    "sandwich": [
        "https://images.unsplash.com/photo-1528735602780-2552fd46c7af?w=400&h=300&fit=crop",
        "https://images.unsplash.com/photo-1553909489-cd47e0907980?w=400&h=300&fit=crop",
    ],
    "blt": [
        "https://images.unsplash.com/photo-1619096252214-ef06c45683e3?w=400&h=300&fit=crop",
    ],
    "cheesesteak": [
        "https://images.unsplash.com/photo-1594212699903-ec8a3eca50f5?w=400&h=300&fit=crop",
    ],
    # Chicken
    "chicken": [
        "https://images.unsplash.com/photo-1598103442097-8b74394b95c6?w=400&h=300&fit=crop",
        "https://images.unsplash.com/photo-1604908176997-125f25cc6f3d?w=400&h=300&fit=crop",
    ],
    "wings": [
        "https://images.unsplash.com/photo-1608039829572-e4e8d1ec5de9?w=400&h=300&fit=crop",
        "https://images.unsplash.com/photo-1567620832903-9fc6debc209f?w=400&h=300&fit=crop",
    ],
    # Sides
    "fries": [
        "https://images.unsplash.com/photo-1573080496219-bb080dd4f877?w=400&h=300&fit=crop",
        "https://images.unsplash.com/photo-1518013431117-eb1465fa5752?w=400&h=300&fit=crop",
    ],
    "potato": [
        "https://images.unsplash.com/photo-1568569350062-ebfa3cb195df?w=400&h=300&fit=crop",
    ],
    "onion rings": [
        "https://images.unsplash.com/photo-1639024471283-03518883512d?w=400&h=300&fit=crop",
    ],
    "rice": [
        "https://images.unsplash.com/photo-1603133872878-684f208fb84b?w=400&h=300&fit=crop",
    ],
    # Appetizers
    "calamari": [
        "https://images.unsplash.com/photo-1599487488170-d11ec9c172f0?w=400&h=300&fit=crop",
    ],
    "mozzarella": [
        "https://images.unsplash.com/photo-1531749668029-2db88e27a9d9?w=400&h=300&fit=crop",
    ],
    # Desserts
    "cake": [
        "https://images.unsplash.com/photo-1578985545062-69928b1d9587?w=400&h=300&fit=crop",
        "https://images.unsplash.com/photo-1606890737304-57a1ca8a5b62?w=400&h=300&fit=crop",
    ],
    "chocolate": [
        "https://images.unsplash.com/photo-1606313564200-e75d5e30476c?w=400&h=300&fit=crop",
    ],
    "pie": [
        "https://images.unsplash.com/photo-1535920527002-b35e96722eb9?w=400&h=300&fit=crop",
        "https://images.unsplash.com/photo-1621743478914-cc8a86d7e7b5?w=400&h=300&fit=crop",
    ],
    "cheesecake": [
        "https://images.unsplash.com/photo-1524351199678-941a58a3df50?w=400&h=300&fit=crop",
    ],
    "sundae": [
        "https://images.unsplash.com/photo-1563805042-7684c019e1cb?w=400&h=300&fit=crop",
    ],
    "brownie": [
        "https://images.unsplash.com/photo-1564355808539-22fda35bed7e?w=400&h=300&fit=crop",
    ],
    # Beverages
    "coffee": [
        "https://images.unsplash.com/photo-1509042239860-f550ce710b93?w=400&h=300&fit=crop",
    ],
    "cappuccino": [
        "https://images.unsplash.com/photo-1572442388796-11668a67e53d?w=400&h=300&fit=crop",
    ],
    "tea": [
        "https://images.unsplash.com/photo-1556679343-c7306c1976bc?w=400&h=300&fit=crop",
    ],
    "lemonade": [
        "https://images.unsplash.com/photo-1621263764928-df1444c5e859?w=400&h=300&fit=crop",
    ],
    "cola": [
        "https://images.unsplash.com/photo-1629203851122-3726ecdf080e?w=400&h=300&fit=crop",
    ],
    "beer": [
        "https://images.unsplash.com/photo-1608270586620-248524c67de9?w=400&h=300&fit=crop",
    ],
    "wine": [
        "https://images.unsplash.com/photo-1510812431401-41d2bd2722f3?w=400&h=300&fit=crop",
    ],
    "margarita": [
        "https://images.unsplash.com/photo-1544145945-f90425340c7e?w=400&h=300&fit=crop",
    ],
    "mojito": [
        "https://images.unsplash.com/photo-1551024709-8f23befc6f87?w=400&h=300&fit=crop",
    ],
}

# Default food images for items that don't match any category
DEFAULT_IMAGES = [
    "https://images.unsplash.com/photo-1546069901-ba9599a7e63c?w=400&h=300&fit=crop",
    "https://images.unsplash.com/photo-1565299624946-b28f40a0ae38?w=400&h=300&fit=crop",
    "https://images.unsplash.com/photo-1540189549336-e6e99c3679fe?w=400&h=300&fit=crop",
    "https://images.unsplash.com/photo-1476224203421-9ac39bcb3327?w=400&h=300&fit=crop",
    "https://images.unsplash.com/photo-1504674900247-0877df9cc836?w=400&h=300&fit=crop",
]


def get_image_for_item(name: str, description: str = "") -> str:
    """Get an appropriate image URL for a menu item."""
    text = f"{name} {description}".lower()
    
    for keyword, images in FOOD_IMAGES.items():
        if keyword in text:
            return random.choice(images)
    
    return random.choice(DEFAULT_IMAGES)


def main():
    conn = sqlite3.connect("restaurant.db")
    cursor = conn.cursor()
    
    # Get all menu items
    cursor.execute("SELECT id, name, description FROM menu_items")
    items = cursor.fetchall()
    
    print(f"Updating {len(items)} menu items with images...")
    
    for item_id, name, description in items:
        image_url = get_image_for_item(name, description or "")
        cursor.execute(
            "UPDATE menu_items SET image_url = ? WHERE id = ?",
            (image_url, item_id)
        )
    
    conn.commit()
    conn.close()
    print("Done! All menu items now have images.")


if __name__ == "__main__":
    main()
