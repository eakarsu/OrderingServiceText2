import re

CATEGORIES = [
    {"name": "Acai Bowls", "description": "Fresh acai bowls with toppings"},
    {"name": "Bottled Drinks", "description": "Sodas, juices, energy drinks, and water"},
    {"name": "Breakfast Combos", "description": "Classic breakfast platters and combos"},
    {"name": "BYO Breakfast", "description": "Build your own breakfast with custom options"},
    {"name": "BYO Sandwiches", "description": "Build your own sandwich with fresh ingredients"},
    {"name": "Chips", "description": "Assorted chip varieties"},
    {"name": "Chopped Salad", "description": "Build your own chopped salad"},
    {"name": "Coffee", "description": "Hot coffee drinks in various sizes"},
    {"name": "Tea", "description": "Hot and iced tea options"},
    {"name": "Cold Sandwiches", "description": "Classic cold hero sandwiches"},
    {"name": "Desserts", "description": "Sweet treats and desserts"},
    {"name": "Grill Menu", "description": "Grilled sandwiches and wraps"},
    {"name": "Hot Sandwiches", "description": "Hot toasted hero sandwiches"},
    {"name": "Iced Tea and Lemonade", "description": "Fresh iced tea and lemonade"},
    {"name": "Muffins & Pastries", "description": "Fresh baked muffins and pastries"},
    {"name": "Omelets", "description": "Made-to-order omelets"},
    {"name": "Paninis", "description": "Pressed panini sandwiches"},
    {"name": "Salads", "description": "Fresh salads with dressings"},
    {"name": "Sliced Cold Cuts", "description": "Deli cold cuts and cheeses by weight"},
    {"name": "Snacks & Light Meals", "description": "Yogurt parfaits, overnight oats, and light bites"},
]


def seed_categories(conn):
    print("Seeding categories...")
    with conn.cursor() as cur:
        for i, cat in enumerate(CATEGORIES):
            cur.execute("SELECT id FROM categories WHERE name = %s", (cat["name"],))
            if cur.fetchone():
                continue
            cur.execute(
                """INSERT INTO categories (name, description, sort_order, is_active)
                   VALUES (%s, %s, %s, true)""",
                (cat["name"], cat["description"], i * 10),
            )
        conn.commit()
    print(f"  Seeded {len(CATEGORIES)} categories")
