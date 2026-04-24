"""Seed customization rules from misc2/rules.txt"""

import re


RULES_DATA = {
    "BYO Breakfast": {
        "Bagel Options": {"min": 1, "max": 1, "options": [
            ("Cinnamon Raisin", 1.50, "Small"), ("Egg", 1.50, "Small"), ("Everything", 1.50, "Small"),
            ("Onion", 1.50, "Small"), ("Plain", 1.50, "Small"), ("Poppy", 1.50, "Small"),
            ("Salt", 1.50, "Small"), ("Sesame", 1.50, "Small"), ("Whole Wheat", 1.50, "Small"),
            ("Hero", 0.00, "Small"), ("Plain Flat", 2.00, "Small"), ("Everything Flat", 2.00, "Small"),
        ]},
        "Bagel Spreads": {"min": 1, "max": 6, "options": [
            ("Butter", 1.00, "Small"), ("Cream Cheese", 2.00, "Small"), ("Peanut Butter", 2.00, "Small"),
            ("Bacon", 4.00, "Small"), ("Lox", 6.00, "Small"), ("Plain", 0.00, "Small"),
            ("Toasted", 0.00, "Small"), ("Grape Jelly", 1.00, "Small"),
            ("Scallion Cream Cheese", 2.50, "Small"), ("Vegetable Cream Cheese", 2.50, "Small"),
            ("Green Olive Cream Cheese", 2.50, "Small"), ("Tomato", 0.75, "Small"),
            ("Onion", 0.75, "Small"),
        ]},
        "Breakfast Add-ons": {"min": 0, "max": 4, "options": [
            ("Avocado", 2.00, "Small"), ("Hash Brown", 1.50, "Small"), ("Home-Fries", 1.50, "Small"),
            ("Broccoli", 0.50, "Small"), ("Green Peppers", 0.50, "Small"), ("Hot Peppers", 0.50, "Small"),
            ("Mushrooms", 0.50, "Small"), ("Onions", 0.50, "Small"), ("Spinach", 0.50, "Small"),
        ]},
        "Breakfast Bread": {"min": 1, "max": 1, "options": [
            ("Roll", 0.00, "Medium"), ("Hero", 1.00, "Medium"), ("Plain Bagel", 0.50, "Medium"),
            ("Everything Bagel", 0.50, "Medium"), ("Whole Wheat Bagel", 0.50, "Medium"),
            ("Croissant", 2.00, "Medium"), ("English Muffin", 0.50, "Medium"),
            ("White Bread (sliced)", 0.00, "Medium"), ("Whole Wheat Bread (sliced)", 0.00, "Medium"),
            ("Rye Bread (sliced)", 0.00, "Medium"), ("White Wrap", 1.00, "Medium"),
            ("Whole Wheat Wrap", 1.00, "Medium"), ("Spinach Wrap", 1.00, "Medium"),
            ("Pita", 1.00, "Medium"),
        ]},
        "Breakfast Cheese": {"min": 0, "max": 2, "options": [
            ("American Cheese", 1.00, "Medium"), ("Cheddar", 1.00, "Medium"),
            ("Swiss", 1.00, "Medium"), ("Provolone", 1.00, "Medium"),
            ("Mozzarella", 1.00, "Medium"), ("Muenster", 1.00, "Medium"),
            ("Pepperjack", 1.00, "Medium"), ("Feta Cheese", 1.00, "Medium"),
            ("Fresh Mozzarella", 1.00, "Medium"), ("Blue Cheese Crumble", 1.00, "Medium"),
        ]},
        "Breakfast Egg Option": {"min": 0, "max": 1, "options": [
            ("Scrambled", 0.00, "Small"), ("Over Easy", 0.00, "Small"), ("Over Hard", 0.00, "Small"),
            ("Sunny Side Up", 0.00, "Small"), ("Egg Whites", 0.00, "Small"), ("Omelet", 0.00, "Small"),
            ("Fried Well Done", 0.00, "Small"),
        ]},
        "Breakfast Egg Quantity": {"min": 1, "max": 1, "options": [
            ("1 Egg", 0.75, "Small"), ("2 Eggs", 1.50, "Small"), ("3 Eggs", 2.25, "Small"),
            ("5 Eggs", 3.75, "Small"), ("6 Eggs", 4.50, "Small"),
        ]},
        "Breakfast Meat": {"min": 0, "max": 1, "options": [
            ("Bacon", 2.00, "Small"), ("Ham", 1.50, "Small"), ("Sausage", 1.50, "Small"),
            ("Turkey", 2.00, "Small"), ("Turkey Bacon", 2.00, "Small"),
            ("Steak", 2.00, "Small"), ("Grilled Chicken", 2.50, "Small"),
            ("Fried Chicken Cutlet", 2.50, "Small"), ("No Meat", 0.00, "Small"),
        ]},
    },
    "BYO Sandwiches": {
        "Bread": {"min": 1, "max": 1, "options": [
            ("Roll", 0.00, "Medium"), ("Hero", 1.00, "Medium"), ("Plain Bagel", 0.50, "Medium"),
            ("Everything Bagel", 0.50, "Medium"), ("White Bread (sliced)", 0.00, "Medium"),
            ("Whole Wheat Wrap", 1.00, "Medium"), ("Pita", 1.00, "Medium"),
        ]},
        "Cheese": {"min": 0, "max": 5, "options": [
            ("American Cheese", 1.00, "Medium"), ("Cheddar", 1.00, "Medium"),
            ("Swiss", 1.00, "Medium"), ("Provolone", 1.00, "Medium"),
            ("Mozzarella", 1.00, "Medium"), ("Pepperjack", 1.00, "Medium"),
            ("Fresh Mozzarella", 1.00, "Medium"), ("Feta Cheese", 1.00, "Medium"),
        ]},
        "Protein": {"min": 1, "max": 5, "options": [
            ("Bacon", 3.00, "Medium"), ("Ham", 2.00, "Medium"), ("Roast beef", 3.00, "Medium"),
            ("House Roast Turkey", 2.00, "Medium"), ("Grilled chicken", 2.00, "Medium"),
            ("Tuna salad", 2.00, "Medium"), ("Pastrami", 3.00, "Medium"),
            ("Salami", 3.00, "Medium"), ("No meat", 0.00, "Medium"),
        ]},
        "Toppings": {"min": 0, "max": 10, "options": [
            ("Lettuce", 1.00, "Medium"), ("Avocado", 1.50, "Medium"),
            ("Cucumber", 0.75, "Medium"), ("Onion", 0.75, "Medium"),
            ("Roasted Red Peppers", 1.00, "Medium"), ("Spinach", 1.00, "Medium"),
            ("Cole-Slaw", 1.50, "Medium"), ("Cherry Peppers", 0.75, "Medium"),
        ]},
    },
    "Chopped Salad": {
        "Salad Base": {"min": 1, "max": 1, "options": [
            ("Mixed Greens", 0.00, "Small"), ("Romaine", 0.00, "Small"), ("Spinach", 0.00, "Small"),
        ]},
        "Salad Add-ons": {"min": 0, "max": 10, "options": [
            ("Grilled chicken", 2.00, "Small"), ("Grilled Salmon", 3.00, "Small"),
            ("Feta cheese", 2.00, "Small"), ("Cheddar cheese", 2.00, "Small"),
            ("Croutons", 0.50, "Small"), ("Cucumber", 0.50, "Small"),
            ("Tomatoes", 0.50, "Small"), ("Black olives", 0.50, "Small"),
            ("Almonds", 2.00, "Small"), ("Walnuts", 2.00, "Small"),
        ]},
        "Salad Dressing": {"min": 0, "max": 3, "options": [
            ("Balsamic Vinaigrette", 0.00, "Small"), ("Ranch", 0.00, "Small"),
            ("Caesar", 0.00, "Small"), ("Italian", 0.00, "Small"),
            ("Oil & Vinegar", 0.00, "Small"), ("Russian", 0.00, "Small"),
            ("Raspberry Vinaigrette", 0.00, "Small"),
        ]},
    },
    "Coffee": {
        "Coffee Creamers": {"min": 0, "max": 1, "options": [
            ("Milk", 0.00, "Small"), ("Fat-Free Milk", 0.00, "Small"),
            ("Half and Half", 0.00, "Small"), ("French Vanilla", 0.00, "Small"),
            ("Irish Cream", 0.00, "Small"), ("Caramel", 0.00, "Small"),
            ("Hazelnut", 0.00, "Small"),
        ]},
    },
}


def seed_rules(conn):
    print("Seeding customization rules...")
    count = 0
    with conn.cursor() as cur:
        for category_name, rules in RULES_DATA.items():
            # Get menu items for this category
            cur.execute(
                """SELECT mi.id FROM menu_items mi
                   JOIN categories c ON c.id = mi.category_id
                   WHERE c.name = %s""",
                (category_name,),
            )
            item_ids = [r[0] for r in cur.fetchall()]

            for rule_name, rule_data in rules.items():
                # Check if rule exists
                cur.execute("SELECT id FROM customization_rules WHERE name = %s", (rule_name,))
                row = cur.fetchone()
                if row:
                    rule_id = row[0]
                else:
                    cur.execute(
                        """INSERT INTO customization_rules (name, min_selections, max_selections)
                           VALUES (%s, %s, %s) RETURNING id""",
                        (rule_name, rule_data["min"], rule_data["max"]),
                    )
                    rule_id = cur.fetchone()[0]
                    count += 1

                # Add options
                for opt_name, opt_price, opt_size in rule_data["options"]:
                    cur.execute(
                        "SELECT id FROM customization_options WHERE rule_id = %s AND name = %s",
                        (rule_id, opt_name),
                    )
                    if not cur.fetchone():
                        cur.execute(
                            """INSERT INTO customization_options (rule_id, name, price, size)
                               VALUES (%s, %s, %s, %s)""",
                            (rule_id, opt_name, opt_price, opt_size),
                        )

                # Link rule to menu items
                for item_id in item_ids:
                    cur.execute(
                        "SELECT id FROM menu_item_rules WHERE menu_item_id = %s AND rule_id = %s",
                        (item_id, rule_id),
                    )
                    if not cur.fetchone():
                        cur.execute(
                            "INSERT INTO menu_item_rules (menu_item_id, rule_id) VALUES (%s, %s)",
                            (item_id, rule_id),
                        )

        conn.commit()
    print(f"  Seeded {count} customization rules with options")
