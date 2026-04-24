import json
import random
from datetime import datetime, timedelta

STATUSES = ["pending", "confirmed", "preparing", "ready", "delivered", "cancelled"]
PHONES = ["+15551234567", "+15559876543", "+15551112222", "+15553334444", "+15555556666"]

SAMPLE_ITEMS = [
    {"item": "Acai Bowl", "size": "Regular", "quantity": 1, "custom": "", "price": "$12.97"},
    {"item": "Turkey Club Hero", "size": "Regular", "quantity": 1, "custom": "", "price": "$17.95"},
    {"item": "Cappuccino, Large", "size": "Large", "quantity": 2, "custom": "", "price": "$2.76"},
    {"item": "Hungry Man", "size": "Regular", "quantity": 1, "custom": "", "price": "$12.95"},
    {"item": "Philly Cheese Steak", "size": "Regular", "quantity": 1, "custom": "no onions", "price": "$14.24"},
    {"item": "California Panini", "size": "Regular", "quantity": 1, "custom": "", "price": "$15.95"},
    {"item": "Greek Salad", "size": "Regular", "quantity": 1, "custom": "dressing on side", "price": "$15.95"},
    {"item": "Coke 20oz soda", "size": "Regular", "quantity": 1, "custom": "", "price": "$3.59"},
    {"item": "Chocolate Chip Cookies", "size": "Regular", "quantity": 3, "custom": "", "price": "$2.29"},
    {"item": "Blueberry Muffin", "size": "Regular", "quantity": 1, "custom": "", "price": "$3.59"},
    {"item": "Chef Salad", "size": "Regular", "quantity": 1, "custom": "", "price": "$15.95"},
    {"item": "Italian Hero", "size": "Regular", "quantity": 1, "custom": "", "price": "$17.95"},
    {"item": "Western Omelet", "size": "Regular", "quantity": 1, "custom": "", "price": "$10.32"},
    {"item": "Falafel Wrap", "size": "Regular", "quantity": 1, "custom": "extra tahini", "price": "$11.64"},
    {"item": "Beef gyro", "size": "Regular", "quantity": 2, "custom": "", "price": "$12.94"},
]


def seed_orders(conn):
    print("Seeding orders...")
    count = 0
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM orders")
        existing = cur.fetchone()[0]
        if existing >= 20:
            print(f"  Already have {existing} orders, skipping")
            return

        for i in range(25):
            phone = random.choice(PHONES)
            num_items = random.randint(1, 4)
            items = random.sample(SAMPLE_ITEMS, min(num_items, len(SAMPLE_ITEMS)))
            total = sum(
                float(it["price"].replace("$", "")) * it["quantity"]
                for it in items
            )
            order_data = {
                "message_type": "order",
                "phone_number": phone,
                "menu_items_ordered": items,
                "pickup_or_delivery": random.choice(["pickup", "delivery"]),
                "payment_type": random.choice(["cash", "credit", "venmo"]),
                "address": "123 Main St" if random.random() > 0.5 else "",
                "total_price": f"${total:.2f}",
            }
            status = random.choice(STATUSES)
            days_ago = random.randint(0, 30)
            order_date = datetime.now() - timedelta(days=days_ago, hours=random.randint(0, 23))

            cur.execute(
                """INSERT INTO orders (phone_number, order_data, status, orderdate)
                   VALUES (%s, %s, %s, %s)""",
                (phone, json.dumps(order_data), status, order_date),
            )
            count += 1
        conn.commit()
    print(f"  Seeded {count} orders")
