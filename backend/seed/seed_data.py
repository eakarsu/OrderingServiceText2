"""Main seed runner. Run: python -m backend.seed.seed_data"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import psycopg2
from backend.config import settings
from backend.seed.seed_users import seed_users
from backend.seed.seed_categories import seed_categories
from backend.seed.seed_menu_items import seed_menu_items
from backend.seed.seed_orders import seed_orders
from backend.seed.seed_rules import seed_rules


def run():
    print("Starting seed...")
    conn = psycopg2.connect(
        host=settings.DB_HOST,
        database=settings.DB_DATABASE,
        user=settings.DB_USER,
        password=settings.DB_PASSWORD,
        port=settings.DB_PORT,
    )
    try:
        seed_users(conn)
        seed_categories(conn)
        seed_menu_items(conn)
        seed_rules(conn)
        seed_orders(conn)
        print("Seed complete!")
    finally:
        conn.close()


if __name__ == "__main__":
    run()
