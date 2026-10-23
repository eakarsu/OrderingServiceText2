import os

from backend.security import hash_password

USERS = [
    # Admins
    {"email": "admin@restaurant.com", "password": "", "first_name": "John", "last_name": "Admin", "role": "admin", "phone": "+15551000001"},
    {"email": "sarah.admin@restaurant.com", "password": "", "first_name": "Sarah", "last_name": "Chen", "role": "admin", "phone": "+15551000002"},
    {"email": "mike.admin@restaurant.com", "password": "", "first_name": "Mike", "last_name": "Johnson", "role": "admin", "phone": "+15551000003"},
    # Managers
    {"email": "emily.mgr@restaurant.com", "password": "", "first_name": "Emily", "last_name": "Davis", "role": "manager", "phone": "+15552000001"},
    {"email": "james.mgr@restaurant.com", "password": "", "first_name": "James", "last_name": "Wilson", "role": "manager", "phone": "+15552000002"},
    {"email": "lisa.mgr@restaurant.com", "password": "", "first_name": "Lisa", "last_name": "Martinez", "role": "manager", "phone": "+15552000003"},
    {"email": "david.mgr@restaurant.com", "password": "", "first_name": "David", "last_name": "Brown", "role": "manager", "phone": "+15552000004"},
    # Staff
    {"email": "alex.staff@restaurant.com", "password": "", "first_name": "Alex", "last_name": "Turner", "role": "staff", "phone": "+15553000001"},
    {"email": "maria.staff@restaurant.com", "password": "", "first_name": "Maria", "last_name": "Garcia", "role": "staff", "phone": "+15553000002"},
    {"email": "tom.staff@restaurant.com", "password": "", "first_name": "Tom", "last_name": "Anderson", "role": "staff", "phone": "+15553000003"},
    {"email": "nina.staff@restaurant.com", "password": "", "first_name": "Nina", "last_name": "Patel", "role": "staff", "phone": "+15553000004"},
    {"email": "kevin.staff@restaurant.com", "password": "", "first_name": "Kevin", "last_name": "Lee", "role": "staff", "phone": "+15553000005"},
    {"email": "rachel.staff@restaurant.com", "password": "", "first_name": "Rachel", "last_name": "Kim", "role": "staff", "phone": "+15553000006"},
    {"email": "carlos.staff@restaurant.com", "password": "", "first_name": "Carlos", "last_name": "Rodriguez", "role": "staff", "phone": "+15553000007"},
    {"email": "sophie.staff@restaurant.com", "password": "", "first_name": "Sophie", "last_name": "Taylor", "role": "staff", "phone": "+15553000008"},
    {"email": "brian.staff@restaurant.com", "password": "", "first_name": "Brian", "last_name": "White", "role": "staff", "phone": "+15553000009"},
]


def seed_users(conn):
    seed_password = os.getenv("SEED_USER_PASSWORD", "")
    if len(seed_password) < 12:
        raise RuntimeError("SEED_USER_PASSWORD must be set to at least 12 characters before seeding")
    print("Seeding users...")
    with conn.cursor() as cur:
        for u in USERS:
            cur.execute("SELECT id FROM users WHERE email = %s", (u["email"],))
            if cur.fetchone():
                continue
            cur.execute(
                """INSERT INTO users (email, password_hash, first_name, last_name, role, phone, is_active, is_verified)
                   VALUES (%s, %s, %s, %s, %s, %s, true, true)""",
                (u["email"], hash_password(seed_password), u["first_name"], u["last_name"], u["role"], u["phone"]),
            )
        conn.commit()
    print(f"  Seeded {len(USERS)} users")
