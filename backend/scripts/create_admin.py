"""Provision one explicitly configured administrator for a controlled environment."""

import os

import psycopg2
from passlib.context import CryptContext


def main() -> None:
    if os.environ.get("NODE_ENV") != "test" and os.environ.get("BOOTSTRAP_ACKNOWLEDGEMENT") != "create-initial-admin":
        raise SystemExit("Refusing admin provisioning without explicit acknowledgement")
    database_url = os.environ.get("DATABASE_URL")
    email = os.environ.get("ADMIN_EMAIL")
    password = os.environ.get("ADMIN_PASSWORD")
    if not database_url or not email or not password or len(password) < 12:
        raise SystemExit("DATABASE_URL and explicit strong ADMIN_EMAIL/ADMIN_PASSWORD are required")
    password_hash = CryptContext(schemes=["bcrypt"], deprecated="auto").hash(password)
    connection = psycopg2.connect(database_url)
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO users(email,password_hash,first_name,last_name,role,is_active,is_verified)
                VALUES (%s,%s,'Runtime','Acceptance','admin',true,true)
                ON CONFLICT(email) DO UPDATE SET
                  password_hash=EXCLUDED.password_hash,is_active=true,is_verified=true
                """,
                (email, password_hash),
            )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


if __name__ == "__main__":
    main()
