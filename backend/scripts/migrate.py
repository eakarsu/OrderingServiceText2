"""Apply the tracked PostgreSQL migrations without seeding or resetting data."""

import os
from pathlib import Path

import psycopg2


def main() -> None:
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise SystemExit("DATABASE_URL is required")
    migrations = Path(__file__).resolve().parents[1] / "migrations"
    connection = psycopg2.connect(database_url)
    try:
        with connection.cursor() as cursor:
            for migration in sorted(migrations.glob("*.sql")):
                cursor.execute(migration.read_text(encoding="utf-8"))
                print(f"Applied {migration.name}")
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


if __name__ == "__main__":
    main()
