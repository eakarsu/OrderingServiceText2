import math


def get_users(conn, page=1, page_size=10, search="", role="", sort_by="created_at", sort_dir="desc"):
    offset = (page - 1) * page_size
    where_clauses = []
    params = []

    if search:
        where_clauses.append("(first_name ILIKE %s OR last_name ILIKE %s OR email ILIKE %s)")
        params.extend([f"%{search}%"] * 3)
    if role:
        where_clauses.append("role = %s")
        params.append(role)

    where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

    allowed_sort = {"created_at", "email", "first_name", "last_name", "role"}
    if sort_by not in allowed_sort:
        sort_by = "created_at"
    sort_dir_sql = "DESC" if sort_dir.lower() == "desc" else "ASC"

    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM users {where_sql}", params)
        total = cur.fetchone()[0]

        cur.execute(
            f"""SELECT id, email, first_name, last_name, role, phone, is_active, is_verified, created_at
                FROM users {where_sql}
                ORDER BY {sort_by} {sort_dir_sql}
                LIMIT %s OFFSET %s""",
            params + [page_size, offset],
        )
        rows = cur.fetchall()

    users = [
        {
            "id": r[0], "email": r[1], "first_name": r[2], "last_name": r[3],
            "role": r[4], "phone": r[5], "is_active": r[6], "is_verified": r[7],
            "created_at": str(r[8]),
        }
        for r in rows
    ]
    return {
        "items": users,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": math.ceil(total / page_size) if total > 0 else 1,
    }


def get_user_by_id(conn, user_id):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, email, first_name, last_name, role, phone, is_active, is_verified, created_at FROM users WHERE id = %s",
            (user_id,),
        )
        r = cur.fetchone()
    if not r:
        return None
    return {
        "id": r[0], "email": r[1], "first_name": r[2], "last_name": r[3],
        "role": r[4], "phone": r[5], "is_active": r[6], "is_verified": r[7],
        "created_at": str(r[8]),
    }


def update_user(conn, user_id, updates: dict):
    set_parts = []
    params = []
    for key, val in updates.items():
        if val is not None:
            set_parts.append(f"{key} = %s")
            params.append(val)
    if not set_parts:
        return get_user_by_id(conn, user_id)
    set_parts.append("updated_at = NOW()")
    params.append(user_id)
    with conn.cursor() as cur:
        cur.execute(
            f"UPDATE users SET {', '.join(set_parts)} WHERE id = %s", params
        )
        conn.commit()
    return get_user_by_id(conn, user_id)


def delete_user(conn, user_id):
    with conn.cursor() as cur:
        cur.execute("DELETE FROM users WHERE id = %s", (user_id,))
        conn.commit()
        return cur.rowcount > 0


def bulk_delete_users(conn, ids):
    with conn.cursor() as cur:
        cur.execute("DELETE FROM users WHERE id = ANY(%s)", (ids,))
        conn.commit()
        return cur.rowcount


def bulk_update_users(conn, ids, updates: dict):
    set_parts = []
    params = []
    for key, val in updates.items():
        if val is not None:
            set_parts.append(f"{key} = %s")
            params.append(val)
    if not set_parts:
        return 0
    set_parts.append("updated_at = NOW()")
    params.append(ids)
    with conn.cursor() as cur:
        cur.execute(
            f"UPDATE users SET {', '.join(set_parts)} WHERE id = ANY(%s)", params
        )
        conn.commit()
        return cur.rowcount
