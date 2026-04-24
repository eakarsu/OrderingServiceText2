import math


def get_categories(conn, include_count=True):
    with conn.cursor() as cur:
        if include_count:
            cur.execute("""
                SELECT c.id, c.name, c.description, c.image_url, c.sort_order, c.is_active, c.created_at,
                       COUNT(mi.id) as item_count
                FROM categories c
                LEFT JOIN menu_items mi ON mi.category_id = c.id
                GROUP BY c.id
                ORDER BY c.sort_order, c.name
            """)
        else:
            cur.execute("SELECT id, name, description, image_url, sort_order, is_active, created_at FROM categories ORDER BY sort_order, name")
        rows = cur.fetchall()

    return [
        {
            "id": r[0], "name": r[1], "description": r[2], "image_url": r[3],
            "sort_order": r[4], "is_active": r[5], "created_at": str(r[6]),
            "item_count": r[7] if include_count else 0,
        }
        for r in rows
    ]


def get_category_by_id(conn, cat_id):
    with conn.cursor() as cur:
        cur.execute(
            """SELECT c.id, c.name, c.description, c.image_url, c.sort_order, c.is_active, c.created_at,
                      COUNT(mi.id)
               FROM categories c LEFT JOIN menu_items mi ON mi.category_id = c.id
               WHERE c.id = %s GROUP BY c.id""",
            (cat_id,),
        )
        r = cur.fetchone()
    if not r:
        return None
    return {
        "id": r[0], "name": r[1], "description": r[2], "image_url": r[3],
        "sort_order": r[4], "is_active": r[5], "created_at": str(r[6]),
        "item_count": r[7],
    }


def create_category(conn, name, description=None, image_url=None, sort_order=0, is_active=True):
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO categories (name, description, image_url, sort_order, is_active)
               VALUES (%s, %s, %s, %s, %s) RETURNING id""",
            (name, description, image_url, sort_order, is_active),
        )
        cat_id = cur.fetchone()[0]
        conn.commit()
    return get_category_by_id(conn, cat_id)


def update_category(conn, cat_id, updates: dict):
    sets = []
    params = []
    for key, val in updates.items():
        if val is not None:
            sets.append(f"{key} = %s")
            params.append(val)
    if not sets:
        return get_category_by_id(conn, cat_id)
    sets.append("updated_at = NOW()")
    params.append(cat_id)
    with conn.cursor() as cur:
        cur.execute(f"UPDATE categories SET {', '.join(sets)} WHERE id = %s", params)
        conn.commit()
    return get_category_by_id(conn, cat_id)


def delete_category(conn, cat_id):
    with conn.cursor() as cur:
        cur.execute("DELETE FROM categories WHERE id = %s", (cat_id,))
        conn.commit()
        return cur.rowcount > 0


# Menu Items

def get_menu_items(conn, page=1, page_size=10, search="", category_id=None, available=None, sort_by="name", sort_dir="asc"):
    offset = (page - 1) * page_size
    wheres = []
    params = []

    if search:
        wheres.append("(mi.name ILIKE %s OR mi.description ILIKE %s)")
        params.extend([f"%{search}%"] * 2)
    if category_id:
        wheres.append("mi.category_id = %s")
        params.append(category_id)
    if available is not None:
        wheres.append("mi.is_available = %s")
        params.append(available)

    where_sql = "WHERE " + " AND ".join(wheres) if wheres else ""

    allowed_sort = {"name", "price", "created_at", "category_id"}
    if sort_by not in allowed_sort:
        sort_by = "name"
    sort_dir_sql = "DESC" if sort_dir.lower() == "desc" else "ASC"

    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM menu_items mi {where_sql}", params)
        total = cur.fetchone()[0]

        cur.execute(
            f"""SELECT mi.id, mi.category_id, c.name, mi.name, mi.description, mi.price, mi.image_url, mi.is_available, mi.created_at
                FROM menu_items mi
                LEFT JOIN categories c ON c.id = mi.category_id
                {where_sql}
                ORDER BY mi.{sort_by} {sort_dir_sql}
                LIMIT %s OFFSET %s""",
            params + [page_size, offset],
        )
        rows = cur.fetchall()

    items = [
        {
            "id": r[0], "category_id": r[1], "category_name": r[2] or "",
            "name": r[3], "description": r[4], "price": float(r[5]),
            "image_url": r[6], "is_available": r[7], "created_at": str(r[8]),
        }
        for r in rows
    ]
    return {
        "items": items,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": math.ceil(total / page_size) if total > 0 else 1,
    }


def get_menu_item_by_id(conn, item_id):
    with conn.cursor() as cur:
        cur.execute(
            """SELECT mi.id, mi.category_id, c.name, mi.name, mi.description, mi.price, mi.image_url, mi.is_available, mi.created_at
               FROM menu_items mi LEFT JOIN categories c ON c.id = mi.category_id
               WHERE mi.id = %s""",
            (item_id,),
        )
        r = cur.fetchone()
    if not r:
        return None
    return {
        "id": r[0], "category_id": r[1], "category_name": r[2] or "",
        "name": r[3], "description": r[4], "price": float(r[5]),
        "image_url": r[6], "is_available": r[7], "created_at": str(r[8]),
    }


def create_menu_item(conn, category_id, name, description=None, price=0.0, image_url=None, is_available=True):
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO menu_items (category_id, name, description, price, image_url, is_available)
               VALUES (%s, %s, %s, %s, %s, %s) RETURNING id""",
            (category_id, name, description, price, image_url, is_available),
        )
        item_id = cur.fetchone()[0]
        conn.commit()
    return get_menu_item_by_id(conn, item_id)


def update_menu_item(conn, item_id, updates: dict):
    sets = []
    params = []
    for key, val in updates.items():
        if val is not None:
            sets.append(f"{key} = %s")
            params.append(val)
    if not sets:
        return get_menu_item_by_id(conn, item_id)
    sets.append("updated_at = NOW()")
    params.append(item_id)
    with conn.cursor() as cur:
        cur.execute(f"UPDATE menu_items SET {', '.join(sets)} WHERE id = %s", params)
        conn.commit()
    return get_menu_item_by_id(conn, item_id)


def delete_menu_item(conn, item_id):
    with conn.cursor() as cur:
        cur.execute("DELETE FROM menu_items WHERE id = %s", (item_id,))
        conn.commit()
        return cur.rowcount > 0


def bulk_delete_menu_items(conn, ids):
    with conn.cursor() as cur:
        cur.execute("DELETE FROM menu_items WHERE id = ANY(%s)", (ids,))
        conn.commit()
        return cur.rowcount


def bulk_update_menu_items(conn, ids, updates: dict):
    sets = []
    params = []
    for key, val in updates.items():
        if val is not None:
            sets.append(f"{key} = %s")
            params.append(val)
    if not sets:
        return 0
    sets.append("updated_at = NOW()")
    params.append(ids)
    with conn.cursor() as cur:
        cur.execute(f"UPDATE menu_items SET {', '.join(sets)} WHERE id = ANY(%s)", params)
        conn.commit()
        return cur.rowcount
