import json
import math


def get_orders(conn, page=1, page_size=10, search="", status="", sort_by="orderdate", sort_dir="desc", date_from="", date_to=""):
    offset = (page - 1) * page_size
    where_clauses = []
    params = []

    if search:
        where_clauses.append("(phone_number ILIKE %s OR order_data::text ILIKE %s)")
        params.extend([f"%{search}%"] * 2)
    if status:
        where_clauses.append("status = %s")
        params.append(status)
    if date_from:
        where_clauses.append("orderdate >= %s")
        params.append(date_from)
    if date_to:
        where_clauses.append("orderdate <= %s")
        params.append(date_to)

    where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

    allowed_sort = {"orderdate", "phone_number", "status", "id"}
    if sort_by not in allowed_sort:
        sort_by = "orderdate"
    sort_dir_sql = "DESC" if sort_dir.lower() == "desc" else "ASC"

    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM orders {where_sql}", params)
        total = cur.fetchone()[0]

        cur.execute(
            f"""SELECT id, phone_number, order_data, COALESCE(status, 'pending'), orderdate, updated_at
                FROM orders {where_sql}
                ORDER BY {sort_by} {sort_dir_sql}
                LIMIT %s OFFSET %s""",
            params + [page_size, offset],
        )
        rows = cur.fetchall()

    orders = []
    for r in rows:
        od = r[2] if isinstance(r[2], dict) else json.loads(r[2]) if r[2] else {}
        orders.append({
            "id": r[0], "phone_number": r[1], "order_data": od,
            "status": r[3] or "pending", "orderdate": str(r[4]),
            "updated_at": str(r[5]) if r[5] else None,
        })

    return {
        "items": orders,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": math.ceil(total / page_size) if total > 0 else 1,
    }


def get_order_by_id(conn, order_id):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, phone_number, order_data, COALESCE(status, 'pending'), orderdate, updated_at FROM orders WHERE id = %s",
            (order_id,),
        )
        r = cur.fetchone()
    if not r:
        return None
    od = r[2] if isinstance(r[2], dict) else json.loads(r[2]) if r[2] else {}
    return {
        "id": r[0], "phone_number": r[1], "order_data": od,
        "status": r[3] or "pending", "orderdate": str(r[4]),
        "updated_at": str(r[5]) if r[5] else None,
    }


def update_order(conn, order_id, status=None, order_data=None):
    sets = []
    params = []
    if status:
        sets.append("status = %s")
        params.append(status)
    if order_data:
        sets.append("order_data = %s")
        params.append(json.dumps(order_data))
    if not sets:
        return get_order_by_id(conn, order_id)
    sets.append("updated_at = NOW()")
    params.append(order_id)
    with conn.cursor() as cur:
        cur.execute(f"UPDATE orders SET {', '.join(sets)} WHERE id = %s", params)
        conn.commit()
    return get_order_by_id(conn, order_id)


def delete_order(conn, order_id):
    with conn.cursor() as cur:
        cur.execute("DELETE FROM orders WHERE id = %s", (order_id,))
        conn.commit()
        return cur.rowcount > 0


def bulk_delete_orders(conn, ids):
    with conn.cursor() as cur:
        cur.execute("DELETE FROM orders WHERE id = ANY(%s)", (ids,))
        conn.commit()
        return cur.rowcount


def bulk_update_orders(conn, ids, updates: dict):
    sets = []
    params = []
    if "status" in updates and updates["status"]:
        sets.append("status = %s")
        params.append(updates["status"])
    if not sets:
        return 0
    sets.append("updated_at = NOW()")
    params.append(ids)
    with conn.cursor() as cur:
        cur.execute(f"UPDATE orders SET {', '.join(sets)} WHERE id = ANY(%s)", params)
        conn.commit()
        return cur.rowcount


def get_order_stats(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM orders")
        total = cur.fetchone()[0]

        cur.execute("""
            SELECT COALESCE(status, 'pending'), COUNT(*)
            FROM orders GROUP BY COALESCE(status, 'pending')
        """)
        by_status = dict(cur.fetchall())

        cur.execute("""
            SELECT COALESCE(
                SUM((order_data->>'total_price')::numeric), 0
            ) FROM orders
            WHERE order_data->>'total_price' IS NOT NULL
        """)
        revenue_raw = cur.fetchone()[0]
        revenue = float(revenue_raw) if revenue_raw else 0.0

        cur.execute("SELECT COUNT(*) FROM orders WHERE orderdate >= NOW() - INTERVAL '1 day'")
        today = cur.fetchone()[0]

    return {
        "total_orders": total,
        "orders_today": today,
        "revenue": revenue,
        "by_status": by_status,
    }
