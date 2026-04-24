from fastapi import APIRouter, Depends
from backend.database import get_db
from backend.dependencies import get_current_user
from backend.services import order_service

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


@router.get("/stats")
def dashboard_stats(user=Depends(get_current_user), conn=Depends(get_db)):
    stats = order_service.get_order_stats(conn)

    # Recent orders
    recent = order_service.get_orders(conn, page=1, page_size=5, sort_by="orderdate", sort_dir="desc")

    # Menu item counts
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM categories WHERE is_active = true")
        cat_count = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM menu_items WHERE is_available = true")
        item_count = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM users WHERE is_active = true")
        user_count = cur.fetchone()[0]

    return {
        **stats,
        "recent_orders": recent["items"],
        "active_categories": cat_count,
        "active_menu_items": item_count,
        "active_users": user_count,
    }
