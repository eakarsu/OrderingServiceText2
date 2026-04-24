from fastapi import APIRouter, Depends, HTTPException, Query
from backend.database import get_db
from backend.dependencies import get_current_user, require_role
from backend.schemas.menu import MenuItemCreateRequest, MenuItemUpdateRequest
from backend.schemas.common import BulkDeleteRequest, BulkUpdateRequest
from backend.services import menu_service

router = APIRouter(prefix="/api/menu-items", tags=["menu-items"])


@router.get("/")
def list_menu_items(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    search: str = "",
    category_id: int = None,
    available: bool = None,
    sort_by: str = "name",
    sort_dir: str = "asc",
    user=Depends(get_current_user),
    conn=Depends(get_db),
):
    return menu_service.get_menu_items(conn, page, page_size, search, category_id, available, sort_by, sort_dir)


@router.get("/{item_id}")
def get_menu_item(item_id: int, user=Depends(get_current_user), conn=Depends(get_db)):
    result = menu_service.get_menu_item_by_id(conn, item_id)
    if not result:
        raise HTTPException(status_code=404, detail="Menu item not found")
    return result


@router.post("/")
def create_menu_item(req: MenuItemCreateRequest, user=Depends(require_role("admin", "manager")), conn=Depends(get_db)):
    return menu_service.create_menu_item(conn, req.category_id, req.name, req.description, req.price, req.image_url, req.is_available)


@router.put("/{item_id}")
def update_menu_item(item_id: int, req: MenuItemUpdateRequest, user=Depends(require_role("admin", "manager")), conn=Depends(get_db)):
    updates = req.model_dump(exclude_none=True)
    result = menu_service.update_menu_item(conn, item_id, updates)
    if not result:
        raise HTTPException(status_code=404, detail="Menu item not found")
    return result


@router.delete("/{item_id}")
def delete_menu_item(item_id: int, user=Depends(require_role("admin")), conn=Depends(get_db)):
    ok = menu_service.delete_menu_item(conn, item_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Menu item not found")
    return {"message": "Menu item deleted"}


@router.post("/bulk-delete")
def bulk_delete(req: BulkDeleteRequest, user=Depends(require_role("admin")), conn=Depends(get_db)):
    count = menu_service.bulk_delete_menu_items(conn, req.ids)
    return {"message": f"{count} menu items deleted"}


@router.put("/bulk-update")
def bulk_update(req: BulkUpdateRequest, user=Depends(require_role("admin", "manager")), conn=Depends(get_db)):
    count = menu_service.bulk_update_menu_items(conn, req.ids, req.updates)
    return {"message": f"{count} menu items updated"}
