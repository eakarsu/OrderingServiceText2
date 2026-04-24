from fastapi import APIRouter, Depends, HTTPException
from backend.database import get_db
from backend.dependencies import get_current_user, require_role
from backend.schemas.menu import CategoryCreateRequest, CategoryUpdateRequest
from backend.services import menu_service

router = APIRouter(prefix="/api/categories-mgmt", tags=["categories"])


@router.get("/")
def list_categories(user=Depends(get_current_user), conn=Depends(get_db)):
    return menu_service.get_categories(conn)


@router.get("/{cat_id}")
def get_category(cat_id: int, user=Depends(get_current_user), conn=Depends(get_db)):
    result = menu_service.get_category_by_id(conn, cat_id)
    if not result:
        raise HTTPException(status_code=404, detail="Category not found")
    return result


@router.post("/")
def create_category(req: CategoryCreateRequest, user=Depends(require_role("admin", "manager")), conn=Depends(get_db)):
    return menu_service.create_category(conn, req.name, req.description, req.image_url, req.sort_order, req.is_active)


@router.put("/{cat_id}")
def update_category(cat_id: int, req: CategoryUpdateRequest, user=Depends(require_role("admin", "manager")), conn=Depends(get_db)):
    updates = req.model_dump(exclude_none=True)
    result = menu_service.update_category(conn, cat_id, updates)
    if not result:
        raise HTTPException(status_code=404, detail="Category not found")
    return result


@router.delete("/{cat_id}")
def delete_category(cat_id: int, user=Depends(require_role("admin")), conn=Depends(get_db)):
    ok = menu_service.delete_category(conn, cat_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Category not found")
    return {"message": "Category deleted"}
