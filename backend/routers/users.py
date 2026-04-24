from fastapi import APIRouter, Depends, HTTPException, Query
from backend.database import get_db
from backend.dependencies import get_current_user, require_role
from backend.schemas.user import UserUpdateRequest, AdminUserUpdateRequest
from backend.schemas.auth import ChangePasswordRequest
from backend.schemas.common import BulkDeleteRequest, BulkUpdateRequest
from backend.services import user_service
from backend.security import verify_password, hash_password

router = APIRouter(prefix="/api/users", tags=["users"])


@router.get("/me")
def get_profile(user=Depends(get_current_user)):
    return user


@router.put("/me")
def update_profile(req: UserUpdateRequest, user=Depends(get_current_user), conn=Depends(get_db)):
    updates = req.model_dump(exclude_none=True)
    result = user_service.update_user(conn, user["id"], updates)
    return result


@router.put("/me/password")
def change_password(req: ChangePasswordRequest, user=Depends(get_current_user), conn=Depends(get_db)):
    with conn.cursor() as cur:
        cur.execute("SELECT password_hash FROM users WHERE id = %s", (user["id"],))
        row = cur.fetchone()
    if not row or not verify_password(req.current_password, row[0]):
        raise HTTPException(status_code=400, detail="Current password incorrect")
    pw_hash = hash_password(req.new_password)
    with conn.cursor() as cur:
        cur.execute("UPDATE users SET password_hash = %s, updated_at = NOW() WHERE id = %s", (pw_hash, user["id"]))
        conn.commit()
    return {"message": "Password updated"}


@router.get("/")
def list_users(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    search: str = "",
    role: str = "",
    sort_by: str = "created_at",
    sort_dir: str = "desc",
    user=Depends(require_role("admin")),
    conn=Depends(get_db),
):
    return user_service.get_users(conn, page, page_size, search, role, sort_by, sort_dir)


@router.get("/{user_id}")
def get_user(user_id: int, user=Depends(require_role("admin")), conn=Depends(get_db)):
    result = user_service.get_user_by_id(conn, user_id)
    if not result:
        raise HTTPException(status_code=404, detail="User not found")
    return result


@router.put("/{user_id}")
def update_user(user_id: int, req: AdminUserUpdateRequest, user=Depends(require_role("admin")), conn=Depends(get_db)):
    updates = req.model_dump(exclude_none=True)
    result = user_service.update_user(conn, user_id, updates)
    if not result:
        raise HTTPException(status_code=404, detail="User not found")
    return result


@router.delete("/{user_id}")
def delete_user(user_id: int, user=Depends(require_role("admin")), conn=Depends(get_db)):
    if user_id == user["id"]:
        raise HTTPException(status_code=400, detail="Cannot delete yourself")
    ok = user_service.delete_user(conn, user_id)
    if not ok:
        raise HTTPException(status_code=404, detail="User not found")
    return {"message": "User deleted"}


@router.post("/bulk-delete")
def bulk_delete(req: BulkDeleteRequest, user=Depends(require_role("admin")), conn=Depends(get_db)):
    count = user_service.bulk_delete_users(conn, req.ids)
    return {"message": f"{count} users deleted"}


@router.put("/bulk-update")
def bulk_update(req: BulkUpdateRequest, user=Depends(require_role("admin")), conn=Depends(get_db)):
    count = user_service.bulk_update_users(conn, req.ids, req.updates)
    return {"message": f"{count} users updated"}
