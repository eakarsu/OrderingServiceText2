from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse, Response
from backend.database import get_db
from backend.dependencies import get_current_user, require_role
from backend.schemas.order import OrderUpdateRequest
from backend.schemas.common import BulkDeleteRequest, BulkUpdateRequest
from backend.services import order_service, export_service
import io

router = APIRouter(prefix="/api/orders", tags=["orders"])


@router.get("/stats")
def order_stats(user=Depends(get_current_user), conn=Depends(get_db)):
    return order_service.get_order_stats(conn)


@router.get("/export/csv")
def export_csv(
    search: str = "",
    status: str = "",
    date_from: str = "",
    date_to: str = "",
    user=Depends(require_role("admin", "manager")),
    conn=Depends(get_db),
):
    data = order_service.get_orders(conn, page=1, page_size=10000, search=search, status=status, date_from=date_from, date_to=date_to)
    csv_content = export_service.export_orders_csv(data["items"])
    return StreamingResponse(
        io.StringIO(csv_content),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=orders.csv"},
    )


@router.get("/export/pdf")
def export_pdf(
    search: str = "",
    status: str = "",
    date_from: str = "",
    date_to: str = "",
    user=Depends(require_role("admin", "manager")),
    conn=Depends(get_db),
):
    data = order_service.get_orders(conn, page=1, page_size=10000, search=search, status=status, date_from=date_from, date_to=date_to)
    pdf_bytes = export_service.export_orders_pdf(data["items"])
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=orders.pdf"},
    )


@router.get("/")
def list_orders(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    search: str = "",
    status: str = "",
    sort_by: str = "orderdate",
    sort_dir: str = "desc",
    date_from: str = "",
    date_to: str = "",
    user=Depends(get_current_user),
    conn=Depends(get_db),
):
    return order_service.get_orders(conn, page, page_size, search, status, sort_by, sort_dir, date_from, date_to)


@router.get("/{order_id}")
def get_order(order_id: int, user=Depends(get_current_user), conn=Depends(get_db)):
    result = order_service.get_order_by_id(conn, order_id)
    if not result:
        raise HTTPException(status_code=404, detail="Order not found")
    return result


@router.put("/{order_id}")
def update_order(order_id: int, req: OrderUpdateRequest, user=Depends(require_role("admin", "manager")), conn=Depends(get_db)):
    result = order_service.update_order(conn, order_id, req.status, req.order_data)
    if not result:
        raise HTTPException(status_code=404, detail="Order not found")
    return result


@router.delete("/{order_id}")
def delete_order(order_id: int, user=Depends(require_role("admin")), conn=Depends(get_db)):
    ok = order_service.delete_order(conn, order_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Order not found")
    return {"message": "Order deleted"}


@router.post("/bulk-delete")
def bulk_delete(req: BulkDeleteRequest, user=Depends(require_role("admin")), conn=Depends(get_db)):
    count = order_service.bulk_delete_orders(conn, req.ids)
    return {"message": f"{count} orders deleted"}


@router.put("/bulk-update")
def bulk_update(req: BulkUpdateRequest, user=Depends(require_role("admin", "manager")), conn=Depends(get_db)):
    count = order_service.bulk_update_orders(conn, req.ids, req.updates)
    return {"message": f"{count} orders updated"}
