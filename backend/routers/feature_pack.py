"""Authenticated tasks, JSON export, and AI fallback over the app's own records."""
import json
import uuid

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from backend.database import get_db
from backend.dependencies import get_current_user
from backend.services import feature_pack as fp
from backend.services import order_service, menu_service

router = APIRouter(prefix="/api/feature", tags=["feature-pack"])

_ensured = False


def _ensure_tasks(conn):
    global _ensured
    if _ensured:
        return
    with conn.cursor() as cur:
        cur.execute("""CREATE TABLE IF NOT EXISTS ordering_workspace_tasks (
            id TEXT PRIMARY KEY, user_id INTEGER NOT NULL, title TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'open', priority TEXT NOT NULL DEFAULT 'medium',
            description TEXT NOT NULL DEFAULT '', created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW())""")
    conn.commit()
    _ensured = True


class TaskIn(BaseModel):
    title: str | None = None
    status: str | None = None
    priority: str | None = None
    description: str | None = None


@router.get("/tasks")
def list_tasks(q: str = "", status: str = "", user=Depends(get_current_user), conn=Depends(get_db)):
    _ensure_tasks(conn)
    clauses, params = ["user_id=%s"], [user["id"]]
    if status:
        params.append(status); clauses.append("status=%s")
    if q:
        params.append(f"%{q}%"); clauses.append("title ILIKE %s")
    with conn.cursor() as cur:
        cur.execute(f"SELECT id,title,status,priority,description,created_at,updated_at FROM ordering_workspace_tasks WHERE {' AND '.join(clauses)} ORDER BY created_at DESC LIMIT 200", params)
        rows = cur.fetchall()
    return {"tasks": [{"id": r[0], "title": r[1], "status": r[2], "priority": r[3], "description": r[4], "created_at": str(r[5]), "updated_at": str(r[6])} for r in rows]}


@router.post("/tasks", status_code=201)
def create_task(body: TaskIn, user=Depends(get_current_user), conn=Depends(get_db)):
    _ensure_tasks(conn)
    title = (body.title or "").strip()
    if not title:
        raise HTTPException(status_code=400, detail="title is required")
    tid = str(uuid.uuid4())
    with conn.cursor() as cur:
        cur.execute("INSERT INTO ordering_workspace_tasks(id,user_id,title,status,priority,description) VALUES(%s,%s,%s,%s,%s,%s) RETURNING id,title,status,priority,description",
                    (tid, user["id"], title, body.status or "open", body.priority or "medium", body.description or ""))
        row = cur.fetchone()
    conn.commit()
    return {"task": {"id": row[0], "title": row[1], "status": row[2], "priority": row[3], "description": row[4]}}


@router.patch("/tasks/{task_id}")
def update_task(task_id: str, body: TaskIn, user=Depends(get_current_user), conn=Depends(get_db)):
    _ensure_tasks(conn)
    fields, params = [], []
    for k in ("title", "status", "priority", "description"):
        v = getattr(body, k)
        if v is not None:
            params.append(v); fields.append(f"{k}=%s")
    if not fields:
        raise HTTPException(status_code=400, detail="No changes provided")
    params.extend([task_id, user["id"]])
    with conn.cursor() as cur:
        cur.execute(f"UPDATE ordering_workspace_tasks SET {','.join(fields)},updated_at=NOW() WHERE id=%s AND user_id=%s RETURNING id,title,status,priority,description", params)
        row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Task not found")
    conn.commit()
    return {"task": {"id": row[0], "title": row[1], "status": row[2], "priority": row[3], "description": row[4]}}


@router.get("/orders/export")
def export_orders(format: str = "csv", search: str = "", status: str = "", user=Depends(get_current_user), conn=Depends(get_db)):
    data = order_service.get_orders(conn, page=1, page_size=1000, search=search, status=status)
    rows = data["items"]
    if format == "json":
        return PlainTextResponse(fp.records_to_json(rows), media_type="application/json",
                                 headers={"Content-Disposition": "attachment; filename=orders.json"})
    return PlainTextResponse(fp.records_to_csv(rows, ["id", "phone_number", "status", "orderdate"]),
                             media_type="text/csv", headers={"Content-Disposition": "attachment; filename=orders.csv"})


@router.post("/ai/order-summary")
async def ai_order_summary(user=Depends(get_current_user), conn=Depends(get_db)):
    rows = order_service.get_orders(conn, page=1, page_size=200)["items"]
    if not fp.has_provider():
        return fp.fallback_order_summary(rows)
    return await fp.call_provider("order-summary", f"Summarise orders.\nRecords: {json.dumps(rows, default=str)[:6000]}")


@router.post("/ai/menu-insights")
async def ai_menu_insights(user=Depends(get_current_user), conn=Depends(get_db)):
    rows = menu_service.get_menu_items(conn, page=1, page_size=200).get("items", [])
    if not fp.has_provider():
        return fp.fallback_menu_insights(rows)
    return await fp.call_provider("menu-insights", f"Analyse menu.\nRecords: {json.dumps(rows, default=str)[:6000]}")


@router.post("/ai/fulfillment-risk")
async def ai_fulfillment_risk(user=Depends(get_current_user), conn=Depends(get_db)):
    rows = order_service.get_orders(conn, page=1, page_size=200)["items"]
    if not fp.has_provider():
        return fp.fallback_fulfillment_risk(rows)
    return await fp.call_provider("fulfillment-risk", f"Assess fulfilment risk.\nRecords: {json.dumps(rows, default=str)[:6000]}")
