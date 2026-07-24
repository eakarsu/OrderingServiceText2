"""Canonical FastAPI entry point for OrderingServiceText.

Wires up:
  * backend/ admin routers (auth, users, orders, categories, menu_items,
    dashboard) — previously dark code with no entry point.
  * the fail-fast governed order workflow. Legacy direct-model/generated
    capability routes are quarantined rather than presented as order actions.
  * CORS for localhost dev (5173 Vite + 8000 served HTML).
  * Static FE at /static (vanilla JS forms — see static/index.html).
  * Root / serves the static index for one-click launch UX.

Launch:
    uvicorn app:app --reload --host 0.0.0.0 --port 8000

Required env:
    DB_*                — for /api/auth, /api/users, etc. (Postgres)
    JWT_SECRET          — for token signing
"""

import os
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from backend.routers.governed_orders import router as governed_orders_router
from backend.routers.application_ai import router as application_ai_router

# Ensure project root is on sys.path so `from orderChat import orderChat`
# resolves when launched from any cwd.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv()

app = FastAPI(title="OrderingServiceText API")
app.include_router(governed_orders_router)
app.include_router(application_ai_router)

# ----- CORS (localhost dev) -----
_cors_origins_env = os.getenv("CORS_ORIGINS", "")
_default_localhost = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
_cors_origins = (
    [o.strip() for o in _cors_origins_env.split(",") if o.strip()]
    if _cors_origins_env
    else _default_localhost
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----- Mount admin routers (best-effort: skip on import error) -----
_mount_errors: list = []


def _try_include(module_path: str, attr: str = "router") -> None:
    try:
        module = __import__(module_path, fromlist=[attr])
        app.include_router(getattr(module, attr))
    except Exception as exc:  # noqa: BLE001
        _mount_errors.append(f"{module_path}: {exc}")


for _modpath in (
    "backend.routers.auth",
    "backend.routers.users",
    "backend.routers.orders",
    "backend.routers.categories",
    "backend.routers.menu_items",
    "backend.routers.dashboard",
):
    _try_include(_modpath)


@app.api_route(
    "/api/{legacy_surface}/{legacy_path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
    include_in_schema=False,
)
def retired_generated_surface(legacy_surface: str, legacy_path: str):
    if legacy_surface not in {"ai", "ai-extras", "custom-views"}:
        raise HTTPException(status_code=404, detail="Not found")
    raise HTTPException(
        status_code=410,
        detail="Generated/direct-model order actions are retired; use the governed order API",
    )


# ----- Health & meta -----
@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "governed_orders": "mounted",
        "router_mount_errors": _mount_errors,
    }


@app.get("/api/meta")
def meta():
    """Return mounted route paths so the FE can introspect."""
    routes = []
    for r in app.routes:
        path = getattr(r, "path", None)
        methods = getattr(r, "methods", None)
        if path and methods:
            routes.append({"path": path, "methods": sorted(methods)})
    return {"routes": routes, "mount_errors": _mount_errors}


# ----- Static FE -----
_STATIC_DIR = ROOT / "static"
if _STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    @app.get("/")
    def index_page():
        idx = _STATIC_DIR / "index.html"
        if idx.exists():
            return FileResponse(str(idx))
        return JSONResponse({"message": "OrderingServiceText API"})
else:
    @app.get("/")
    def index_fallback():
        return {"message": "OrderingServiceText API (no static FE)"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
