"""Canonical FastAPI entry point for OrderingServiceText.

Wires up:
  * backend/ admin routers (auth, users, orders, categories, menu_items,
    dashboard) — previously dark code with no entry point.
  * NEW backend.routers.ai — wraps standalone AI scripts (orderChat,
    menuIndexer, orderProcessor) as POST /api/ai/* endpoints. Returns 503
    when OPENROUTER_API_KEY is missing.
  * CORS for localhost dev (5173 Vite + 8000 served HTML).
  * Static FE at /static (vanilla JS forms — see static/index.html).
  * Root / serves the static index for one-click launch UX.

Launch:
    uvicorn app:app --reload --host 0.0.0.0 --port 8000

Required env:
    OPENROUTER_API_KEY  — for /api/ai/* (otherwise endpoints return 503)
    DB_*                — for /api/auth, /api/users, etc. (Postgres)
    JWT_SECRET          — for token signing
"""

import os
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

# Ensure project root is on sys.path so `from orderChat import orderChat`
# resolves when launched from any cwd.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv()

app = FastAPI(title="OrderingServiceText API")

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
    "backend.routers.ai",  # NEW — wraps standalone AI scripts
):
    _try_include(_modpath)
_try_include("backend.routers.ai_extras")  # Custom Feature Suggestions (batch 11)
_try_include("backend.routers.customViews")  # Order Views (kanban, volume, broadcast, receipt)


# ----- Health & meta -----
@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "ai_configured": bool(os.getenv("OPENROUTER_API_KEY")),
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
