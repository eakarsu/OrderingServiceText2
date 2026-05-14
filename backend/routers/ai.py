"""AI router — wraps standalone AI scripts (orderChat, orderProcessor,
menuIndexer) as HTTP POST endpoints under /api/ai/*.

Design:
  * Lazy-imports orderChat (and its heavy deps: chromadb, langchain, psycopg2)
    so the FastAPI app can boot even when the AI stack is unconfigured.
  * Each endpoint returns HTTP 503 with {"error": "AI not configured"} when
    OPENROUTER_API_KEY is missing or the underlying script raises during init.
  * Sessions are kept in-memory keyed by caller_id, mirroring api/index.py.
"""

import os
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/ai", tags=["ai"])

# In-memory session store: caller_id -> orderChat instance.
_sessions: dict = {}


def _ai_configured() -> bool:
    """Return True iff the OpenRouter key is set."""
    return bool(os.getenv("OPENROUTER_API_KEY"))


def _ai_unavailable():
    raise HTTPException(status_code=503, detail={"error": "AI not configured"})


def _get_or_create_session(caller_id: str):
    """Lazy-import orderChat and create a session, or return existing one.

    Raises HTTPException(503) on any import or init failure so missing
    deps (chromadb, postgres, etc.) degrade gracefully.
    """
    if not _ai_configured():
        _ai_unavailable()
    if caller_id in _sessions:
        return _sessions[caller_id]
    try:
        # Import lazily — these modules pull chromadb, langchain, psycopg2.
        from orderChat import orderChat  # type: ignore
        session = orderChat(caller_id)
    except Exception as exc:  # noqa: BLE001 — surface as 503
        raise HTTPException(
            status_code=503,
            detail={"error": "AI not configured", "reason": str(exc)},
        ) from exc
    _sessions[caller_id] = session
    return session


# ---------- Request schemas ----------

class ChatRequest(BaseModel):
    input: str
    caller_id: Optional[str] = "+19175587915"


class ResetRequest(BaseModel):
    caller_id: Optional[str] = "+19175587915"


class ProcessOrderRequest(BaseModel):
    query: str
    caller_id: Optional[str] = "+19175587915"


class IndexMenuRequest(BaseModel):
    menu_file: str
    rules_file: Optional[str] = None


# ---------- Endpoints ----------

@router.get("/health")
def ai_health():
    """Lightweight probe — does NOT instantiate orderChat."""
    return {
        "ai_configured": _ai_configured(),
        "active_sessions": len(_sessions),
    }


@router.post("/chat")
def ai_chat(req: ChatRequest):
    """Send a customer message to the conversational ordering agent."""
    if not req.input or not req.input.strip():
        raise HTTPException(status_code=400, detail="input is required")
    session = _get_or_create_session(req.caller_id or "+19175587915")
    try:
        response = session.chatAway(req.input)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=503,
            detail={"error": "AI not configured", "reason": str(exc)},
        ) from exc
    return {"status": "success", "response": response}


@router.post("/reset")
def ai_reset(req: ResetRequest):
    """Drop a caller's session so the next /chat starts fresh."""
    caller_id = req.caller_id or "+19175587915"
    if caller_id in _sessions:
        del _sessions[caller_id]
        return {"status": "success", "message": "session reset"}
    return {"status": "success", "message": "no active session"}


@router.post("/process-order")
def ai_process_order(req: ProcessOrderRequest):
    """Run a single-shot menu lookup via OrderProcessor (skips the LangGraph
    state machine). Useful for menu Q&A without creating an order session.
    """
    if not _ai_configured():
        _ai_unavailable()
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="query is required")
    session = _get_or_create_session(req.caller_id or "+19175587915")
    try:
        result = session.processor.process_order(req.query)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=503,
            detail={"error": "AI not configured", "reason": str(exc)},
        ) from exc
    return {"status": "success", "result": result}


@router.get("/categories")
def ai_categories(caller_id: str = "+19175587915"):
    """Return the indexed menu categories from the Chroma vector store."""
    if not _ai_configured():
        _ai_unavailable()
    session = _get_or_create_session(caller_id)
    try:
        cats = session.processor.indexer.categories_col.get()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=503,
            detail={"error": "AI not configured", "reason": str(exc)},
        ) from exc
    docs = cats.get("documents", []) if cats else []
    return {"categories": docs}


@router.post("/index-menu")
def ai_index_menu(req: IndexMenuRequest):
    """Re-run the menu / rules ingestion pipeline against on-disk text files.

    Wraps menuIndexer.MenuIndexer + MenuParser. Heavy operation; intended
    for ops/admin use, not customer traffic.
    """
    if not _ai_configured():
        _ai_unavailable()
    if not req.menu_file:
        raise HTTPException(status_code=400, detail="menu_file is required")
    try:
        from menuIndexer import MenuIndexer, MenuParser  # type: ignore
        parser = MenuParser()
        parser.parse_menu_file(req.menu_file)
        if req.rules_file:
            parser.parse_rules_file(req.rules_file)
        indexer = MenuIndexer()
        indexer.index_menu_and_rules(parser)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=f"file not found: {exc}") from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=503,
            detail={"error": "AI not configured", "reason": str(exc)},
        ) from exc
    return {"status": "success", "message": "menu indexed"}
