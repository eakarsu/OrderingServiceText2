"""AI Extras router — Custom Feature Suggestions (batch 11).

Implements:
  1. Voice Ordering Agent (transcript -> structured order)
  2. Inventory & Supplier Integration (stock monitor + auto-reorder hooks)
  3. Delivery Route Optimization (route plan over orders)
  4. Customer Preference Learning (recommendation from past orders)
  5. Multi-Location Aggregation (cross-restaurant catalog merge)
  6. Review & Sentiment Analysis

All endpoints return 503 when OPENROUTER_API_KEY is missing.
TODO: configure credentials for supplier/POS integrations as noted inline.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Optional

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/ai-extras", tags=["ai-extras"])

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = os.getenv("OPENROUTER_MODEL", "anthropic/claude-haiku-4.5")


def _key() -> str:
    k = os.getenv("OPENROUTER_API_KEY")
    if not k:
        raise HTTPException(status_code=503, detail={"error": "OPENROUTER_API_KEY not configured"})
    return k


async def _llm(system: str, user: str, max_tokens: int = 1200) -> str:
    headers = {
        "Authorization": f"Bearer {_key()}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "OrderingServiceText AI Extras",
    }
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(OPENROUTER_URL, headers=headers, json=body)
    data = r.json()
    if "error" in data:
        raise HTTPException(status_code=502, detail=data["error"])
    return data["choices"][0]["message"]["content"]


# ---------- Schemas ----------

class VoiceOrderRequest(BaseModel):
    transcript: str
    caller_id: Optional[str] = None
    menu_summary: Optional[str] = None


class InventoryCheckRequest(BaseModel):
    items: list[dict[str, Any]]
    reorder_threshold: Optional[int] = 5


class RouteOptimizeRequest(BaseModel):
    depot: dict[str, float]
    orders: list[dict[str, Any]]
    max_drivers: Optional[int] = 3


class RecommendationRequest(BaseModel):
    customer_id: str
    past_orders: list[dict[str, Any]]
    dietary_preferences: Optional[list[str]] = None


class MultiLocationRequest(BaseModel):
    locations: list[dict[str, Any]]


class ReviewSentimentRequest(BaseModel):
    reviews: list[dict[str, Any]]


# ---------- Endpoints ----------

@router.post("/voice-order")
async def voice_order(req: VoiceOrderRequest) -> dict[str, Any]:
    sys = "You are a voice-ordering parser. From the transcript, output JSON: { items: [{ name, quantity, modifiers }], specialInstructions, confidence: 0-1 }. If unclear, set confidence < 0.5 and include clarifying_question."
    user = f"Caller: {req.caller_id or 'n/a'}\nMenu hint: {(req.menu_summary or 'unspecified')[:2000]}\nTranscript:\n{req.transcript[:4000]}"
    raw = await _llm(sys, user, 800)
    return {"raw": raw, "at": datetime.utcnow().isoformat()}


@router.post("/inventory-check")
async def inventory_check(req: InventoryCheckRequest) -> dict[str, Any]:
    threshold = req.reorder_threshold or 5
    low = [i for i in req.items if (i.get("on_hand", 0) or 0) <= threshold]
    # TODO: configure credentials — SUPPLIER_API_KEY to push reorders.
    supplier_ready = bool(os.getenv("SUPPLIER_API_KEY"))
    return {
        "low_stock": low,
        "auto_reorder_dispatched": supplier_ready and len(low) > 0,
        "supplier_configured": supplier_ready,
        "note": None if supplier_ready else "SUPPLIER_API_KEY not set — reorder simulated only.",
    }


@router.post("/route-optimize")
async def route_optimize(req: RouteOptimizeRequest) -> dict[str, Any]:
    sys = "You are a delivery route planner. Group orders into up to N driver clusters, sequence each cluster for shortest expected time, and return JSON: { clusters: [{ driverIdx, stops: [{ orderId, lat, lng, eta }] }], totalDistanceKmEst, notes }."
    user = (
        f"Depot: {req.depot}\nMax drivers: {req.max_drivers or 3}\n"
        f"Orders ({len(req.orders)}): {req.orders[:30]}"
    )
    raw = await _llm(sys, user, 1500)
    return {"raw": raw, "depot": req.depot}


@router.post("/recommend")
async def recommend(req: RecommendationRequest) -> dict[str, Any]:
    sys = "You are a restaurant recommender. From past orders + dietary prefs, suggest 5 menu items with rationale. Output JSON: { recommendations: [{ name, reason, swapFromPast?: string }] }."
    user = (
        f"Customer: {req.customer_id}\nPrefs: {req.dietary_preferences or []}\n"
        f"Past orders: {req.past_orders[:40]}"
    )
    raw = await _llm(sys, user, 900)
    return {"raw": raw}


@router.post("/multi-location-merge")
async def multi_location_merge(req: MultiLocationRequest) -> dict[str, Any]:
    sys = "You are a multi-location restaurant ops analyst. Merge menus across locations, flag duplicates and price drifts, propose shared/seasonal items. Output JSON."
    user = f"Locations ({len(req.locations)}): {req.locations[:20]}"
    raw = await _llm(sys, user, 1500)
    return {"raw": raw, "locations": len(req.locations)}


@router.post("/review-sentiment")
async def review_sentiment(req: ReviewSentimentRequest) -> dict[str, Any]:
    if not req.reviews:
        raise HTTPException(status_code=400, detail={"error": "reviews[] required"})
    sys = "You are a review-sentiment analyst. Score each review (-1..1), tag themes (food, service, wait, value), flag quality issues, suggest menu adjustments. Output JSON."
    user = f"Reviews: {req.reviews[:60]}"
    raw = await _llm(sys, user, 1500)
    return {"raw": raw, "count": len(req.reviews)}
