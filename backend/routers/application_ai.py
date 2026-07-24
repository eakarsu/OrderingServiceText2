"""Authenticated, durable OpenRouter assistance for the ordering application."""

import os
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, field_validator

from backend.database import get_db
from backend.dependencies import get_current_user

router = APIRouter(prefix="/api/application-ai", tags=["application-ai"])
REQUIRED_BASE_URL = "https://openrouter.ai/api/v1"


class OrderAdviceRequest(BaseModel):
    prompt: str

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 10 or len(value) > 5000:
            raise ValueError("prompt must contain between 10 and 5000 characters")
        return value


@router.post("/order-advice")
async def order_advice(
    request: OrderAdviceRequest,
    user: dict[str, Any] = Depends(get_current_user),
    conn=Depends(get_db),
):
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    model = os.getenv("OPENROUTER_MODEL", "")
    base_url = os.getenv("OPENROUTER_BASE_URL", "")
    if not api_key:
        raise HTTPException(status_code=503, detail="OPENROUTER_API_KEY is required")
    if not model:
        raise HTTPException(status_code=503, detail="OPENROUTER_MODEL is required")
    if base_url != REQUIRED_BASE_URL:
        raise HTTPException(status_code=503, detail="OPENROUTER_BASE_URL must use the configured OpenRouter API")

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are an ordering-service operations advisor. Give concise, practical guidance while flagging inventory, allergy, tax, payment, fulfillment, and human-approval constraints. Never claim a provider action was executed.",
            },
            {"role": "user", "content": request.prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 700,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.getenv("CLIENT_URL", "http://127.0.0.1"),
        "X-Title": "Ordering Service Text",
    }
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(f"{REQUIRED_BASE_URL}/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        provider_payload = response.json()
        advice = provider_payload.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        if not advice:
            raise ValueError("OpenRouter returned no ordering advice")
    except (httpx.HTTPError, ValueError, KeyError, IndexError) as exc:
        raise HTTPException(status_code=502, detail="OpenRouter advice request failed") from exc

    with conn.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO ordering_ai_results(user_id,prompt,model,provider_receipt_id,result,usage)
            VALUES (%s,%s,%s,%s,%s,%s::jsonb)
            RETURNING id,created_at
            """,
            (
                user["id"],
                request.prompt,
                model,
                provider_payload.get("id"),
                advice,
                __import__("json").dumps(provider_payload.get("usage", {})),
            ),
        )
        stored = cursor.fetchone()
        conn.commit()
    return {"id": stored[0], "advice": advice, "model": model, "createdAt": stored[1].isoformat()}
