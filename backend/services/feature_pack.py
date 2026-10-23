"""Pure helpers: export, readable rendering, deterministic AI fallback (no provider call)."""
import csv
import io
import json
import os


def records_to_csv(rows, fields):
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(fields)
    for r in rows:
        writer.writerow(["" if r.get(k) is None else r.get(k) for k in fields])
    return out.getvalue()


def records_to_json(rows):
    return json.dumps({"exportedAt": "now", "records": rows}, indent=2, default=str)


def _label(k):
    import re
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(k)).replace("_", " ").replace("-", " ")
    return s[:1].upper() + s[1:] if s else s


def _scalar(v):
    if v is None:
        return "(not supplied)"
    if v == "":
        return "(empty)"
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def readable_sections(value):
    if isinstance(value, list):
        return "\n".join(readable_sections(v) for v in value) if value else "(empty list)"
    if isinstance(value, dict):
        if not value:
            return "(no fields)"
        return "\n".join(
            f"{_label(k)}:\n{readable_sections(v)}" if isinstance(v, (dict, list)) else f"{_label(k)}: {_scalar(v)}"
            for k, v in value.items()
        )
    return _scalar(value)


def has_provider(env=None):
    env = os.environ if env is None else env
    return bool((env.get("OPENROUTER_API_KEY") or env.get("OPENAI_API_KEY") or "").strip())


def _count_by(rows, key):
    out = {}
    for r in rows or []:
        k = r.get(key, "(empty)") if isinstance(r, dict) else "(empty)"
        if k is None or k == "":
            k = "(empty)"
        out[k] = out.get(k, 0) + 1
    return out


def fallback_order_summary(orders):
    by_status = _count_by(orders, "status")
    total = 0.0
    for o in orders or []:
        od = o.get("order_data") or {}
        try:
            total += float(od.get("total_price") or 0)
        except (TypeError, ValueError):
            pass
    sections = {"totalOrders": len(orders or []), "totalRevenue": total, "byStatus": by_status}
    return {"aiGenerated": False, "source": "deterministic-fallback", "providerConfigured": False,
            "feature": "order-summary", "summary": f"{len(orders or [])} order(s) totalling {total}.",
            "sections": sections, "readable": readable_sections(sections)}


def fallback_menu_insights(items):
    by_cat = _count_by(items, "category_id")
    unavailable = sum(1 for i in items or [] if not i.get("is_available", i.get("available", True)))
    sections = {"totalItems": len(items or []), "byCategory": by_cat, "unavailableCount": unavailable,
                "zeroPriceCount": sum(1 for i in items or [] if not (i.get("price") or 0))}
    return {"aiGenerated": False, "source": "deterministic-fallback", "providerConfigured": False,
            "feature": "menu-insights", "summary": f"{len(items or [])} menu item(s); {unavailable} unavailable.",
            "sections": sections, "readable": readable_sections(sections)}


def fallback_fulfillment_risk(orders):
    by_status = _count_by(orders, "status")
    pending = by_status.get("pending", 0)
    sections = {"totalOrders": len(orders or []), "byStatus": by_status, "pendingCount": pending}
    return {"aiGenerated": False, "source": "deterministic-fallback", "providerConfigured": False,
            "feature": "fulfillment-risk", "summary": f"{pending} order(s) pending fulfilment.",
            "sections": sections, "readable": readable_sections(sections)}


async def call_provider(feature, prompt, env=None):
    env = os.environ if env is None else env
    key = (env.get("OPENROUTER_API_KEY") or env.get("OPENAI_API_KEY") or "").strip()
    if not key:
        return None
    import httpx
    is_or = bool((env.get("OPENROUTER_API_KEY") or "").strip())
    base = "https://openrouter.ai/api/v1" if is_or else "https://api.openai.com/v1"
    model = env.get("OPENROUTER_MODEL") or env.get("OPENAI_MODEL") or ("openrouter/free" if is_or else "gpt-4o-mini")
    async with httpx.AsyncClient(timeout=60.0) as client:
        res = await client.post(f"{base}/chat/completions", headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                                json={"model": model, "messages": [{"role": "system", "content": "Return JSON {summary, sections} grounded only in supplied records."}, {"role": "user", "content": prompt}], "temperature": 0.2})
    if res.status_code != 200:
        from fastapi import HTTPException
        raise HTTPException(status_code=502, detail=f"AI provider failed with HTTP {res.status_code}")
    data = res.json()
    content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
    if not content or not data.get("id"):
        from fastapi import HTTPException
        raise HTTPException(status_code=502, detail="AI provider returned incomplete response")
    try:
        parsed = json.loads(content)
    except (ValueError, TypeError):
        parsed = {"summary": content, "sections": {}}
    return {"aiGenerated": True, "source": "openrouter" if is_or else "openai", "providerConfigured": True,
            "feature": feature, "model": data.get("model") or model, "providerReceiptId": data.get("id"),
            **parsed, "readable": readable_sections(parsed.get("sections") or parsed)}
