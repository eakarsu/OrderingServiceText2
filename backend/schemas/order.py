from pydantic import BaseModel
from typing import Optional


class OrderUpdateRequest(BaseModel):
    status: Optional[str] = None
    order_data: Optional[dict] = None


class OrderResponse(BaseModel):
    id: int
    phone_number: str
    order_data: dict
    status: str
    orderdate: str
    updated_at: Optional[str] = None
