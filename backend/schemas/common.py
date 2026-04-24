from pydantic import BaseModel
from typing import List, Optional, Any


class PaginatedResponse(BaseModel):
    items: List[Any]
    total: int
    page: int
    page_size: int
    total_pages: int


class BulkDeleteRequest(BaseModel):
    ids: List[int]


class BulkUpdateRequest(BaseModel):
    ids: List[int]
    updates: dict


class MessageResponse(BaseModel):
    message: str
