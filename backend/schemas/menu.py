from pydantic import BaseModel
from typing import Optional, List


class CategoryCreateRequest(BaseModel):
    name: str
    description: Optional[str] = None
    image_url: Optional[str] = None
    sort_order: int = 0
    is_active: bool = True


class CategoryUpdateRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    image_url: Optional[str] = None
    sort_order: Optional[int] = None
    is_active: Optional[bool] = None


class CategoryResponse(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    image_url: Optional[str] = None
    sort_order: int
    is_active: bool
    created_at: str
    item_count: int = 0


class MenuItemCreateRequest(BaseModel):
    category_id: int
    name: str
    description: Optional[str] = None
    price: float = 0.00
    image_url: Optional[str] = None
    is_available: bool = True


class MenuItemUpdateRequest(BaseModel):
    category_id: Optional[int] = None
    name: Optional[str] = None
    description: Optional[str] = None
    price: Optional[float] = None
    image_url: Optional[str] = None
    is_available: Optional[bool] = None


class MenuItemResponse(BaseModel):
    id: int
    category_id: int
    category_name: str = ""
    name: str
    description: Optional[str] = None
    price: float
    image_url: Optional[str] = None
    is_available: bool
    created_at: str
