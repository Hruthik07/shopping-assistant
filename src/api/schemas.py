"""API request/response schemas."""
from pydantic import BaseModel
from typing import Optional, List, Dict, Any


class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = None
    persona: Optional[str] = "friendly"  # friendly, professional, casual, expert
    tone: Optional[str] = "warm"  # warm, confident, enthusiastic, calm


class ChatResponse(BaseModel):
    response: str
    session_id: str
    tools_used: Optional[List[str]] = None
    products: Optional[List[Dict[str, Any]]] = None
    request_id: Optional[str] = None
    latency_breakdown: Optional[Dict[str, float]] = None
    ttft: Optional[float] = None  # Time To First Token
    cache_stats: Optional[Dict[str, Any]] = None  # Cache hit rate and statistics


class CartItemCreate(BaseModel):
    product_id: str
    quantity: int = 1


class CartItemResponse(BaseModel):
    id: int
    product_id: str
    quantity: int
    price: Optional[float] = None
    product_name: Optional[str] = None

