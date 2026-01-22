"""Pydantic schemas for API requests/responses."""
from pydantic import BaseModel, EmailStr
from typing import Optional, List, Dict, Any
from datetime import datetime


# User Schemas
class UserBase(BaseModel):
    username: str
    email: EmailStr


class UserCreate(UserBase):
    password: str


class UserResponse(UserBase):
    id: int
    created_at: datetime
    
    class Config:
        from_attributes = True


# Session Schemas
class SessionCreate(BaseModel):
    user_id: Optional[int] = None


class SessionResponse(BaseModel):
    id: int
    session_id: str
    user_id: Optional[int]
    created_at: datetime
    
    class Config:
        from_attributes = True


# Conversation Schemas
class ConversationCreate(BaseModel):
    session_id: int
    user_message: str


class ConversationResponse(BaseModel):
    id: int
    session_id: int
    user_message: str
    agent_response: str
    tools_used: Optional[List[str]] = None
    created_at: datetime
    
    class Config:
        from_attributes = True


# Chat Schemas
class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    tools_used: Optional[List[str]] = None
    products: Optional[List[Dict[str, Any]]] = None


# Product Schemas
class ProductResponse(BaseModel):
    id: str
    name: str
    description: str
    price: float
    category: str
    rating: Optional[float] = None
    image_url: Optional[str] = None
    availability: bool = True


# Cart Schemas
class CartItemCreate(BaseModel):
    product_id: str
    quantity: int = 1


class CartItemResponse(BaseModel):
    id: int
    product_id: str
    product_name: str
    quantity: int
    price: float
    added_at: datetime
    
    class Config:
        from_attributes = True


# User Preference Schemas
class UserPreferenceCreate(BaseModel):
    category: Optional[str] = None
    brand: Optional[str] = None
    price_range_min: Optional[float] = None
    price_range_max: Optional[float] = None
    preferences: Optional[Dict[str, Any]] = None


class UserPreferenceResponse(BaseModel):
    id: int
    user_id: int
    category: Optional[str]
    brand: Optional[str]
    price_range_min: Optional[float]
    price_range_max: Optional[float]
    preferences: Optional[Dict[str, Any]]
    created_at: datetime
    
    class Config:
        from_attributes = True



