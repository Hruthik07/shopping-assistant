"""SQLAlchemy database models."""

from sqlalchemy import Column, Integer, String, Float, Text, DateTime, Boolean, JSON, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from src.database.db import Base


class User(Base):
    """User model."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    sessions = relationship("Session", back_populates="user")
    preferences = relationship("UserPreference", back_populates="user")
    cart_items = relationship("CartItem", back_populates="user")


class Session(Base):
    """User session model."""

    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    session_id = Column(String, unique=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    user = relationship("User", back_populates="sessions")
    conversations = relationship("Conversation", back_populates="session")


class Conversation(Base):
    """Conversation history model."""

    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id"))
    user_message = Column(Text)
    agent_response = Column(Text)
    tools_used = Column(JSON)  # List of tools used
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    session = relationship("Session", back_populates="conversations")


class UserPreference(Base):
    """User preference model."""

    __tablename__ = "user_preferences"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    category = Column(String)  # e.g., "electronics", "clothing"
    brand = Column(String)
    price_range_min = Column(Float, nullable=True)
    price_range_max = Column(Float, nullable=True)
    preferences = Column(JSON)  # Additional preferences
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    user = relationship("User", back_populates="preferences")


class CartItem(Base):
    """Shopping cart item model."""

    __tablename__ = "cart_items"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    product_id = Column(String, index=True)
    product_name = Column(String)
    quantity = Column(Integer, default=1)
    price = Column(Float)
    added_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    user = relationship("User", back_populates="cart_items")


class PriceHistory(Base):
    """Price history tracking model."""

    __tablename__ = "price_history"

    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(String, index=True)  # Normalized identifier (UPC/GTIN/EAN)
    product_name = Column(String, index=True)
    retailer = Column(String, index=True)
    price = Column(Float)
    currency = Column(String, default="USD")
    shipping_cost = Column(Float, default=0.0)
    total_cost = Column(Float)  # price + shipping
    original_price = Column(Float, nullable=True)  # Price before discount
    discount_amount = Column(Float, default=0.0)
    discount_percent = Column(Float, default=0.0)
    url = Column(Text)
    in_stock = Column(Boolean, default=True)
    availability = Column(Boolean, default=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    source = Column(String)  # Data source name

    # Product identifiers for deduplication
    upc = Column(String, index=True, nullable=True)
    gtin = Column(String, index=True, nullable=True)
    ean = Column(String, index=True, nullable=True)
    sku = Column(String, index=True, nullable=True)    # Metadata
    product_metadata = Column(
        JSON, nullable=True
    )  # Additional source-specific data (renamed from 'metadata' - SQLAlchemy reserved)
