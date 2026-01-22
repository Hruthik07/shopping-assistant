"""Shopping cart API routes."""
from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.orm import Session
from typing import Optional
from starlette.exceptions import HTTPException as StarletteHTTPException
from src.api.schemas import CartItemCreate, CartItemResponse
from src.mcp.tools.cart_tools import add_to_cart_tool, get_cart_tool, remove_from_cart_tool
from src.database.db import get_db
from src.analytics.logger import logger

router = APIRouter(prefix="/api/cart", tags=["cart"])


@router.post("/items")
async def add_item(
    item: CartItemCreate,
    user_id: Optional[int] = None,  # In real app, get from auth token
    db: Session = Depends(get_db)
):
    """Add item to cart."""
    try:
        if not user_id:
            raise HTTPException(status_code=401, detail="Authentication required")
        
        result = await add_to_cart_tool.execute(
            product_id=item.product_id,
            quantity=item.quantity,
            user_id=user_id,
            db=db
        )
        
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("message", "Failed to add item"))
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        # Defensive: if something raised Starlette HTTPException, preserve its status code
        if isinstance(e, StarletteHTTPException):
            raise e
        logger.error(f"Error adding to cart: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/")
async def get_cart(
    user_id: Optional[int] = Query(None, description="User ID (in real app, derive from auth token)"),
    db: Session = Depends(get_db)
):
    """Get shopping cart."""
    try:
        if not user_id:
            raise HTTPException(status_code=401, detail="Authentication required")
        result = await get_cart_tool.execute(user_id=user_id, db=db)
        return result
    except HTTPException:
        raise
    except Exception as e:
        if isinstance(e, StarletteHTTPException):
            raise e
        logger.error(f"Error getting cart: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/items/{item_id}")
async def remove_item(
    item_id: int,
    user_id: Optional[int] = Query(None, description="User ID (in real app, derive from auth token)"),
    db: Session = Depends(get_db)
):
    """Remove item from cart."""
    try:
        if not user_id:
            raise HTTPException(status_code=401, detail="Authentication required")
        result = await remove_from_cart_tool.execute(
            user_id=user_id,
            cart_item_id=item_id,
            db=db
        )
        
        if not result.get("success"):
            raise HTTPException(status_code=404, detail=result.get("message", "Item not found"))
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        if isinstance(e, StarletteHTTPException):
            raise e
        logger.error(f"Error removing from cart: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

