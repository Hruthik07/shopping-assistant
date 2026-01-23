"""Shopping cart MCP tools."""

from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
from src.mcp.mcp_client import MCPTool, tool_registry
from src.database.models import CartItem, User
from src.database.db import SessionLocal
from src.data.document_loader import document_loader
from src.analytics.logger import logger


class AddToCartTool(MCPTool):
    """Add a product to the shopping cart."""

    def __init__(self):
        super().__init__(
            name="add_to_cart",
            description="Add a product to the user's shopping cart. Requires user_id and product_id.",
        )

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "integer",
                    "description": "User ID (optional, can use session)",
                },
                "product_id": {"type": "string", "description": "Product ID to add to cart"},
                "quantity": {
                    "type": "integer",
                    "description": "Quantity to add (default: 1)",
                    "default": 1,
                },
            },
            "required": ["product_id"],
        }

    async def execute(
        self,
        product_id: str,
        quantity: int = 1,
        user_id: Optional[int] = None,
        db: Optional[Session] = None,
    ) -> Dict[str, Any]:
        """Add product to cart."""
        should_close = False
        if db is None:
            db = SessionLocal()
            should_close = True

        if not user_id:
            return {
                "success": False,
                "message": "User ID required. Please login or provide user_id.",
            }

        try:
            # Get product details
            product = await document_loader.get_product_by_id(product_id)
            if not product:
                return {"success": False, "message": f"Product {product_id} not found"}

            # Check if item already in cart
            existing_item = (
                db.query(CartItem)
                .filter(CartItem.user_id == user_id, CartItem.product_id == product_id)
                .first()
            )

            if existing_item:
                existing_item.quantity += quantity
                db.commit()
                return {
                    "success": True,
                    "message": f"Updated quantity in cart. Total: {existing_item.quantity}",
                    "cart_item_id": existing_item.id,
                }
            else:
                cart_item = CartItem(
                    user_id=user_id,
                    product_id=product_id,
                    product_name=product.get("name", ""),
                    quantity=quantity,
                    price=product.get("price", 0),
                )
                db.add(cart_item)
                db.commit()
                db.refresh(cart_item)

                return {
                    "success": True,
                    "message": f"Added {product.get('name', 'product')} to cart",
                    "cart_item_id": cart_item.id,
                }
        except Exception as e:
            logger.error(f"Error adding to cart: {e}")
            if db:
                db.rollback()
            return {"success": False, "error": str(e)}
        finally:
            if should_close and db:
                db.close()


class GetCartTool(MCPTool):
    """Get user's shopping cart."""

    def __init__(self):
        super().__init__(
            name="get_cart",
            description="Get the user's shopping cart with all items and total price.",
        )

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {"user_id": {"type": "integer", "description": "User ID"}},
            "required": ["user_id"],
        }

    async def execute(self, user_id: int, db: Optional[Session] = None) -> Dict[str, Any]:
        """Get cart contents."""
        should_close = False
        if db is None:
            db = SessionLocal()
            should_close = True

        try:
            cart_items = db.query(CartItem).filter(CartItem.user_id == user_id).all()

            items = []
            total = 0.0

            for item in cart_items:
                item_total = item.price * item.quantity
                total += item_total
                items.append(
                    {
                        "id": item.id,
                        "product_id": item.product_id,
                        "product_name": item.product_name,
                        "quantity": item.quantity,
                        "price": item.price,
                        "total": item_total,
                    }
                )

            return {
                "user_id": user_id,
                "items": items,
                "items_count": len(items),
                "total": round(total, 2),
                "message": f"Cart has {len(items)} items, total: ${total:.2f}",
            }
        except Exception as e:
            logger.error(f"Error getting cart: {e}")
            return {"user_id": user_id, "items": [], "error": str(e)}
        finally:
            if should_close and db:
                db.close()


class RemoveFromCartTool(MCPTool):
    """Remove item from cart."""

    def __init__(self):
        super().__init__(
            name="remove_from_cart", description="Remove an item from the shopping cart."
        )

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "user_id": {"type": "integer", "description": "User ID"},
                "cart_item_id": {"type": "integer", "description": "Cart item ID to remove"},
            },
            "required": ["user_id", "cart_item_id"],
        }

    async def execute(
        self, user_id: int, cart_item_id: int, db: Optional[Session] = None
    ) -> Dict[str, Any]:
        """Remove item from cart."""
        should_close = False
        if db is None:
            db = SessionLocal()
            should_close = True

        try:
            cart_item = (
                db.query(CartItem)
                .filter(CartItem.id == cart_item_id, CartItem.user_id == user_id)
                .first()
            )

            if not cart_item:
                return {"success": False, "message": "Cart item not found"}

            product_name = cart_item.product_name
            db.delete(cart_item)
            db.commit()

            return {"success": True, "message": f"Removed {product_name} from cart"}
        except Exception as e:
            logger.error(f"Error removing from cart: {e}")
            if db:
                db.rollback()
            return {"success": False, "error": str(e)}
        finally:
            if should_close and db:
                db.close()


# Register cart tools
add_to_cart_tool = AddToCartTool()
get_cart_tool = GetCartTool()
remove_from_cart_tool = RemoveFromCartTool()

tool_registry.register(add_to_cart_tool)
tool_registry.register(get_cart_tool)
tool_registry.register(remove_from_cart_tool)
