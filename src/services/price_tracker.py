"""Price history tracking service."""
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import desc
from src.database.db import get_db
from src.database.models import PriceHistory
from src.analytics.logger import logger


class PriceTracker:
    """Tracks price history for products."""
    
    # Data retention policy (days)
    RETENTION_DAYS = 90  # Keep 90 days of price history
    
    async def record_price(
        self,
        product: Dict[str, Any],
        db: Optional[Session] = None
    ) -> None:
        """Record current price for a product.
        
        Args:
            product: Product dictionary with price information
            db: Database session (optional, will create if not provided)
        """
        db_provided = db is not None
        if db is None:
            db = next(get_db())
        
        try:
            
            # Extract product identifier
            product_id = (
                product.get("upc") or
                product.get("gtin") or
                product.get("ean") or
                product.get("sku") or
                product.get("id", "")
            )
            
            if not product_id:
                logger.warning("Cannot record price: product has no identifier")
                return
            
            price = product.get("price", 0.0)
            shipping = product.get("shipping_cost", 0.0)
            total_cost = price + shipping
            original_price = product.get("original_price")
            discount_amount = 0.0
            discount_percent = 0.0
            
            if original_price and original_price > price:
                discount_amount = original_price - price
                discount_percent = (discount_amount / original_price) * 100
            
            price_record = PriceHistory(
                product_id=product_id,
                product_name=product.get("name", ""),
                retailer=product.get("retailer", ""),
                price=price,
                currency=product.get("currency", "USD"),
                shipping_cost=shipping,
                total_cost=total_cost,
                original_price=original_price,
                discount_amount=discount_amount,
                discount_percent=discount_percent,
                url=product.get("product_url", ""),
                in_stock=product.get("in_stock", True),
                availability=product.get("availability", True),
                timestamp=datetime.utcnow(),
                source=product.get("source", ""),
                upc=product.get("upc"),
                gtin=product.get("gtin"),
                ean=product.get("ean"),
                sku=product.get("sku"),
                product_metadata=product.get("metadata", {})
            )
            
            db.add(price_record)
            db.commit()
            
        except Exception as e:
            logger.error(f"Error recording price history: {e}")
            if db:
                db.rollback()
        finally:
            if not db_provided and db:
                db.close()
    
    async def get_price_history(
        self,
        product_id: str,
        retailer: Optional[str] = None,
        days: int = 30,
        db: Optional[Session] = None
    ) -> List[Dict[str, Any]]:
        """Get price history for a product.
        
        Args:
            product_id: Product identifier (UPC/GTIN/EAN/SKU)
            retailer: Optional retailer filter
            days: Number of days of history to retrieve
            db: Database session
            
        Returns:
            List of price history records
        """
        db_provided = db is not None
        if db is None:
            db = next(get_db())
        
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            query = db.query(PriceHistory).filter(
                PriceHistory.product_id == product_id,
                PriceHistory.timestamp >= cutoff_date
            )
            
            if retailer:
                query = query.filter(PriceHistory.retailer == retailer)
            
            records = query.order_by(desc(PriceHistory.timestamp)).all()
            
            return [
                {
                    "price": record.price,
                    "total_cost": record.total_cost,
                    "currency": record.currency,
                    "retailer": record.retailer,
                    "timestamp": record.timestamp.isoformat(),
                    "discount_percent": record.discount_percent,
                    "in_stock": record.in_stock
                }
                for record in records
            ]
            
        except Exception as e:
            logger.error(f"Error retrieving price history: {e}")
            return []
        finally:
            if not db_provided and db:
                db.close()
    
    async def get_price_statistics(
        self,
        product_id: str,
        retailer: Optional[str] = None,
        days: int = 30,
        db: Optional[Session] = None
    ) -> Dict[str, Any]:
        """Get price statistics for a product.
        
        Returns:
            Dictionary with: average_price, lowest_price, highest_price, current_price,
            price_trend (increasing/decreasing/stable), savings_percent
        """
        try:
            history = await self.get_price_history(product_id, retailer, days, db)
            
            if not history:
                return {}
            
            prices = [h["total_cost"] for h in history]
            current_price = prices[0] if prices else 0.0
            average_price = sum(prices) / len(prices) if prices else 0.0
            lowest_price = min(prices) if prices else 0.0
            highest_price = max(prices) if prices else 0.0
            
            # Calculate price trend (compare recent vs older prices)
            recent_prices = prices[:len(prices)//3] if len(prices) >= 3 else prices[:1]
            older_prices = prices[len(prices)//3:] if len(prices) >= 3 else []
            
            price_trend = "stable"
            if recent_prices and older_prices:
                recent_avg = sum(recent_prices) / len(recent_prices)
                older_avg = sum(older_prices) / len(older_prices)
                
                if recent_avg < older_avg * 0.95:  # 5% decrease
                    price_trend = "decreasing"
                elif recent_avg > older_avg * 1.05:  # 5% increase
                    price_trend = "increasing"
            
            # Calculate savings
            savings_percent = 0.0
            if average_price > 0 and current_price < average_price:
                savings_percent = ((average_price - current_price) / average_price) * 100
            
            return {
                "current_price": current_price,
                "average_price": round(average_price, 2),
                "lowest_price": lowest_price,
                "highest_price": highest_price,
                "price_trend": price_trend,
                "savings_percent": round(savings_percent, 1),
                "price_drop": current_price < average_price,
                "days_tracked": days,
                "data_points": len(history)
            }
            
        except Exception as e:
            logger.error(f"Error calculating price statistics: {e}")
            return {}
    
    async def cleanup_old_records(
        self,
        days: Optional[int] = None,
        db: Optional[Session] = None
    ) -> int:
        """Clean up old price history records beyond retention period.
        
        Args:
            days: Retention period in days (default: RETENTION_DAYS)
            db: Database session
            
        Returns:
            Number of records deleted
        """
        db_provided = db is not None
        if db is None:
            db = next(get_db())
        
        try:
            retention_days = days or self.RETENTION_DAYS
            cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
            
            # Delete old records
            deleted = db.query(PriceHistory).filter(
                PriceHistory.timestamp < cutoff_date
            ).delete()
            
            db.commit()
            logger.info(f"Cleaned up {deleted} old price history records (older than {retention_days} days)")
            return deleted
            
        except Exception as e:
            logger.error(f"Error cleaning up old price records: {e}")
            if db:
                db.rollback()
            return 0
        finally:
            if not db_provided and db:
                db.close()
    
    async def get_price_history_optimized(
        self,
        product_id: str,
        retailer: Optional[str] = None,
        days: int = 30,
        limit: Optional[int] = None,
        db: Optional[Session] = None
    ) -> List[Dict[str, Any]]:
        """Get price history with optimized query (uses indexes).
        
        Args:
            product_id: Product identifier
            retailer: Optional retailer filter
            days: Number of days of history
            limit: Maximum number of records to return
            db: Database session
            
        Returns:
            List of price history records
        """
        db_provided = db is not None
        if db is None:
            db = next(get_db())
        
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            # Optimized query using indexes
            query = db.query(PriceHistory).filter(
                PriceHistory.product_id == product_id,
                PriceHistory.timestamp >= cutoff_date
            )
            
            if retailer:
                query = query.filter(PriceHistory.retailer == retailer)
            
            # Order by timestamp (uses index)
            query = query.order_by(desc(PriceHistory.timestamp))
            
            # Apply limit if specified
            if limit:
                query = query.limit(limit)
            
            records = query.all()
            
            return [
                {
                    "price": record.price,
                    "total_cost": record.total_cost,
                    "currency": record.currency,
                    "retailer": record.retailer,
                    "timestamp": record.timestamp.isoformat(),
                    "discount_percent": record.discount_percent,
                    "in_stock": record.in_stock
                }
                for record in records
            ]
            
        except Exception as e:
            logger.error(f"Error retrieving optimized price history: {e}")
            return []
        finally:
            if not db_provided and db:
                db.close()


# Global price tracker instance
price_tracker = PriceTracker()
