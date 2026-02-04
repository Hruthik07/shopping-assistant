"""Deal detection service to identify price drops and best deals."""

from typing import List, Dict, Any, Optional
from datetime import datetime
from src.services.price_tracker import price_tracker
from src.analytics.logger import logger


class DealDetector:
    """Detects deals, price drops, and best prices."""

    def __init__(self):
        self.price_drop_thresholds = {
            "significant": 10.0,  # 10% drop
            "major": 20.0,  # 20% drop
            "extreme": 30.0,  # 30% drop
        }
        # Seasonal deal detection (month-based)
        self.seasonal_months = {
            "black_friday": [11],  # November
            "cyber_monday": [11],  # November
            "christmas": [12],  # December
            "new_year": [1],  # January
            "summer_sales": [6, 7, 8],  # June, July, August
            "back_to_school": [8, 9],  # August, September
        }

    async def detect_deals(self, products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect deals for a list of products.

        Args:
            products: List of products to analyze

        Returns:
            Products with deal information added
        """
        products_with_deals = []

        for product in products:
            try:
                product = await self._analyze_product_deals(product)
            except Exception as e:
                logger.error(
                    f"Error detecting deals for product {product.get('id', 'unknown')}: {e}",
                    exc_info=True,
                )
                # Add product without deal info rather than skipping
                if "deal_info" not in product:
                    product["deal_info"] = {
                        "is_deal": False,
                        "deal_type": None,
                        "deal_badge": None,
                        "savings_percent": 0.0,
                    }
            products_with_deals.append(product)

        # Track deal detection metrics
        if products_with_deals:
            from src.analytics.tracker import tracker

            deals_found = sum(
                1 for p in products_with_deals if p.get("deal_info", {}).get("is_deal", False)
            )
            total_savings = sum(
                p.get("deal_info", {}).get("savings_amount", 0) for p in products_with_deals
            )
            avg_savings = sum(
                p.get("deal_info", {}).get("savings_percent", 0)
                for p in products_with_deals
                if p.get("deal_info", {}).get("is_deal", False)
            )
            avg_savings = avg_savings / deals_found if deals_found > 0 else 0.0

            tracker.track_deal_detection(
                products_analyzed=len(products_with_deals),
                deals_found=deals_found,
                total_savings=total_savings,
                average_savings_percent=avg_savings,
            )

        return products_with_deals

    def _is_seasonal_period(self, month: int) -> bool:
        """Check if current month is a seasonal sale period.

        Args:
            month: Current month (1-12)

        Returns:
            True if in seasonal period
        """
        for season, months in self.seasonal_months.items():
            if month in months:
                return True
        return False

    async def _analyze_product_deals(self, product: Dict[str, Any]) -> Dict[str, Any]:  # noqa: C901
        """Analyze a single product for deals."""
        # Get product identifier
        product_id = (
            product.get("upc")
            or product.get("gtin")
            or product.get("ean")
            or product.get("sku")
            or product.get("id", "")
        )

        if not product_id:
            return product

        # Get price statistics with error handling
        try:
            stats = await price_tracker.get_price_statistics(product_id, days=30)
        except Exception as e:
            logger.warning(f"Error getting price statistics for {product_id}: {e}")
            stats = {}

        if not stats:
            # No history yet - record current price (with error handling)
            try:
                await price_tracker.record_price(product)
            except Exception as e:
                logger.warning(f"Error recording price for {product_id}: {e}")

            # Add default deal_info even when no history
            product["deal_info"] = {
                "is_deal": False,
                "deal_type": None,
                "deal_badge": None,
                "savings_percent": 0.0,
                "savings_amount": 0.0,
                "price_trend": "new",
                "vs_average": 0.0,
                "vs_lowest": 0.0,
                "is_lowest_price": True,  # First time seeing it, so it's the lowest
                "is_limited_time": False,
                "is_seasonal": False,
                "price_statistics": {},
            }
            return product

        current_price = product.get("price", 0.0)
        shipping = product.get("shipping_cost", 0.0)
        total_cost = current_price + shipping

        # Compare with historical data
        average_price = stats.get("average_price", total_cost)
        lowest_price = stats.get("lowest_price", total_cost)
        price_trend = stats.get("price_trend", "stable")
        savings_percent = stats.get("savings_percent", 0.0)

        # Check for seasonal deals
        current_month = datetime.utcnow().month
        is_seasonal = self._is_seasonal_period(current_month)

        # Detect deal type
        deal_type = None
        deal_badge = None
        is_limited_time = False

        if total_cost <= lowest_price * 1.01:  # Within 1% of lowest
            deal_type = "best_price"
            deal_badge = "Best Price"
            if is_seasonal:
                deal_badge += " - Limited Time"
                is_limited_time = True
        elif savings_percent >= self.price_drop_thresholds["extreme"]:
            deal_type = "extreme_drop"
            deal_badge = f"Save {savings_percent:.0f}%"
            if is_seasonal:
                deal_badge += " - Seasonal Deal"
                is_limited_time = True
        elif savings_percent >= self.price_drop_thresholds["major"]:
            deal_type = "major_drop"
            deal_badge = f"Save {savings_percent:.0f}%"
            if is_seasonal:
                is_limited_time = True
        elif savings_percent >= self.price_drop_thresholds["significant"]:
            deal_type = "significant_drop"
            deal_badge = f"Save {savings_percent:.0f}%"
        elif price_trend == "decreasing":
            deal_type = "price_dropping"
            deal_badge = "Price Dropping"

        # Add limited time indicator for significant deals during seasonal periods
        if is_seasonal and deal_type and savings_percent >= 15.0:
            is_limited_time = True

        # Add deal information
        product["deal_info"] = {
            "is_deal": deal_type is not None,
            "deal_type": deal_type,
            "deal_badge": deal_badge,
            "savings_percent": round(savings_percent, 1),
            "savings_amount": (
                round(average_price - total_cost, 2) if average_price > total_cost else 0.0
            ),
            "price_trend": price_trend,
            "vs_average": round(average_price - total_cost, 2),
            "vs_lowest": round(total_cost - lowest_price, 2),
            "is_lowest_price": total_cost <= lowest_price * 1.01,
            "is_limited_time": is_limited_time,
            "is_seasonal": is_seasonal,
            "price_statistics": stats,
        }

        # Record current price for future tracking
        await price_tracker.record_price(product)

        return product

    def is_best_deal(
        self, product: Dict[str, Any], comparison_products: List[Dict[str, Any]]
    ) -> bool:
        """Check if a product is the best deal among comparison products."""
        if not comparison_products:
            return False

        product_price = product.get("price", 0.0) + product.get("shipping_cost", 0.0)

        for comp_product in comparison_products:
            comp_price = comp_product.get("price", 0.0) + comp_product.get("shipping_cost", 0.0)
            if comp_price < product_price:
                return False

        return True


# Global deal detector instance
deal_detector = DealDetector()
