"""Deal analytics tracking."""

from typing import Dict, Any, List
from src.analytics.tracker import tracker
from src.analytics.logger import logger


class DealAnalytics:
    """Track deal-related metrics."""

    def track_deal_found(self, query: str, products: List[Dict[str, Any]]) -> None:
        """Track when deals are found for a query.

        Args:
            query: Search query
            products: List of products with deal information
        """
        deals_found = 0
        total_savings = 0.0
        best_savings = 0.0

        for product in products:
            deal_info = product.get("deal_info", {})
            coupon_info = product.get("coupon_info", {})

            if deal_info.get("is_deal") or coupon_info.get("has_coupon"):
                deals_found += 1

                savings = deal_info.get("savings_amount", 0) + coupon_info.get("savings_amount", 0)
                total_savings += savings
                best_savings = max(best_savings, savings)

        if deals_found > 0:
            tracker.track_event(
                "deal_found",
                {
                    "query": query,
                    "deals_count": deals_found,
                    "total_products": len(products),
                    "deals_percentage": (deals_found / len(products)) * 100 if products else 0,
                    "total_savings": total_savings,
                    "best_savings": best_savings,
                },
            )

    def track_price_comparison(self, query: str, products: List[Dict[str, Any]]) -> None:
        """Track price comparison metrics.

        Args:
            query: Search query
            products: List of products with price comparison data
        """
        products_with_comparison = 0
        total_retailers = 0
        total_savings = 0.0

        for product in products:
            price_comp = product.get("price_comparison", {})
            retailer_count = price_comp.get("retailer_count", 0)

            if retailer_count > 1:
                products_with_comparison += 1
                total_retailers += retailer_count
                savings = price_comp.get("savings", 0)
                total_savings += savings

        if products_with_comparison > 0:
            tracker.track_event(
                "price_comparison",
                {
                    "query": query,
                    "products_with_comparison": products_with_comparison,
                    "average_retailers_per_product": (
                        total_retailers / products_with_comparison
                        if products_with_comparison > 0
                        else 0
                    ),
                    "total_savings": total_savings,
                },
            )


# Global deal analytics instance
deal_analytics = DealAnalytics()
