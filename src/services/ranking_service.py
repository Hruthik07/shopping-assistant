"""Customer-first ranking service with transparency."""

from typing import List, Dict, Any, Optional
from src.services.customer_value_calculator import customer_value_calculator
from src.services.price_comparison import price_comparator
from src.analytics.logger import logger


class RankingService:
    """Customer-first ranking service with transparent scoring."""

    def rank_products(
        self, products: List[Dict[str, Any]], sort_by: str = "customer_value"
    ) -> List[Dict[str, Any]]:
        """Rank products using customer-first algorithm.

        Args:
            products: List of products to rank
            sort_by: Sort method - "customer_value", "price", "rating", "savings", "relevance"

        Returns:
            Ranked products with ranking explanation
        """
        if not products:
            return []

        # Calculate customer value scores for all products with error handling
        for product in products:
            try:
                customer_value_calculator.calculate_customer_value_score(product)
            except Exception as e:
                logger.error(
                    f"Error calculating customer value for product {product.get('id', 'unknown')}: {e}",
                    exc_info=True,
                )
                # Add default customer value on error
                if "customer_value" not in product:
                    product["customer_value"] = {
                        "score": 0.0,
                        "breakdown": {},
                        "total_cost": product.get("price", 0) + product.get("shipping_cost", 0),
                    }

        # Sort based on requested method
        if sort_by == "customer_value":
            sorted_products = sorted(
                products, key=lambda p: p.get("customer_value", {}).get("score", 0), reverse=True
            )
        elif sort_by == "price":
            sorted_products = price_comparator.rank_by_customer_value(products)
        elif sort_by == "rating":
            sorted_products = sorted(
                products,
                key=lambda p: (p.get("rating", 0) or 0, p.get("reviews", 0) or 0),
                reverse=True,
            )
        elif sort_by == "savings":
            sorted_products = sorted(
                products,
                key=lambda p: (
                    p.get("deal_info", {}).get("savings_percent", 0),
                    p.get("coupon_info", {}).get("savings_percent", 0),
                ),
                reverse=True,
            )
        else:  # relevance (default)
            # Use semantic score if available, otherwise customer value
            sorted_products = sorted(
                products,
                key=lambda p: (
                    p.get("_semantic_score", 0),
                    p.get("customer_value", {}).get("score", 0),
                ),
                reverse=True,
            )

        # Add ranking position and explanation
        for idx, product in enumerate(sorted_products, 1):
            product["rank"] = idx
            product["ranking_explanation"] = self._generate_ranking_explanation(product, sort_by)

        return sorted_products

    def _generate_ranking_explanation(self, product: Dict[str, Any], sort_by: str) -> str:
        """Generate human-readable explanation of why product is ranked."""
        explanations = []

        customer_value = product.get("customer_value", {})
        breakdown = customer_value.get("breakdown", {})

        if sort_by == "customer_value":
            score = customer_value.get("score", 0)
            explanations.append(f"Customer value score: {score:.2f}")

            if breakdown.get("price_score", 0) > 0.7:
                explanations.append("Excellent price")
            if breakdown.get("discount_score", 0) > 0.5:
                explanations.append("Great discount available")
            if breakdown.get("deal_score", 0) > 0.5:
                explanations.append("Price drop detected")

        deal_info = product.get("deal_info", {})
        if deal_info.get("is_deal"):
            badge = deal_info.get("deal_badge", "")
            if badge:
                explanations.append(badge)
            savings = deal_info.get("savings_percent", 0)
            if savings > 0:
                explanations.append(f"Save {savings:.1f}% vs average")

        coupon_info = product.get("coupon_info", {})
        if coupon_info.get("has_coupon"):
            coupon_savings = coupon_info.get("savings_percent", 0)
            if coupon_savings > 0:
                explanations.append(f"Coupon saves {coupon_savings:.1f}%")

        price_comp = product.get("price_comparison", {})
        if price_comp.get("retailer_count", 0) > 1:
            savings = price_comp.get("savings_percent", 0)
            if savings > 0:
                explanations.append(
                    f"Best price among {price_comp.get('retailer_count')} retailers"
                )

        rating = product.get("rating", 0)
        if rating and rating >= 4.0:
            explanations.append(f"Highly rated ({rating:.1f}/5)")

        if not explanations:
            explanations.append("Good match for your search")

        return " • ".join(explanations)

    def get_ranking_factors(self, product: Dict[str, Any]) -> Dict[str, Any]:
        """Get detailed ranking factors for transparency."""
        return {
            "customer_value_score": product.get("customer_value", {}).get("score", 0),
            "price": product.get("price", 0),
            "total_cost": product.get("customer_value", {}).get("total_cost", 0),
            "rating": product.get("rating", 0),
            "reviews": product.get("reviews", 0),
            "has_deal": product.get("deal_info", {}).get("is_deal", False),
            "has_coupon": product.get("coupon_info", {}).get("has_coupon", False),
            "retailer_count": product.get("price_comparison", {}).get("retailer_count", 1),
            "explanation": product.get("ranking_explanation", ""),
        }


# Global ranking service instance
ranking_service = RankingService()
