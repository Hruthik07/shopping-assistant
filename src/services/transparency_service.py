"""Transparency service to explain ranking factors and deal reasoning."""
from typing import Dict, Any, List


class TransparencyService:
    """Service for providing transparent explanations of ranking and deals."""
    
    def explain_ranking(self, product: Dict[str, Any]) -> str:
        """Generate human-readable explanation of why a product is ranked.
        
        Args:
            product: Product dictionary with ranking information
            
        Returns:
            Explanation string
        """
        explanation = product.get("ranking_explanation", "")
        if explanation:
            return explanation
        
        # Build explanation from available data
        parts = []
        
        customer_value = product.get("customer_value", {})
        score = customer_value.get("score", 0)
        if score > 0.7:
            parts.append("High customer value score")
        
        deal_info = product.get("deal_info", {})
        if deal_info.get("is_deal"):
            badge = deal_info.get("deal_badge", "")
            if badge:
                parts.append(badge)
        
        coupon_info = product.get("coupon_info", {})
        if coupon_info.get("has_coupon"):
            parts.append("Coupon available")
        
        price_comp = product.get("price_comparison", {})
        if price_comp.get("retailer_count", 0) > 1:
            parts.append(f"Best price among {price_comp.get('retailer_count')} retailers")
        
        if not parts:
            parts.append("Good match for your search")
        
        return " • ".join(parts)
    
    def explain_deal(self, product: Dict[str, Any]) -> Dict[str, Any]:
        """Explain why a product is a good deal.
        
        Returns:
            Dictionary with deal explanation
        """
        deal_info = product.get("deal_info", {})
        coupon_info = product.get("coupon_info", {})
        price_comp = product.get("price_comparison", {})
        
        explanations = []
        savings_total = 0.0
        
        # Deal savings
        if deal_info.get("is_deal"):
            savings = deal_info.get("savings_percent", 0)
            if savings > 0:
                explanations.append(f"Price dropped {savings:.1f}% vs 30-day average")
                savings_total += deal_info.get("savings_amount", 0)
        
        # Coupon savings
        if coupon_info.get("has_coupon"):
            coupon_savings = coupon_info.get("savings_percent", 0)
            if coupon_savings > 0:
                explanations.append(f"Coupon saves additional {coupon_savings:.1f}%")
                savings_total += coupon_info.get("savings_amount", 0)
        
        # Price comparison savings
        if price_comp.get("retailer_count", 0) > 1:
            comp_savings = price_comp.get("savings_percent", 0)
            if comp_savings > 0:
                explanations.append(f"Best price - save {comp_savings:.1f}% vs other retailers")
        
        return {
            "is_good_deal": len(explanations) > 0,
            "explanations": explanations,
            "total_savings": round(savings_total, 2),
            "total_savings_percent": round(
                (savings_total / product.get("price", 1)) * 100, 1
            ) if product.get("price", 0) > 0 else 0.0
        }
    
    def get_ranking_factors(self, product: Dict[str, Any]) -> Dict[str, Any]:
        """Get detailed breakdown of ranking factors.
        
        Returns:
            Dictionary with all ranking factors
        """
        customer_value = product.get("customer_value", {})
        breakdown = customer_value.get("breakdown", {})
        
        return {
            "customer_value_score": customer_value.get("score", 0),
            "price_score": breakdown.get("price_score", 0),
            "shipping_score": breakdown.get("shipping_score", 0),
            "discount_score": breakdown.get("discount_score", 0),
            "rating_score": breakdown.get("rating_score", 0),
            "deal_score": breakdown.get("deal_score", 0),
            "total_cost": customer_value.get("total_cost", 0),
            "has_deal": product.get("deal_info", {}).get("is_deal", False),
            "has_coupon": product.get("coupon_info", {}).get("has_coupon", False),
            "retailer_count": product.get("price_comparison", {}).get("retailer_count", 1),
            "ranking_explanation": self.explain_ranking(product)
        }


# Global transparency service instance
transparency_service = TransparencyService()
