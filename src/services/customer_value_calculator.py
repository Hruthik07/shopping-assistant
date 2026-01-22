"""Customer value calculator for ranking products."""
from typing import Dict, Any, Optional


class CustomerValueCalculator:
    """Calculates customer value score for products.
    
    Customer value prioritizes:
    1. Lowest total cost (price + shipping - discounts)
    2. Higher discounts/deals
    3. Better ratings
    4. More reviews
    5. No affiliate bias
    """
    
    def __init__(self):
        # Default weighting factors (sum should be ~1.0)
        # These can be overridden for A/B testing or category-specific optimization
        self.default_weights = {
            "price": 0.4,  # Lower price = higher score
            "shipping": 0.2,  # Free/low shipping = higher score
            "discount": 0.2,  # Higher discount = higher score
            "rating": 0.1,  # Higher rating = higher score
            "deal": 0.1  # Price drop = higher score
        }
        
        # Category-specific weight adjustments
        self.category_weights = {
            "electronics": {
                "price": 0.35,
                "rating": 0.15,  # Higher weight on rating for electronics
                "deal": 0.15
            },
            "clothing": {
                "price": 0.45,
                "rating": 0.05  # Lower weight on rating for clothing
            },
            "books": {
                "price": 0.5,
                "shipping": 0.3  # Shipping matters more for books
            }
        }
        
        self.weights = self.default_weights.copy()
    
    def get_weights_for_category(self, category: str) -> Dict[str, float]:
        """Get optimized weights for a specific product category.
        
        Args:
            category: Product category
            
        Returns:
            Weight dictionary for the category
        """
        if not category:
            return self.default_weights.copy()
        
        category_lower = category.lower()
        weights = self.default_weights.copy()
        
        # Apply category-specific adjustments
        for cat_key, cat_weights in self.category_weights.items():
            if cat_key in category_lower:
                weights.update(cat_weights)
                break
        
        # Normalize weights to sum to 1.0
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        
        return weights
    
    def calculate_customer_value_score(
        self,
        product: Dict[str, Any],
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """Calculate customer value score for a product.
        
        Args:
            product: Product dictionary with price, shipping, discounts, etc.
            weights: Optional custom weights (if None, uses category-optimized or default)
            
        Returns:
            Customer value score (0.0 to 1.0, higher is better)
        """
        # Get weights for product category
        if weights is None:
            category = product.get("category", "")
            weights = self.get_weights_for_category(category)
        else:
            weights = weights.copy()
        
        # Calculate price score (inverse: lower price = higher score)
        price = product.get("price", 0.0)
        shipping = product.get("shipping_cost", 0.0)
        total_cost = price + shipping
        
        # Apply coupon discount if available
        coupon_info = product.get("coupon_info", {})
        if coupon_info.get("has_coupon"):
            total_cost = coupon_info.get("discounted_price", total_cost) + shipping
        
        # Normalize price score (assume max price of $10000 for normalization)
        # Lower price gets higher score
        max_price = 10000.0
        price_score = max(0.0, 1.0 - (total_cost / max_price))
        
        # Calculate shipping score (free shipping = 1.0, expensive = 0.0)
        shipping_score = 1.0 if shipping == 0.0 else max(0.0, 1.0 - (shipping / 100.0))
        
        # Calculate discount score
        discount_score = 0.0
        if coupon_info.get("has_coupon"):
            discount_percent = coupon_info.get("savings_percent", 0.0)
            discount_score = min(1.0, discount_percent / 50.0)  # 50% discount = max score
        
        # Also check deal info
        deal_info = product.get("deal_info", {})
        if deal_info.get("is_deal"):
            savings_percent = deal_info.get("savings_percent", 0.0)
            deal_discount_score = min(1.0, savings_percent / 50.0)
            discount_score = max(discount_score, deal_discount_score)
        
        # Calculate rating score (normalize 0-5 to 0-1)
        rating = product.get("rating", 0) or 0
        rating_score = rating / 5.0 if rating > 0 else 0.0
        
        # Calculate deal score (price drop = bonus)
        deal_score = 0.0
        if deal_info.get("is_deal"):
            if deal_info.get("deal_type") == "best_price":
                deal_score = 1.0
            elif deal_info.get("deal_type") == "extreme_drop":
                deal_score = 0.9
            elif deal_info.get("deal_type") == "major_drop":
                deal_score = 0.7
            elif deal_info.get("deal_type") == "significant_drop":
                deal_score = 0.5
            elif deal_info.get("deal_type") == "price_dropping":
                deal_score = 0.3
        
        # Calculate weighted customer value score
        customer_value_score = (
            price_score * weights.get("price", 0.4) +
            shipping_score * weights.get("shipping", 0.2) +
            discount_score * weights.get("discount", 0.2) +
            rating_score * weights.get("rating", 0.1) +
            deal_score * weights.get("deal", 0.1)
        )
        
        # Store score breakdown for transparency
        product["customer_value"] = {
            "score": round(customer_value_score, 3),
            "breakdown": {
                "price_score": round(price_score, 3),
                "shipping_score": round(shipping_score, 3),
                "discount_score": round(discount_score, 3),
                "rating_score": round(rating_score, 3),
                "deal_score": round(deal_score, 3)
            },
            "total_cost": round(total_cost, 2),
            "weights": weights,
            "category": product.get("category", "")
        }
        
        return customer_value_score


# Global customer value calculator instance
customer_value_calculator = CustomerValueCalculator()
