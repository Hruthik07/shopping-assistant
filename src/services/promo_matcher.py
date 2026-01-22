"""Promo code matching service."""
from typing import List, Dict, Any, Optional
from src.services.coupon_service import coupon_service
from src.analytics.logger import logger


class PromoMatcher:
    """Matches promo codes to products and calculates final prices."""
    
    async def match_promos_to_products(
        self,
        products: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Match available promo codes to products.
        
        Args:
            products: List of products
            
        Returns:
            Products with promo code information added
        """
        if not coupon_service.is_available():
            # Add default coupon_info if service unavailable
            for product in products:
                if "coupon_info" not in product:
                    product["coupon_info"] = {"has_coupon": False, "available_coupons": 0}
            return products
        
        products_with_promos = []
        
        for product in products:
            try:
                # Get retailer from product
                retailer = product.get("retailer", "")
                
                if not retailer:
                    if "coupon_info" not in product:
                        product["coupon_info"] = {"has_coupon": False, "available_coupons": 0}
                    products_with_promos.append(product)
                    continue
                
                # Find coupons for this retailer with error handling
                try:
                    coupons = await coupon_service.find_coupons(
                        retailer=retailer,
                        product_url=product.get("product_url")
                    )
                except Exception as e:
                    logger.warning(f"Error finding coupons for {retailer}: {e}")
                    coupons = []
            
                if coupons:
                    # Apply best coupon
                    try:
                        best_coupon = self._select_best_coupon(coupons, product.get("price", 0.0))
                        discount_info = coupon_service.calculate_discounted_price(
                            product.get("price", 0.0),
                            best_coupon
                        )
                        
                        product["coupon_info"] = {
                            "has_coupon": True,
                            "coupon_code": discount_info.get("coupon_code", ""),
                            "coupon_description": discount_info.get("coupon_description", ""),
                            "original_price": product.get("price", 0.0),
                            "discounted_price": discount_info.get("discounted_price", product.get("price", 0.0)),
                            "savings_amount": discount_info.get("savings_amount", 0.0),
                            "savings_percent": discount_info.get("savings_percent", 0.0),
                            "available_coupons": len(coupons)
                        }
                        
                        # Update price to discounted price
                        product["price_after_coupon"] = discount_info.get("discounted_price", product.get("price", 0.0))
                    except Exception as e:
                        logger.warning(f"Error applying coupon to product: {e}")
                        product["coupon_info"] = {"has_coupon": False, "available_coupons": 0}
                else:
                    product["coupon_info"] = {
                        "has_coupon": False,
                        "available_coupons": 0
                    }
            except Exception as e:
                logger.error(f"Error matching promos for product {product.get('id', 'unknown')}: {e}", exc_info=True)
                # Add default coupon_info on error
                if "coupon_info" not in product:
                    product["coupon_info"] = {"has_coupon": False, "available_coupons": 0}
            
            products_with_promos.append(product)
        
        return products_with_promos
    
    def _select_best_coupon(
        self,
        coupons: List[Dict[str, Any]],
        original_price: float
    ) -> Dict[str, Any]:
        """Select the coupon that provides the best savings."""
        if not coupons:
            return {}
        
        best_coupon = None
        best_savings = 0.0
        
        for coupon in coupons:
            discount_info = coupon_service.calculate_discounted_price(original_price, coupon)
            savings = discount_info["savings_amount"]
            
            if savings > best_savings:
                best_savings = savings
                best_coupon = coupon
        
        return best_coupon or coupons[0]


# Global promo matcher instance
promo_matcher = PromoMatcher()
