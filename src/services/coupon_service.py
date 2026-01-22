"""Coupon and promo code integration service."""
import httpx
from typing import List, Dict, Any, Optional
from src.utils.config import settings
from src.analytics.logger import logger


class CouponService:
    """Service for finding and matching coupons/promo codes."""
    
    def __init__(self):
        self.honey_api_key = settings.honey_api_key
        self.retailmenot_api_key = settings.retailmenot_api_key
        self.couponfollow_api_key = settings.couponfollow_api_key
    
    def is_available(self) -> bool:
        """Check if any coupon API is available."""
        return bool(self.honey_api_key or self.retailmenot_api_key or self.couponfollow_api_key)
    
    async def find_coupons(
        self,
        retailer: str,
        product_url: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Find available coupons for a retailer.
        
        Args:
            retailer: Retailer name (e.g., "Amazon", "Walmart")
            product_url: Optional product URL for context
            
        Returns:
            List of available coupons
        """
        coupons = []
        
        # Try Honey API
        if self.honey_api_key:
            honey_coupons = await self._fetch_honey_coupons(retailer, product_url)
            coupons.extend(honey_coupons)
        
        # Try RetailMeNot API
        if self.retailmenot_api_key:
            rmn_coupons = await self._fetch_retailmenot_coupons(retailer)
            coupons.extend(rmn_coupons)
        
        # Try CouponFollow API
        if self.couponfollow_api_key:
            cf_coupons = await self._fetch_couponfollow_coupons(retailer)
            coupons.extend(cf_coupons)
        
        # Remove duplicates and return
        seen_codes = set()
        unique_coupons = []
        for coupon in coupons:
            code = coupon.get("code", "")
            if code and code not in seen_codes:
                seen_codes.add(code)
                unique_coupons.append(coupon)
        
        return unique_coupons
    
    async def _fetch_honey_coupons(
        self,
        retailer: str,
        product_url: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Fetch coupons from Honey API."""
        # Note: Honey API structure may vary - this is a placeholder implementation
        # Actual implementation would depend on Honey's API documentation
        try:
            # Placeholder - would need actual Honey API integration
            logger.debug(f"Fetching Honey coupons for {retailer}")
            return []
        except Exception as e:
            logger.error(f"Error fetching Honey coupons: {e}")
            return []
    
    async def _fetch_retailmenot_coupons(self, retailer: str) -> List[Dict[str, Any]]:
        """Fetch coupons from RetailMeNot API."""
        # Placeholder - would need actual RetailMeNot API integration
        try:
            logger.debug(f"Fetching RetailMeNot coupons for {retailer}")
            return []
        except Exception as e:
            logger.error(f"Error fetching RetailMeNot coupons: {e}")
            return []
    
    async def _fetch_couponfollow_coupons(self, retailer: str) -> List[Dict[str, Any]]:
        """Fetch coupons from CouponFollow API."""
        # Placeholder - would need actual CouponFollow API integration
        try:
            logger.debug(f"Fetching CouponFollow coupons for {retailer}")
            return []
        except Exception as e:
            logger.error(f"Error fetching CouponFollow coupons: {e}")
            return []
    
    def calculate_discounted_price(
        self,
        original_price: float,
        coupon: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate price after applying coupon.
        
        Args:
            original_price: Original product price
            coupon: Coupon dictionary with discount information
            
        Returns:
            Dictionary with: discounted_price, savings_amount, savings_percent
        """
        discount_type = coupon.get("discount_type", "percentage")  # "percentage" or "fixed"
        discount_value = coupon.get("discount_value", 0.0)
        
        if discount_type == "percentage":
            savings_amount = original_price * (discount_value / 100)
        else:  # fixed amount
            savings_amount = min(discount_value, original_price)
        
        discounted_price = max(0.0, original_price - savings_amount)
        savings_percent = (savings_amount / original_price * 100) if original_price > 0 else 0.0
        
        return {
            "discounted_price": round(discounted_price, 2),
            "savings_amount": round(savings_amount, 2),
            "savings_percent": round(savings_percent, 1),
            "coupon_code": coupon.get("code", ""),
            "coupon_description": coupon.get("description", "")
        }


# Global coupon service instance
coupon_service = CouponService()
