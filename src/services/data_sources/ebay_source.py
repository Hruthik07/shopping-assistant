"""eBay Finding API data source."""
import httpx
from typing import List, Dict, Any, Optional
from src.services.product_data_source import ProductDataSource
from src.utils.config import settings
from src.analytics.logger import logger


class eBayDataSource(ProductDataSource):
    """eBay Finding API data source."""
    
    def __init__(self):
        self.api_key = settings.ebay_api_key
        # eBay Finding API endpoint
        self.endpoint = "https://svcs.ebay.com/services/search/FindingService/v1"
    
    def get_source_name(self) -> str:
        return "ebay"
    
    def is_available(self) -> bool:
        return bool(self.api_key)
    
    async def search_products(
        self,
        query: str,
        num_results: int = 10,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search products using eBay Finding API."""
        if not self.is_available():
            return []
        
        try:
            params = {
                "OPERATION-NAME": "findItemsAdvanced",
                "SERVICE-VERSION": "1.0.0",
                "SECURITY-APPNAME": self.api_key,
                "RESPONSE-DATA-FORMAT": "JSON",
                "REST-PAYLOAD": "",
                "keywords": query,
                "paginationInput.entriesPerPage": min(num_results, 100)
            }
            
            if min_price:
                params["itemFilter(0).name"] = "MinPrice"
                params["itemFilter(0).value"] = str(min_price)
            
            if max_price:
                params["itemFilter(1).name"] = "MaxPrice"
                params["itemFilter(1).value"] = str(max_price)
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(self.endpoint, params=params)
                response.raise_for_status()
                data = response.json()
                
                products = []
                items = data.get("findItemsAdvancedResponse", [{}])[0].get("searchResult", [{}])[0].get("item", [])
                
                for item in items[:num_results]:
                    normalized = self.normalize_product(item)
                    products.append(normalized)
                
                return products
        except Exception as e:
            logger.error(f"Error fetching from eBay API: {e}")
            return []
    
    def normalize_product(self, product: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize eBay product to standard format."""
        # eBay API returns nested structure
        item_id = product.get("itemId", [""])[0] if isinstance(product.get("itemId"), list) else product.get("itemId", "")
        title = product.get("title", [""])[0] if isinstance(product.get("title"), list) else product.get("title", "")
        
        # Extract price
        selling_status = product.get("sellingStatus", [{}])[0] if isinstance(product.get("sellingStatus"), list) else product.get("sellingStatus", {})
        current_price = selling_status.get("currentPrice", [{}])[0] if isinstance(selling_status.get("currentPrice"), list) else selling_status.get("currentPrice", {})
        price_value = float(current_price.get("__value__", 0)) if isinstance(current_price, dict) else 0.0
        currency = current_price.get("@currencyId", "USD") if isinstance(current_price, dict) else "USD"
        
        # Extract shipping
        shipping_info = product.get("shippingInfo", [{}])[0] if isinstance(product.get("shippingInfo"), list) else product.get("shippingInfo", {})
        shipping_cost = 0.0
        if shipping_info.get("shippingServiceCost", [{}])[0].get("__value__"):
            shipping_cost = float(shipping_info.get("shippingServiceCost", [{}])[0].get("__value__", 0))
        
        # Extract image
        gallery_url = product.get("galleryURL", [""])[0] if isinstance(product.get("galleryURL"), list) else product.get("galleryURL", "")
        view_item_url = product.get("viewItemURL", [""])[0] if isinstance(product.get("viewItemURL"), list) else product.get("viewItemURL", "")
        
        return {
            "id": f"ebay_{item_id}",
            "name": title,
            "description": "",
            "price": price_value,
            "currency": currency,
            "original_price": None,
            "shipping_cost": shipping_cost,
            "image_url": gallery_url,
            "product_url": view_item_url,
            "rating": None,
            "reviews": None,
            "category": product.get("primaryCategoryName", [""])[0] if isinstance(product.get("primaryCategoryName"), list) else "general",
            "brand": None,
            "availability": True,
            "in_stock": True,
            "retailer": "eBay",
            "source": "ebay",
            "upc": None,
            "gtin": None,
            "ean": None,
            "sku": None,
            "metadata": {"itemId": item_id}
        }
