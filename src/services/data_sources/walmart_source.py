"""Walmart Open API data source."""
import httpx
from typing import List, Dict, Any, Optional
from src.services.product_data_source import ProductDataSource
from src.utils.config import settings
from src.analytics.logger import logger


class WalmartDataSource(ProductDataSource):
    """Walmart Open API data source."""
    
    def __init__(self):
        self.api_key = settings.walmart_api_key
        self.base_url = "https://api.walmartlabs.com/v1"
    
    def get_source_name(self) -> str:
        return "walmart"
    
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
        """Search products using Walmart Open API."""
        if not self.is_available():
            return []
        
        try:
            params = {
                "apiKey": self.api_key,
                "query": query,
                "format": "json",
                "numItems": min(num_results, 25)  # Walmart API limit
            }
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.base_url}/search", params=params)
                response.raise_for_status()
                data = response.json()
                
                products = []
                items = data.get("items", [])
                
                for item in items[:num_results]:
                    normalized = self.normalize_product(item)
                    products.append(normalized)
                
                return products
        except Exception as e:
            logger.error(f"Error fetching from Walmart API: {e}")
            return []
    
    def normalize_product(self, product: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize Walmart product to standard format."""
        price = product.get("salePrice") or product.get("msrp", 0.0)
        
        return {
            "id": f"walmart_{product.get('itemId', '')}",
            "name": product.get("name", ""),
            "description": product.get("shortDescription", ""),
            "price": float(price),
            "currency": "USD",
            "original_price": product.get("msrp"),
            "shipping_cost": 0.0,  # Walmart provides free shipping on many items
            "image_url": product.get("largeImage", ""),
            "product_url": product.get("productUrl", ""),
            "rating": product.get("customerRating"),
            "reviews": product.get("numReviews", 0),
            "category": product.get("categoryPath", "").split("/")[-1] if product.get("categoryPath") else "general",
            "brand": product.get("brandName"),
            "availability": product.get("availableOnline", True),
            "in_stock": product.get("availableOnline", True),
            "retailer": "Walmart",
            "source": "walmart",
            "upc": product.get("upc"),
            "gtin": None,
            "ean": None,
            "sku": product.get("itemId"),
            "metadata": {"itemId": product.get("itemId", "")}
        }
