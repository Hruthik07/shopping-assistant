"""Best Buy API data source."""

import httpx
from typing import List, Dict, Any, Optional
from src.services.product_data_source import ProductDataSource
from src.utils.config import settings
from src.analytics.logger import logger


class BestBuyDataSource(ProductDataSource):
    """Best Buy API data source."""

    def __init__(self):
        self.api_key = settings.bestbuy_api_key
        self.base_url = "https://api.bestbuy.com/v1"

    def get_source_name(self) -> str:
        return "bestbuy"

    def is_available(self) -> bool:
        return bool(self.api_key)

    async def search_products(
        self,
        query: str,
        num_results: int = 10,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search products using Best Buy API."""
        if not self.is_available():
            return []

        try:
            # Build search query
            search_query = f"search={query}"
            if min_price:
                search_query += f"&salePrice>={min_price}"
            if max_price:
                search_query += f"&salePrice<={max_price}"

            params = {
                "apiKey": self.api_key,
                "format": "json",
                "pageSize": min(num_results, 30),
                "show": "sku,name,salePrice,regularPrice,image,url,customerReviewAverage,customerReviewCount,categoryPath,manufacturer,upc",
            }

            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{self.base_url}/products({search_query})", params=params
                )
                response.raise_for_status()
                data = response.json()

                products = []
                items = data.get("products", [])

                for item in items[:num_results]:
                    normalized = self.normalize_product(item)
                    products.append(normalized)

                return products
        except Exception as e:
            logger.error(f"Error fetching from Best Buy API: {e}")
            return []

    def normalize_product(self, product: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize Best Buy product to standard format."""
        return {
            "id": f"bestbuy_{product.get('sku', '')}",
            "name": product.get("name", ""),
            "description": "",
            "price": product.get("salePrice", 0.0),
            "currency": "USD",
            "original_price": product.get("regularPrice"),
            "shipping_cost": 0.0,  # Best Buy provides shipping info separately
            "image_url": product.get("image", ""),
            "product_url": product.get("url", ""),
            "rating": product.get("customerReviewAverage", 0),
            "reviews": product.get("customerReviewCount", 0),
            "category": (
                product.get("categoryPath", [{}])[-1].get("name", "general")
                if isinstance(product.get("categoryPath"), list)
                else "general"
            ),
            "brand": product.get("manufacturer"),
            "availability": True,
            "in_stock": True,
            "retailer": "Best Buy",
            "source": "bestbuy",
            "upc": product.get("upc"),
            "gtin": None,
            "ean": None,
            "sku": product.get("sku"),
            "metadata": {"sku": product.get("sku", "")},
        }
