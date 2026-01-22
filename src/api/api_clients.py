"""API clients for fetching real product data."""
from typing import List, Dict, Any, Optional
from src.api.product_fetcher import product_fetcher
from src.analytics.logger import logger


class APIClient:
    """Base API client for product data."""
    
    async def search_products(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Search products."""
        raise NotImplementedError
    
    async def get_product(self, product_id: str) -> Optional[Dict[str, Any]]:
        """Get product by ID."""
        raise NotImplementedError


class GoogleShoppingAPIClient(APIClient):
    """Client for Google Shopping via Serper API."""
    
    async def search_products(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Search products using Google Shopping."""
        products = await product_fetcher.search_products(
            query=query,
            num_results=min(limit, 50),
            use_google_shopping=True,
            use_tavily=False
        )
        return products
    
    async def get_product(self, product_id: str) -> Optional[Dict[str, Any]]:
        """Get product details."""
        # For Google Shopping, we'd need to store the product URL
        # For now, return None as we don't have a direct product ID lookup
        return None


class MockAPIClient(APIClient):
    """Mock client that returns empty results (for testing)."""
    
    async def search_products(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Return empty list."""
        return []
    
    async def get_product(self, product_id: str) -> Optional[Dict[str, Any]]:
        """Return None."""
        return None


def get_api_client() -> APIClient:
    """Get the appropriate API client based on configuration."""
    from src.utils.config import settings
    
    if settings.serper_api_key:
        return GoogleShoppingAPIClient()
    else:
        logger.warning("No API keys configured, using mock client")
        return MockAPIClient()

