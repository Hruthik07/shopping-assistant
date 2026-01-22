"""Fetch real product data from external APIs."""
import httpx
from typing import List, Dict, Any, Optional
from src.utils.config import settings
from src.analytics.logger import logger
from src.utils.debug_log import file_debug_log
from src.analytics.latency_tracker import latency_tracker
from src.utils.retry import async_retry, RetryConfig, http_retry
from src.utils.circuit_breaker import async_circuit_breaker, CircuitBreakerConfig, http_circuit_breaker
from src.utils.cache import cache_service


class ProductFetcher:
    """Fetch products from real e-commerce APIs."""
    
    def __init__(self):
        self.serper_api_key = settings.serper_api_key
        self.etsy_api_key = settings.etsy_api_key
    
    async def _fetch_from_serper_api(
        self,
        query: str,
        num_results: int
    ) -> List[Dict[str, Any]]:
        """Internal method to fetch from Serper API (with retry)."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://google.serper.dev/shopping",
                headers={
                    "X-API-KEY": self.serper_api_key,
                    "Content-Type": "application/json"
                },
                json={
                    "q": query,
                    "num": num_results
                },
                timeout=10.0
            )
            response.raise_for_status()
            return response.json()
    
    async def _fallback_to_cache(
        self,
        query: str,
        num_results: int
    ) -> List[Dict[str, Any]]:
        """Fallback to cached products if API fails."""
        logger.info(f"Falling back to cache for query: {query}")
        cached_products = await cache_service.get_product_search(
            query=query,
            max_results=num_results
        )
        if cached_products:
            logger.info(f"Found {len(cached_products)} cached products")
            return cached_products
        logger.warning("No cached products available for fallback")
        return []
    
    @async_circuit_breaker(
        config=CircuitBreakerConfig(
            failure_threshold=5,
            recovery_timeout=30,
            name="serper_api"
        ),
        fallback=None  # Will use method-level fallback
    )
    @async_retry(
        config=RetryConfig(
            max_attempts=3,
            initial_wait=1.0,
            max_wait=10.0
        ),
        fallback=None  # Will use method-level fallback
    )
    async def search_products_google_shopping(
        self,
        query: str,
        num_results: int = 10
    ) -> List[Dict[str, Any]]:
        """Search products using Google Shopping via Serper API with retry and circuit breaker."""
        # #region debug instrumentation
        try:
            file_debug_log(
                "product_fetcher.py:74",
                "search_products_google_shopping called (with decorators)",
                {"query": query[:50], "num_results": num_results},
                hypothesis_id="J",
            )
        except Exception:
            pass
        # #endregion
        if not self.serper_api_key:
            logger.warning("Serper API key not configured, skipping Google Shopping search")
            return await self._fallback_to_cache(query, num_results)
        
        try:
            # Track API call latency
            with latency_tracker.track_component("serper_api_call"):
                data = await self._fetch_from_serper_api(query, num_results)
            
            # Track response parsing
            with latency_tracker.track_component("api_response_parsing"):
                products = []
                for item in data.get("shopping", []):
                    products.append({
                        "id": f"google_{item.get('productId', '')}",
                        "name": item.get("title", ""),
                        "description": item.get("description", ""),
                        "price": self._parse_price(item.get("price", "")),
                        "currency": item.get("currency", "USD"),
                        "image_url": item.get("imageUrl", ""),
                        "product_url": item.get("link", ""),
                        "rating": item.get("rating", 0),
                        "reviews": item.get("reviews", 0),
                        "category": item.get("category", "general"),
                        "availability": True,
                        "source": "google_shopping"
                    })
            
            logger.info(f"Fetched {len(products)} products from Google Shopping for: {query}")
            return products
                
        except Exception as e:
            logger.error(f"Error fetching products from Google Shopping: {e}")
            # Fallback to cache
            return await self._fallback_to_cache(query, num_results)
    
    async def search_products_tavily(
        self,
        query: str,
        num_results: int = 5
    ) -> List[Dict[str, Any]]:
        """Search product information using Tavily web search."""
        if not settings.tavily_api_key:
            logger.warning("Tavily API key not configured")
            return []
        
        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=settings.tavily_api_key)
            
            response = client.search(
                query=f"{query} product review price",
                max_results=num_results,
                search_depth="advanced"
            )
            
            products = []
            for result in response.get("results", []):
                # Extract product info from search results
                products.append({
                    "id": f"tavily_{hash(result.get('url', ''))}",
                    "name": result.get("title", query),
                    "description": result.get("content", "")[:200],
                    "price": None,  # Tavily doesn't provide direct pricing
                    "image_url": None,
                    "product_url": result.get("url", ""),
                    "category": "general",
                    "availability": True,
                    "source": "tavily_web"
                })
            
            logger.info(f"Fetched {len(products)} results from Tavily for: {query}")
            return products
            
        except Exception as e:
            logger.error(f"Error fetching from Tavily: {e}")
            return []
    
    async def get_product_details(
        self,
        product_url: str,
        source: str = "google_shopping"
    ) -> Optional[Dict[str, Any]]:
        """Get detailed product information from URL."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(product_url, timeout=10.0, follow_redirects=True)
                # In a real implementation, you'd parse the HTML
                # For now, return basic info
                return {
                    "url": product_url,
                    "source": source,
                    "available": True
                }
        except Exception as e:
            logger.error(f"Error fetching product details: {e}")
            return None
    
    def _parse_price(self, price_str: str) -> float:
        """Parse price string to float."""
        try:
            # Remove currency symbols and commas
            cleaned = price_str.replace("$", "").replace(",", "").strip()
            return float(cleaned)
        except (ValueError, AttributeError):
            return 0.0
    
    async def search_products(
        self,
        query: str,
        num_results: int = 10,
        use_google_shopping: bool = True,
        use_tavily: bool = False
    ) -> List[Dict[str, Any]]:
        """Search products from multiple sources."""
        all_products = []
        
        if use_google_shopping:
            google_products = await self.search_products_google_shopping(query, num_results)
            all_products.extend(google_products)
        
        if use_tavily and len(all_products) < num_results:
            tavily_products = await self.search_products_tavily(
                query,
                num_results - len(all_products)
            )
            all_products.extend(tavily_products)
        
        # Remove duplicates and return top N
        seen_ids = set()
        unique_products = []
        for product in all_products:
            if product["id"] not in seen_ids:
                seen_ids.add(product["id"])
                unique_products.append(product)
                if len(unique_products) >= num_results:
                    break
        
        return unique_products


# Global product fetcher
product_fetcher = ProductFetcher()



