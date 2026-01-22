"""Load and process documents for RAG - supports both API and local data."""
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from src.rag.api_clients import get_api_client, MockAPIClient
from src.analytics.logger import logger


class DocumentLoader:
    """Load documents from various sources (APIs or local files)."""
    
    def __init__(self, data_dir: str = "data", use_api: bool = True):
        self.data_dir = Path(data_dir)
        self.use_api = use_api
        self.api_client = get_api_client() if use_api else MockAPIClient()
    
    async def load_products(self, query: Optional[str] = None) -> List[Dict[str, Any]]:
        """Load products from API or local file."""
        if self.use_api and query:
            # Fetch from API
            products = await self.api_client.search_products(query, limit=50)
            if products:
                logger.info(f"Loaded {len(products)} products from API")
                return products
        
        # Fallback to local file
        products_file = self.data_dir / "products.json"
        try:
            with open(products_file, "r", encoding="utf-8") as f:
                products = json.load(f)
            logger.info(f"Loaded {len(products)} products from local file")
            return products
        except FileNotFoundError:
            logger.warning(f"Products file not found: {products_file}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing products JSON: {e}")
            return []
    
    def load_reviews(self) -> List[Dict[str, Any]]:
        """Load reviews from JSON file (or from API product data)."""
        reviews_file = self.data_dir / "reviews.json"
        try:
            with open(reviews_file, "r", encoding="utf-8") as f:
                reviews = json.load(f)
            logger.info(f"Loaded {len(reviews)} reviews")
            return reviews
        except FileNotFoundError:
            logger.warning(f"Reviews file not found: {reviews_file}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing reviews JSON: {e}")
            return []
    
    def load_faqs(self) -> List[Dict[str, Any]]:
        """Load FAQs from JSON file."""
        faqs_file = self.data_dir / "faqs.json"
        try:
            with open(faqs_file, "r", encoding="utf-8") as f:
                faqs = json.load(f)
            logger.info(f"Loaded {len(faqs)} FAQs")
            return faqs
        except FileNotFoundError:
            logger.warning(f"FAQs file not found: {faqs_file}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing FAQs JSON: {e}")
            return []
    
    async def get_product_by_id(self, product_id: str) -> Dict[str, Any]:
        """Get a specific product by ID from API or local data."""
        # Try API first
        if self.use_api:
            product = await self.api_client.get_product(product_id)
            if product:
                return product
        
        # Fallback to local data
        products = await self.load_products()
        for product in products:
            if product.get("id") == product_id:
                return product
        return {}
    
    async def fetch_products_from_api(self, query: str, num_results: int = 10) -> List[Dict[str, Any]]:
        """Fetch products from API using search query."""
        if self.use_api:
            products = await self.api_client.search_products(query, limit=num_results)
            if products:
                logger.info(f"Fetched {len(products)} products from API for query: {query}")
                return products
        
        # Fallback to local data if API fails or not configured
        logger.warning(f"API not available, falling back to local products for query: {query}")
        all_products = await self.load_products()
        
        # Simple keyword matching on local products
        query_lower = query.lower()
        matching_products = [
            p for p in all_products
            if query_lower in p.get("name", "").lower() or 
               query_lower in p.get("description", "").lower() or
               query_lower in p.get("category", "").lower()
        ]
        
        return matching_products[:num_results]


# Global document loader (defaults to using API if configured)
document_loader = DocumentLoader(use_api=True)

