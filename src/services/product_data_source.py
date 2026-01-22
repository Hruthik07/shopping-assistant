"""Abstract base class for product data sources."""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class ProductDataSource(ABC):
    """Abstract interface for product data sources."""
    
    @abstractmethod
    async def search_products(
        self,
        query: str,
        num_results: int = 10,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search for products.
        
        Args:
            query: Search query
            num_results: Maximum number of results
            min_price: Minimum price filter
            max_price: Maximum price filter
            category: Category filter
            
        Returns:
            List of normalized product dictionaries
        """
        pass
    
    @abstractmethod
    def get_source_name(self) -> str:
        """Get the name of this data source."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this data source is available (API keys configured)."""
        pass
    
    def normalize_product(self, product: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize product data to standard format.
        
        Standard format:
        {
            "id": str,  # Unique product identifier
            "name": str,
            "description": str,
            "price": float,
            "currency": str,  # ISO currency code (USD, EUR, etc.)
            "original_price": Optional[float],  # Price before discount
            "shipping_cost": float,
            "image_url": Optional[str],
            "product_url": str,
            "rating": Optional[float],  # 0-5 scale
            "reviews": Optional[int],
            "category": str,
            "brand": Optional[str],
            "availability": bool,
            "in_stock": bool,
            "retailer": str,  # Retailer name
            "source": str,  # Data source name
            "upc": Optional[str],  # Universal Product Code
            "gtin": Optional[str],  # Global Trade Item Number
            "ean": Optional[str],  # European Article Number
            "sku": Optional[str],  # Stock Keeping Unit
            "metadata": Dict[str, Any]  # Additional source-specific data
        }
        """
        return product
