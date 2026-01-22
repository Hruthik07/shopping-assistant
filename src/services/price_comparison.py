"""Price comparison engine to compare prices across retailers."""
from typing import List, Dict, Any, Optional
from src.analytics.logger import logger


class PriceComparator:
    """Compares prices across retailers and calculates total cost."""
    
    def compare_prices(self, products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Compare prices for products and identify best deals.
        
        Args:
            products: List of products (may have retailer_options)
            
        Returns:
            Products with price comparison data added
        """
        compared_products = []
        
        for product in products:
            try:
                # If product has retailer_options, compare them
                if "retailer_options" in product:
                    product = self._compare_retailer_options(product)
                else:
                    # Single retailer product - add comparison metadata
                    product = self._add_comparison_metadata(product)
                
                compared_products.append(product)
            except Exception as e:
                logger.error(f"Error comparing prices for product {product.get('id', 'unknown')}: {e}", exc_info=True)
                # Add product without comparison data rather than skipping
                if "price_comparison" not in product:
                    product["price_comparison"] = {
                        "best_price": product.get("price", 0) + product.get("shipping_cost", 0),
                        "retailer_count": 1
                    }
                compared_products.append(product)
        
        # Track price comparison metrics
        if compared_products:
            from src.analytics.tracker import tracker
            products_with_multiple = sum(1 for p in compared_products if p.get("price_comparison", {}).get("retailer_count", 0) > 1)
            avg_retailers = sum(p.get("price_comparison", {}).get("retailer_count", 1) for p in compared_products) / len(compared_products)
            avg_diff = sum(p.get("price_comparison", {}).get("savings", 0) for p in compared_products) / len(compared_products)
            
            tracker.track_price_comparison(
                products_compared=len(compared_products),
                products_with_multiple_retailers=products_with_multiple,
                average_retailers=avg_retailers,
                average_price_diff=avg_diff
            )
        
        return compared_products
    
    def _compare_retailer_options(self, product: Dict[str, Any]) -> Dict[str, Any]:
        """Compare multiple retailer options for the same product."""
        retailer_options = product.get("retailer_options", [])
        
        if not retailer_options:
            return product
        
        # Sort by total cost (price + shipping)
        sorted_options = sorted(
            retailer_options,
            key=lambda x: x.get("total_cost", float('inf'))
        )
        
        best_option = sorted_options[0]
        worst_option = sorted_options[-1]
        
        savings = worst_option.get("total_cost", 0) - best_option.get("total_cost", 0)
        savings_percent = 0.0
        if worst_option.get("total_cost", 0) > 0:
            savings_percent = (savings / worst_option.get("total_cost", 0)) * 100
        
        # Add comparison data
        product["price_comparison"] = {
            "best_price": best_option.get("total_cost", 0),
            "best_retailer": best_option.get("retailer", ""),
            "worst_price": worst_option.get("total_cost", 0),
            "worst_retailer": worst_option.get("retailer", ""),
            "savings": savings,
            "savings_percent": round(savings_percent, 1),
            "retailer_count": len(retailer_options),
            "price_range": {
                "min": best_option.get("total_cost", 0),
                "max": worst_option.get("total_cost", 0)
            }
        }
        
        # Update primary price to best price
        product["price"] = best_option.get("total_cost", 0)
        product["retailer"] = best_option.get("retailer", "")
        product["shipping_cost"] = best_option.get("shipping_cost", 0)
        
        return product
    
    def _add_comparison_metadata(self, product: Dict[str, Any]) -> Dict[str, Any]:
        """Add comparison metadata for single-retailer products."""
        price = product.get("price", 0.0)
        shipping = product.get("shipping_cost", 0.0)
        total_cost = price + shipping
        
        product["price_comparison"] = {
            "best_price": total_cost,
            "best_retailer": product.get("retailer", ""),
            "retailer_count": 1,
            "price_range": {
                "min": total_cost,
                "max": total_cost
            }
        }
        
        return product
    
    def calculate_total_cost(
        self,
        price: float,
        shipping_cost: float = 0.0,
        tax: float = 0.0,
        discount: float = 0.0
    ) -> float:
        """Calculate total cost including all fees and discounts.
        
        Args:
            price: Base price
            shipping_cost: Shipping cost
            tax: Tax amount
            discount: Discount amount
            
        Returns:
            Total cost after all fees and discounts
        """
        return max(0.0, price + shipping_cost + tax - discount)
    
    def rank_by_customer_value(
        self,
        products: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Rank products by customer value (lowest total cost first).
        
        This is a simple customer-first ranking that prioritizes:
        1. Lowest total cost (price + shipping)
        2. Higher ratings (if same price)
        3. More reviews (if same price and rating)
        """
        def get_sort_key(product: Dict[str, Any]) -> tuple:
            price = product.get("price", 0.0)
            shipping = product.get("shipping_cost", 0.0)
            total_cost = price + shipping
            
            # Negative rating and reviews for descending sort
            rating = product.get("rating", 0) or 0
            reviews = product.get("reviews", 0) or 0
            
            return (total_cost, -rating, -reviews)
        
        sorted_products = sorted(products, key=get_sort_key)
        return sorted_products


# Global price comparator instance
price_comparator = PriceComparator()
