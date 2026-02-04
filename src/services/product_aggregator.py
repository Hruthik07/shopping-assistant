"""Product aggregation service that queries multiple sources in parallel."""

import asyncio
from typing import List, Dict, Any, Optional, Set
from src.services.product_data_source import ProductDataSource
from src.services.data_sources import (
    SerperDataSource,
    AmazonDataSource,
    eBayDataSource,
    WalmartDataSource,
    BestBuyDataSource,
)
from src.utils.config import settings
from src.analytics.logger import logger


class ProductAggregator:
    """Aggregates products from multiple data sources."""

    def __init__(self):
        """Initialize data sources."""
        self.sources: List[ProductDataSource] = []

        # Initialize available sources
        if SerperDataSource().is_available():
            self.sources.append(SerperDataSource())
        if AmazonDataSource().is_available():
            self.sources.append(AmazonDataSource())
        if eBayDataSource().is_available():
            self.sources.append(eBayDataSource())
        if WalmartDataSource().is_available():
            self.sources.append(WalmartDataSource())
        if BestBuyDataSource().is_available():
            self.sources.append(BestBuyDataSource())

        # Sort sources by priority
        priority_order = (
            settings.product_source_priority.split(",")
            if hasattr(settings, "product_source_priority") and settings.product_source_priority
            else ["serper"]
        )
        source_priority_map = {
            "serper": "serper_google_shopping",
            "amazon": "amazon",
            "ebay": "ebay",
            "walmart": "walmart",
            "bestbuy": "bestbuy",
        }

        def get_priority(source: ProductDataSource) -> int:
            source_name = source.get_source_name()
            for idx, priority in enumerate(priority_order):
                if source_priority_map.get(priority.strip(), "") == source_name:
                    return idx
            return 999  # Lowest priority if not in list

        self.sources.sort(key=get_priority)
        logger.info(
            f"Initialized ProductAggregator with {len(self.sources)} data sources: {[s.get_source_name() for s in self.sources]}"
        )

    async def search_products(
        self,
        query: str,
        num_results: int = 10,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search products from all available sources in parallel.

        Args:
            query: Search query
            num_results: Maximum number of results
            min_price: Minimum price filter
            max_price: Maximum price filter
            category: Category filter

        Returns:
            List of normalized, deduplicated products
        """
        if not self.sources:
            logger.warning("No product data sources available")
            return []

        # Query all sources in parallel with error handling
        tasks = [
            source.search_products(
                query=query,
                num_results=num_results * 2,  # Get more from each source for better deduplication
                min_price=min_price,
                max_price=max_price,
                category=category,
            )
            for source in self.sources
        ]

        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Combine results from all sources with error handling
            all_products = []
            successful_sources = 0
            for idx, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.warning(
                        f"Source {self.sources[idx].get_source_name()} failed: {result}",
                        exc_info=isinstance(result, Exception),
                    )
                    continue
                if isinstance(result, list):
                    all_products.extend(result)
                    successful_sources += 1
                else:
                    logger.warning(
                        f"Source {self.sources[idx].get_source_name()} returned unexpected type: {type(result)}"
                    )

            if successful_sources == 0:
                logger.error("All data sources failed, returning empty results")
                return []

            logger.info(
                f"Successfully queried {successful_sources}/{len(self.sources)} data sources"
            )

            # Track API call metrics
            from src.analytics.tracker import tracker

            tracker.track_api_call(success=True)
            if successful_sources < len(self.sources):
                tracker.track_api_call(success=False)  # Track failed sources

            # Deduplicate and merge products
            try:
                merged_products = self._deduplicate_and_merge(all_products)
            except Exception as e:
                logger.error(f"Error deduplicating products: {e}", exc_info=True)
                # Fallback: return products without deduplication
                merged_products = all_products

            # Apply filters
            try:
                filtered_products = self._apply_filters(
                    merged_products, min_price, max_price, category
                )
            except Exception as e:
                logger.error(f"Error filtering products: {e}", exc_info=True)
                # Fallback: return unfiltered products
                filtered_products = merged_products

            # Return top N results
            return filtered_products[:num_results]

        except Exception as e:
            logger.error(f"Error aggregating products: {e}", exc_info=True)
            return []

    def _deduplicate_and_merge(self, products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate products by UPC/GTIN/EAN and merge information from multiple sources.

        Products with the same identifier are merged, with information from multiple
        retailers combined into a single product entry with multiple retailer options.
        """
        # Group products by identifier
        product_groups: Dict[str, List[Dict[str, Any]]] = {}

        for product in products:
            # Try to find identifier (UPC, GTIN, EAN, or SKU)
            identifier = (
                product.get("upc")
                or product.get("gtin")
                or product.get("ean")
                or product.get("sku")
            )

            if identifier:
                if identifier not in product_groups:
                    product_groups[identifier] = []
                product_groups[identifier].append(product)
            else:
                # Products without identifiers are kept separate (use name+retailer as key)
                name_key = f"{product.get('name', '')}_{product.get('retailer', '')}"
                if name_key not in product_groups:
                    product_groups[name_key] = []
                product_groups[name_key].append(product)

        # Merge products in each group
        merged_products = []
        for identifier, group in product_groups.items():
            if len(group) == 1:
                merged_products.append(group[0])
            else:
                # Merge multiple sources for same product
                merged = self._merge_product_group(group)
                merged_products.append(merged)

        return merged_products

    def _merge_product_group(self, products: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge multiple product entries (from different retailers) into one.

        Creates a product with:
        - Best available information (name, description, images)
        - Multiple retailer options with prices
        - Best price highlighted
        """
        if not products:
            return {}

        # Use first product as base
        base = products[0].copy()

        # Collect retailer options
        retailer_options = []
        best_price = float("inf")
        best_price_retailer = None

        for product in products:
            price = product.get("price", 0.0)
            shipping = product.get("shipping_cost", 0.0)
            total_cost = price + shipping

            retailer_option = {
                "retailer": product.get("retailer", "Unknown"),
                "price": price,
                "currency": product.get("currency", "USD"),
                "shipping_cost": shipping,
                "total_cost": total_cost,
                "product_url": product.get("product_url", ""),
                "availability": product.get("availability", True),
                "in_stock": product.get("in_stock", True),
                "source": product.get("source", ""),
            }
            retailer_options.append(retailer_option)

            # Track best price
            if total_cost < best_price:
                best_price = total_cost
                best_price_retailer = product.get("retailer")

        # Update base product with merged information
        base["retailer_options"] = retailer_options
        base["best_price"] = best_price
        base["best_price_retailer"] = best_price_retailer
        base["price"] = best_price  # Use best price as primary price
        base["retailer"] = best_price_retailer or base.get("retailer", "Multiple")

        # Merge other fields (use best available)
        for product in products[1:]:
            if not base.get("description") and product.get("description"):
                base["description"] = product["description"]
            if not base.get("image_url") and product.get("image_url"):
                base["image_url"] = product["image_url"]
            if not base.get("rating") and product.get("rating"):
                base["rating"] = product["rating"]
            if not base.get("reviews") and product.get("reviews"):
                base["reviews"] = product["reviews"]
            if not base.get("brand") and product.get("brand"):
                base["brand"] = product["brand"]

        return base

    def _apply_filters(
        self,
        products: List[Dict[str, Any]],
        min_price: Optional[float],
        max_price: Optional[float],
        category: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Apply price and category filters."""
        filtered = products

        if min_price is not None:
            filtered = [p for p in filtered if p.get("price", 0) >= min_price]

        if max_price is not None:
            filtered = [p for p in filtered if p.get("price", 0) <= max_price]

        if category:
            category_lower = category.lower()
            filtered = [p for p in filtered if category_lower in p.get("category", "").lower()]

        return filtered


# Global aggregator instance
product_aggregator = ProductAggregator()
