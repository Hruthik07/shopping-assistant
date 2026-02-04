"""Product-related MCP tools."""

import re
import time
import json as json_module
from typing import Dict, Any, Optional, Tuple
from src.mcp.mcp_client import MCPTool, tool_registry
from src.data.document_loader import document_loader
from src.api.product_fetcher import product_fetcher
from src.analytics.logger import logger
from src.utils.config import settings
from src.utils.debug_log import file_debug_log


def extract_price_range(query: str) -> Optional[Tuple[float, float]]:
    """Extract price range from query string.

    Examples:
        "between 50 to 90 $" -> (50.0, 90.0)
        "50-90 dollars" -> (50.0, 90.0)
        "under 100" -> (0.0, 100.0)
        "over 50" -> (50.0, None)
    """
    # #region debug instrumentation
    try:
        file_debug_log(
            "product_tools.py:19",
            "extract_price_range called",
            {"query": query},
            hypothesis_id="A",
        )
    except Exception:
        pass
    # #endregion
    query_lower = query.lower()

    # Pattern: "between X to Y" or "between X and Y"
    match = re.search(
        r"between\s+\$?(\d+(?:\.\d+)?)\s+(?:to|and)\s+\$?(\d+(?:\.\d+)?)", query_lower
    )
    if match:
        min_price = float(match.group(1))
        max_price = float(match.group(2))
        # #region debug instrumentation
        try:
            file_debug_log(
                "product_tools.py:26",
                "extract_price_range matched pattern",
                {"min_price": min_price, "max_price": max_price, "match_groups": match.groups()},
                hypothesis_id="A",
            )
        except Exception:
            pass
        # #endregion
        return (min_price, max_price)

    # Pattern: "X to Y" or "X-Y" or "X - Y"
    match = re.search(r"\$?(\d+(?:\.\d+)?)\s*[-–—]\s*\$?(\d+(?:\.\d+)?)", query_lower)
    if match:
        min_price = float(match.group(1))
        max_price = float(match.group(2))
        return (min_price, max_price)

    # Pattern: "under X" or "below X" or "less than X"
    match = re.search(r"(?:under|below|less than)\s+\$?(\d+(?:\.\d+)?)", query_lower)
    if match:
        max_price = float(match.group(1))
        return (0.0, max_price)

    # Pattern: "over X" or "above X" or "more than X"
    match = re.search(r"(?:over|above|more than)\s+\$?(\d+(?:\.\d+)?)", query_lower)
    if match:
        min_price = float(match.group(1))
        return (min_price, None)

    # #region debug instrumentation
    try:
        file_debug_log(
            "product_tools.py:47",
            "extract_price_range no match",
            {"query_lower": query_lower},
            hypothesis_id="A",
        )
    except Exception:
        pass
    # #endregion
    return None


class ProductAvailabilityTool(MCPTool):
    """Check product availability."""

    def __init__(self):
        super().__init__(
            name="check_product_availability",
            description="Check if a product is available and in stock. Returns availability status and stock count.",
        )

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "product_id": {"type": "string", "description": "The product ID to check"}
            },
            "required": ["product_id"],
        }

    async def execute(self, product_id: str) -> Dict[str, Any]:
        """Check product availability."""
        try:
            product = await document_loader.get_product_by_id(product_id)

            if not product:
                # Try fetching from API
                return {
                    "product_id": product_id,
                    "available": False,
                    "message": "Product not found",
                }

            availability = product.get("availability", True)
            stock = product.get("stock", 0)

            return {
                "product_id": product_id,
                "product_name": product.get("name", ""),
                "available": availability and stock > 0,
                "stock": stock,
                "message": f"Product is {'available' if availability and stock > 0 else 'out of stock'}",
            }
        except Exception as e:
            logger.error(f"Error checking availability: {e}")
            return {"product_id": product_id, "available": False, "error": str(e)}


class PriceCheckTool(MCPTool):
    """Check and compare product prices."""

    def __init__(self):
        super().__init__(
            name="check_product_price",
            description="Get the current price of a product. Can also compare prices across variants.",
        )

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "product_id": {"type": "string", "description": "The product ID to check price for"}
            },
            "required": ["product_id"],
        }

    async def execute(self, product_id: str) -> Dict[str, Any]:
        """Check product price."""
        try:
            product = await document_loader.get_product_by_id(product_id)

            if not product:
                return {
                    "product_id": product_id,
                    "price": None,
                    "currency": "USD",
                    "message": "Product not found",
                }

            return {
                "product_id": product_id,
                "product_name": product.get("name", ""),
                "price": product.get("price", 0),
                "currency": product.get("currency", "USD"),
                "original_price": product.get("original_price"),
                "discount": product.get("discount"),
                "message": f"Price: ${product.get('price', 0):.2f}",
            }
        except Exception as e:
            logger.error(f"Error checking price: {e}")
            return {"product_id": product_id, "price": None, "error": str(e)}


class ProductSearchTool(MCPTool):
    """Search for products."""

    def __init__(self):
        super().__init__(
            name="search_products",
            description="Search for products by query. Returns list of matching products with details.",
        )

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query for products"},
                "category": {"type": "string", "description": "Filter by category (optional)"},
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results (default: 10)",
                    "default": 10,
                },
                "min_price": {"type": "number", "description": "Minimum price filter (optional)"},
                "max_price": {"type": "number", "description": "Maximum price filter (optional)"},
            },
            "required": ["query"],
        }

    async def execute(  # noqa: C901
        self,
        query: str,
        category: Optional[str] = None,
        max_results: int = 10,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Search for products."""
        # #region debug instrumentation
        try:
            file_debug_log(
                "product_tools.py:240",
                "ProductSearchTool.execute called",
                {
                    "query": query[:50],
                    "category": category,
                    "max_results": max_results,
                    "min_price": min_price,
                    "max_price": max_price,
                },
                hypothesis_id="H",
            )
        except Exception as e:
            # Log the error instead of silently passing
            try:
                file_debug_log(
                    "product_tools.py:240",
                    "ProductSearchTool.execute instrumentation failed",
                    {"error": str(e), "error_type": type(e).__name__},
                    hypothesis_id="H",
                )
            except Exception:
                pass
        # #endregion
        try:
            # Check cache first (skip in semantic-only mode; query-string caches are term-based)
            from src.utils.cache import cache_service

            # #region debug instrumentation
            try:
                file_debug_log(
                    "product_tools.py:252",
                    "Calling cache_service.get_product_search",
                    {
                        "query": query[:50],
                        "cache_enabled": cache_service.enabled,
                        "has_client": cache_service.redis_client is not None,
                    },
                    hypothesis_id="H",
                )
            except Exception:
                pass
            # #endregion
            cached_products = None
            if not getattr(settings, "semantic_only_retrieval", False):
                cached_products = await cache_service.get_product_search(
                    query=query,
                    category=category,
                    min_price=min_price,
                    max_price=max_price,
                    max_results=max_results,
                )
            # #region debug instrumentation
            try:
                file_debug_log(
                    "product_tools.py:265",
                    "cache_service.get_product_search result",
                    {"query": query[:50], "has_cached": cached_products is not None},
                    hypothesis_id="H",
                )
            except Exception:
                pass
            # #endregion
            if cached_products is not None:
                logger.info(
                    f"Cache hit for product search: {query}, enriching with deal features..."
                )
                # Even cached products need deal detection, price comparison, etc.
                # Check if products already have deal features
                needs_enrichment = not any(
                    p.get("deal_info") or p.get("price_comparison") or p.get("coupon_info")
                    for p in cached_products[:3]  # Check first few products
                )

                if needs_enrichment:
                    logger.info("Cached products missing deal features, enriching...")
                    # Import services
                    from src.services.price_comparison import price_comparator
                    from src.services.deal_detector import deal_detector
                    from src.services.promo_matcher import promo_matcher
                    from src.services.ranking_service import ranking_service

                    # Apply deal features to cached products with error handling
                    try:
                        cached_products = price_comparator.compare_prices(cached_products)
                    except Exception as e:
                        logger.error(
                            f"Error enriching cached products with price comparison: {e}",
                            exc_info=True,
                        )

                    try:
                        cached_products = await deal_detector.detect_deals(cached_products)
                    except Exception as e:
                        logger.error(
                            f"Error enriching cached products with deal detection: {e}",
                            exc_info=True,
                        )

                    try:
                        cached_products = await promo_matcher.match_promos_to_products(
                            cached_products
                        )
                    except Exception as e:
                        logger.error(
                            f"Error enriching cached products with promo matching: {e}",
                            exc_info=True,
                        )

                    try:
                        cached_products = ranking_service.rank_products(
                            cached_products, sort_by="customer_value"
                        )
                    except Exception as e:
                        logger.error(
                            f"Error enriching cached products with ranking: {e}", exc_info=True
                        )

                    cached_products = cached_products[:max_results]

                # Log final cached product structure for debugging
                if cached_products:
                    sample_product = cached_products[0]
                    logger.info(f"Cached product structure - Keys: {list(sample_product.keys())}")
                    logger.info(f"Has deal_info: {'deal_info' in sample_product}")
                    logger.info(f"Has price_comparison: {'price_comparison' in sample_product}")
                    logger.info(f"Has coupon_info: {'coupon_info' in sample_product}")

                return {
                    "query": query,
                    "category": category,
                    "min_price": min_price,
                    "max_price": max_price,
                    "results_count": len(cached_products),
                    "products": cached_products,  # Return products with all fields intact
                    "message": f"Found {len(cached_products)} products (cached)",
                    "cached": True,
                }

            # Extract price range from query if not explicitly provided
            if min_price is None and max_price is None:
                price_range = extract_price_range(query)
                if price_range:
                    min_price, max_price = price_range
                    max_str = f"${max_price:.2f}" if max_price else "inf"
                    logger.info(f"Extracted price range from query: ${min_price:.2f} - {max_str}")

            # Use product aggregator to fetch from multiple sources
            from src.services.product_aggregator import product_aggregator
            from src.services.price_comparison import price_comparator
            from src.services.deal_detector import deal_detector
            from src.services.promo_matcher import promo_matcher
            from src.services.ranking_service import ranking_service

            # Fetch from multiple sources in parallel
            fetch_count = (
                max_results * 3
                if (min_price is not None or max_price is not None)
                else max_results * 2
            )
            products = await product_aggregator.search_products(
                query=query,
                num_results=fetch_count,
                min_price=min_price,
                max_price=max_price,
                category=category,
            )

            if not products:
                logger.warning(f"No products found for query: {query}")
                return {
                    "query": query,
                    "category": category,
                    "min_price": min_price,
                    "max_price": max_price,
                    "results_count": 0,
                    "products": [],
                    "message": "No products found",
                    "cached": False,
                }

            # Apply semantic search re-ranking (for relevance)
            try:
                from src.utils.semantic_search import get_semantic_searcher

                semantic_searcher = get_semantic_searcher()
                logger.info(f"Re-ranking {len(products)} products using semantic search")
                products = await semantic_searcher.rerank_products(
                    query, products, top_k=fetch_count
                )
                logger.info(f"Semantic re-ranking completed, {len(products)} products ranked")
            except Exception as e:
                logger.warning(f"Semantic search failed, using original order: {e}")

            # Compare prices across retailers
            try:
                logger.info(f"Comparing prices for {len(products)} products")
                products = price_comparator.compare_prices(products)
                if products:
                    has_price_comp = "price_comparison" in products[0]
                    logger.info(
                        f"After price comparison: {len(products)} products, first product has price_comparison: {has_price_comp}"
                    )
                    if not has_price_comp:
                        logger.warning(
                            "Price comparison service did not add price_comparison field"
                        )
            except Exception as e:
                logger.error(f"Error in price comparison: {e}", exc_info=True)
                # Continue with products even if price comparison fails

            # Detect deals and price drops
            try:
                logger.info(f"Detecting deals for {len(products)} products")
                products = await deal_detector.detect_deals(products)
                if products:
                    has_deal_info = "deal_info" in products[0]
                    logger.info(
                        f"After deal detection: {len(products)} products, first product has deal_info: {has_deal_info}"
                    )
                    if not has_deal_info:
                        logger.warning("Deal detector service did not add deal_info field")
            except Exception as e:
                logger.error(f"Error in deal detection: {e}", exc_info=True)
                # Continue with products even if deal detection fails

            # Match coupons/promo codes
            try:
                logger.info(f"Matching promos for {len(products)} products")
                products = await promo_matcher.match_promos_to_products(products)
                if products:
                    has_coupon_info = "coupon_info" in products[0]
                    logger.info(
                        f"After promo matching: {len(products)} products, first product has coupon_info: {has_coupon_info}"
                    )
                    if not has_coupon_info:
                        logger.warning("Promo matcher service did not add coupon_info field")
            except Exception as e:
                logger.error(f"Error in promo matching: {e}", exc_info=True)
                # Continue with products even if promo matching fails

            # Apply customer-first ranking
            try:
                logger.info(f"Ranking {len(products)} products")
                products = ranking_service.rank_products(products, sort_by="customer_value")
                if products:
                    has_customer_value = "customer_value" in products[0]
                    logger.info(
                        f"After ranking: {len(products)} products, first product has customer_value: {has_customer_value}"
                    )
                    if not has_customer_value:
                        logger.warning("Ranking service did not add customer_value field")
            except Exception as e:
                logger.error(f"Error in ranking: {e}", exc_info=True)
                # Continue with products even if ranking fails

            # Limit to requested number of results
            products = products[:max_results]

            # Cache the results (if not semantic-only mode)
            if not getattr(settings, "semantic_only_retrieval", False) and products:
                try:
                    await cache_service.set_product_search(
                        query=query,
                        products=products,
                        category=category,
                        min_price=min_price,
                        max_price=max_price,
                        max_results=max_results,
                    )
                except Exception as e:
                    logger.debug(f"Failed to cache product search: {e}")

            # Log final product structure for debugging
            if products:
                sample_product = products[0]
                logger.info(f"Final product structure - Keys: {list(sample_product.keys())}")
                logger.info(f"Has deal_info: {'deal_info' in sample_product}")
                logger.info(f"Has price_comparison: {'price_comparison' in sample_product}")
                logger.info(f"Has coupon_info: {'coupon_info' in sample_product}")
                logger.info(f"Has customer_value: {'customer_value' in sample_product}")

            return {
                "query": query,
                "category": category,
                "min_price": min_price,
                "max_price": max_price,
                "results_count": len(products),
                "products": products,  # Return products with all fields intact
                "message": f"Found {len(products)} products with best deals",
                "cached": False,
            }

            # Legacy price filtering code (removed - handled by aggregator)
            # Keeping for reference but not executed
            if False and (min_price is not None or max_price is not None):
                # #region debug instrumentation
                try:
                    file_debug_log(
                        "product_tools.py:220",
                        "Starting price filtering",
                        {
                            "min_price": min_price,
                            "max_price": max_price,
                            "products_before_filter": len(products),
                            "sample_prices": [p.get("price", "N/A") for p in products[:5]],
                        },
                        hypothesis_id="D",
                    )
                except Exception:
                    pass
                # #endregion
                filtered_products = []
                skipped_count = 0
                for p in products:
                    price = p.get("price", 0)
                    # Handle different price formats
                    if isinstance(price, str):
                        # Extract numeric value from string like "$50.00" or "50.00 USD"
                        price_match = re.search(r"(\d+(?:\.\d+)?)", str(price))
                        if price_match:
                            price = float(price_match.group(1))
                        else:
                            price = 0.0
                            skipped_count += 1
                            continue
                    elif isinstance(price, (int, float)):
                        price = float(price)
                    else:
                        price = 0.0
                        skipped_count += 1
                        continue

                    # Check price range
                    if min_price is not None and price < min_price:
                        skipped_count += 1
                        continue
                    if max_price is not None and price > max_price:
                        skipped_count += 1
                        continue

                    filtered_products.append(p)

                min_str = f"${min_price:.2f}" if min_price else "$0"
                max_str = f"${max_price:.2f}" if max_price else "inf"
                logger.info(
                    f"Price filtering: {len(products)} products -> {len(filtered_products)} products (skipped {skipped_count}) in range {min_str}-{max_str}"
                )
                # #region debug instrumentation
                try:
                    file_debug_log(
                        "product_tools.py:252",
                        "Price filtering complete",
                        {
                            "before_count": len(products),
                            "after_count": len(filtered_products),
                            "skipped": skipped_count,
                            "filtered_sample_prices": [
                                p.get("price", "N/A") for p in filtered_products[:5]
                            ],
                        },
                        hypothesis_id="D",
                    )
                except Exception:
                    pass
                # #endregion
                products = filtered_products

            # Cache the results
            final_products = products[:max_results]
            if getattr(settings, "semantic_only_retrieval", False):
                # Do not store query-string keyed cache in semantic-only mode.
                return {
                    "query": query,
                    "category": category,
                    "min_price": min_price,
                    "max_price": max_price,
                    "results_count": len(final_products),
                    "products": final_products,
                    "message": f"Found {len(final_products)} products",
                    "cached": False,
                }
            # #region debug instrumentation
            try:
                file_debug_log(
                    "product_tools.py:357",
                    "Calling cache_service.set_product_search",
                    {
                        "query": query[:50],
                        "cache_enabled": cache_service.enabled,
                        "has_client": cache_service.redis_client is not None,
                        "products_count": len(final_products),
                    },
                    hypothesis_id="H",
                )
            except Exception:
                pass
            # #endregion
            cache_result = await cache_service.set_product_search(
                query=query,
                products=final_products,
                category=category,
                min_price=min_price,
                max_price=max_price,
                max_results=max_results,
                ttl=settings.cache_product_search_ttl,
            )
            # #region debug instrumentation
            try:
                file_debug_log(
                    "product_tools.py:375",
                    "cache_service.set_product_search result",
                    {
                        "query": query[:50],
                        "cache_result": cache_result,
                        "success": cache_result is True,
                    },
                    hypothesis_id="H",
                )
            except Exception:
                pass
            # #endregion

            return {
                "query": query,
                "category": category,
                "min_price": min_price,
                "max_price": max_price,
                "results_count": len(final_products),
                "products": final_products,
                "message": f"Found {len(final_products)} products",
                "cached": False,
            }
        except Exception as e:
            logger.error(f"Error searching products: {e}")
            return {"query": query, "results_count": 0, "products": [], "error": str(e)}


# Register product tools
product_availability_tool = ProductAvailabilityTool()
price_check_tool = PriceCheckTool()
product_search_tool = ProductSearchTool()

tool_registry.register(product_availability_tool)
tool_registry.register(price_check_tool)
tool_registry.register(product_search_tool)
