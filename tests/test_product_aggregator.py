"""Integration tests for product aggregator."""
import pytest
import asyncio
from src.services.product_aggregator import product_aggregator
from src.services.data_sources.serper_source import SerperDataSource
from src.analytics.logger import logger


@pytest.mark.asyncio
async def test_aggregator_initialization():
    """Test that aggregator initializes with available sources."""
    # Aggregator should initialize with at least Serper if configured
    assert len(product_aggregator.sources) > 0, "Aggregator should have at least one data source"
    logger.info(f"Aggregator initialized with {len(product_aggregator.sources)} sources")


@pytest.mark.asyncio
async def test_serper_source_available():
    """Test that Serper source is available if API key is configured."""
    serper_source = SerperDataSource()
    # This will be True if SERPER_API_KEY is set
    is_available = serper_source.is_available()
    logger.info(f"Serper source available: {is_available}")
    # Test passes regardless - just checking availability


@pytest.mark.asyncio
async def test_aggregator_search_products():
    """Test product aggregation with a simple query."""
    if not product_aggregator.sources:
        pytest.skip("No data sources available - skipping integration test")
    
    query = "wireless headphones"
    results = await product_aggregator.search_products(
        query=query,
        num_results=5
    )
    
    assert isinstance(results, list), "Results should be a list"
    logger.info(f"Found {len(results)} products for query: {query}")
    
    if results:
        # Validate product structure
        product = results[0]
        assert "id" in product, "Product should have 'id'"
        assert "name" in product, "Product should have 'name'"
        assert "price" in product, "Product should have 'price'"
        assert "retailer" in product, "Product should have 'retailer'"
        assert "source" in product, "Product should have 'source'"
        logger.info(f"Sample product: {product.get('name', 'N/A')} - ${product.get('price', 0)}")


@pytest.mark.asyncio
async def test_aggregator_price_filtering():
    """Test price filtering in aggregator."""
    if not product_aggregator.sources:
        pytest.skip("No data sources available")
    
    query = "laptop"
    results = await product_aggregator.search_products(
        query=query,
        num_results=10,
        min_price=500.0,
        max_price=1000.0
    )
    
    # Verify all results are within price range
    for product in results:
        price = product.get("price", 0)
        assert price >= 500.0, f"Product {product.get('name')} price {price} below minimum"
        assert price <= 1000.0, f"Product {product.get('name')} price {price} above maximum"
    
    logger.info(f"Price filtering test: {len(results)} products in range $500-$1000")


@pytest.mark.asyncio
async def test_aggregator_category_filtering():
    """Test category filtering."""
    if not product_aggregator.sources:
        pytest.skip("No data sources available")
    
    query = "headphones"
    results = await product_aggregator.search_products(
        query=query,
        num_results=5,
        category="electronics"
    )
    
    logger.info(f"Category filtering test: {len(results)} products in 'electronics' category")


@pytest.mark.asyncio
async def test_aggregator_deduplication():
    """Test that products are deduplicated by identifier."""
    if not product_aggregator.sources:
        pytest.skip("No data sources available")
    
    query = "iPhone"
    results = await product_aggregator.search_products(
        query=query,
        num_results=10
    )
    
    # Check for duplicate IDs
    product_ids = [p.get("id") for p in results]
    unique_ids = set(product_ids)
    
    assert len(product_ids) == len(unique_ids), "Products should be deduplicated"
    logger.info(f"Deduplication test: {len(results)} unique products from {len(product_ids)} total")


@pytest.mark.asyncio
async def test_aggregator_empty_query():
    """Test aggregator handles empty query gracefully."""
    if not product_aggregator.sources:
        pytest.skip("No data sources available")
    
    results = await product_aggregator.search_products(
        query="",
        num_results=5
    )
    
    # Should return empty list or handle gracefully
    assert isinstance(results, list), "Should return a list even for empty query"
    logger.info(f"Empty query test: {len(results)} results")


@pytest.mark.asyncio
async def test_aggregator_error_handling():
    """Test aggregator handles API errors gracefully."""
    if not product_aggregator.sources:
        pytest.skip("No data sources available")
    
    # Test with invalid query that might cause errors
    results = await product_aggregator.search_products(
        query="!@#$%^&*()",
        num_results=5
    )
    
    # Should not crash, return empty list or handle gracefully
    assert isinstance(results, list), "Should return a list even on error"
    logger.info(f"Error handling test: {len(results)} results for invalid query")


if __name__ == "__main__":
    # Run tests
    asyncio.run(test_aggregator_initialization())
    asyncio.run(test_serper_source_available())
    # Only run integration tests if sources are available
    if product_aggregator.sources:
        asyncio.run(test_aggregator_search_products())
        asyncio.run(test_aggregator_price_filtering())
        asyncio.run(test_aggregator_deduplication())
    print("All aggregator tests completed!")
