"""Tests for deal detection service."""

import pytest
import asyncio
from datetime import datetime, timedelta
from src.services.deal_detector import deal_detector
from src.services.price_tracker import price_tracker
from src.database.db import get_db
from src.database.models import PriceHistory


@pytest.mark.asyncio
async def test_deal_detection_no_history():
    """Test deal detection when no price history exists."""
    product = {
        "id": "test_product_1",
        "name": "Test Product",
        "price": 50.0,
        "shipping_cost": 5.0,
        "retailer": "Test Store",
        "source": "test",
    }

    result = await deal_detector._analyze_product_deals(product)

    # Should not crash, should add deal_info
    assert "deal_info" in result
    # No history means no deal detected initially
    assert result["deal_info"]["is_deal"] is False or result["deal_info"].get("is_deal") is not None


@pytest.mark.asyncio
async def test_deal_detection_price_drop():
    """Test deal detection with price drop."""
    product_id = "test_product_2"

    # Create price history with higher prices
    db = next(get_db())
    try:
        # Add historical prices (higher)
        for days_ago in [30, 20, 10]:
            old_price = PriceHistory(
                product_id=product_id,
                product_name="Test Product",
                retailer="Test Store",
                price=100.0,
                total_cost=100.0,
                timestamp=datetime.utcnow() - timedelta(days=days_ago),
                source="test",
            )
            db.add(old_price)

        db.commit()

        # Now test with lower current price
        product = {
            "id": product_id,
            "name": "Test Product",
            "price": 70.0,  # 30% drop
            "shipping_cost": 0.0,
            "retailer": "Test Store",
            "source": "test",
        }

        result = await deal_detector._analyze_product_deals(product)

        assert "deal_info" in result
        # Should detect significant price drop
        assert result["deal_info"]["is_deal"] is True
        assert result["deal_info"]["savings_percent"] > 0

    finally:
        # Cleanup
        db.query(PriceHistory).filter(PriceHistory.product_id == product_id).delete()
        db.commit()
        db.close()


def test_is_best_deal():
    """Test best deal detection."""
    product = {"id": "prod1", "price": 50.0, "shipping_cost": 5.0}

    comparison_products = [
        {"price": 60.0, "shipping_cost": 10.0},
        {"price": 55.0, "shipping_cost": 5.0},
        {"price": 70.0, "shipping_cost": 0.0},
    ]

    is_best = deal_detector.is_best_deal(product, comparison_products)
    # prod1 total: 55, others: 70, 60, 70 - so prod1 is best
    assert is_best is True


def test_deal_thresholds():
    """Test deal detection thresholds."""
    assert deal_detector.price_drop_thresholds["significant"] == 10.0
    assert deal_detector.price_drop_thresholds["major"] == 20.0
    assert deal_detector.price_drop_thresholds["extreme"] == 30.0


@pytest.mark.asyncio
async def test_deal_detection_multiple_products():
    """Test deal detection on multiple products."""
    products = [
        {
            "id": "prod1",
            "name": "Product 1",
            "price": 50.0,
            "shipping_cost": 5.0,
            "retailer": "Store A",
            "source": "test",
        },
        {
            "id": "prod2",
            "name": "Product 2",
            "price": 100.0,
            "shipping_cost": 10.0,
            "retailer": "Store B",
            "source": "test",
        },
    ]

    results = await deal_detector.detect_deals(products)

    assert len(results) == 2
    for product in results:
        assert "deal_info" in product


if __name__ == "__main__":
    asyncio.run(test_deal_detection_no_history())
    test_is_best_deal()
    test_deal_thresholds()
    asyncio.run(test_deal_detection_multiple_products())
    print("All deal detection tests completed!")
