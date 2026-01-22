"""Tests for price comparison engine."""
import pytest
from src.services.price_comparison import price_comparator


def test_price_comparison_single_retailer():
    """Test price comparison with single retailer."""
    products = [
        {
            "id": "prod1",
            "name": "Test Product",
            "price": 50.0,
            "shipping_cost": 5.0,
            "retailer": "Store A"
        }
    ]
    
    compared = price_comparator.compare_prices(products)
    
    assert len(compared) == 1
    assert "price_comparison" in compared[0]
    assert compared[0]["price_comparison"]["retailer_count"] == 1
    assert compared[0]["price_comparison"]["best_price"] == 55.0  # price + shipping


def test_price_comparison_multiple_retailers():
    """Test price comparison with multiple retailers."""
    products = [
        {
            "id": "prod1",
            "name": "Test Product",
            "price": 50.0,
            "shipping_cost": 5.0,
            "retailer": "Store A",
            "retailer_options": [
                {
                    "retailer": "Store A",
                    "price": 50.0,
                    "shipping_cost": 5.0,
                    "total_cost": 55.0
                },
                {
                    "retailer": "Store B",
                    "price": 45.0,
                    "shipping_cost": 10.0,
                    "total_cost": 55.0
                },
                {
                    "retailer": "Store C",
                    "price": 40.0,
                    "shipping_cost": 0.0,
                    "total_cost": 40.0
                }
            ]
        }
    ]
    
    compared = price_comparator.compare_prices(products)
    
    assert len(compared) == 1
    product = compared[0]
    assert "price_comparison" in product
    assert product["price_comparison"]["retailer_count"] == 3
    assert product["price_comparison"]["best_price"] == 40.0
    assert product["price_comparison"]["worst_price"] == 55.0
    assert product["price_comparison"]["savings"] == 15.0
    assert product["price"] == 40.0  # Should be updated to best price


def test_calculate_total_cost():
    """Test total cost calculation."""
    total = price_comparator.calculate_total_cost(
        price=100.0,
        shipping_cost=10.0,
        tax=5.0,
        discount=15.0
    )
    
    assert total == 100.0  # 100 + 10 + 5 - 15


def test_rank_by_customer_value():
    """Test customer value ranking."""
    products = [
        {
            "id": "prod1",
            "name": "Product 1",
            "price": 100.0,
            "shipping_cost": 10.0,
            "rating": 4.5,
            "reviews": 100
        },
        {
            "id": "prod2",
            "name": "Product 2",
            "price": 80.0,
            "shipping_cost": 5.0,
            "rating": 4.0,
            "reviews": 50
        },
        {
            "id": "prod3",
            "name": "Product 3",
            "price": 120.0,
            "shipping_cost": 0.0,
            "rating": 5.0,
            "reviews": 200
        }
    ]
    
    ranked = price_comparator.rank_by_customer_value(products)
    
    # Should be sorted by total cost (price + shipping)
    assert ranked[0]["id"] == "prod2"  # 85.0 total
    assert ranked[1]["id"] == "prod1"  # 110.0 total
    assert ranked[2]["id"] == "prod3"  # 120.0 total


def test_price_comparison_empty_list():
    """Test price comparison with empty list."""
    compared = price_comparator.compare_prices([])
    assert compared == []


if __name__ == "__main__":
    test_price_comparison_single_retailer()
    test_price_comparison_multiple_retailers()
    test_calculate_total_cost()
    test_rank_by_customer_value()
    test_price_comparison_empty_list()
    print("All price comparison tests passed!")
