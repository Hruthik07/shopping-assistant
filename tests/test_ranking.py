"""Tests for customer-first ranking algorithm."""
import pytest
from src.services.ranking_service import ranking_service
from src.services.customer_value_calculator import customer_value_calculator


def test_ranking_by_customer_value():
    """Test ranking by customer value score."""
    products = [
        {
            "id": "prod1",
            "name": "Product 1",
            "price": 100.0,
            "shipping_cost": 10.0,
            "rating": 4.0,
            "reviews": 50
        },
        {
            "id": "prod2",
            "name": "Product 2",
            "price": 80.0,
            "shipping_cost": 5.0,
            "rating": 4.5,
            "reviews": 100
        },
        {
            "id": "prod3",
            "name": "Product 3",
            "price": 120.0,
            "shipping_cost": 0.0,
            "rating": 3.5,
            "reviews": 20
        }
    ]
    
    # Calculate customer value scores
    for product in products:
        customer_value_calculator.calculate_customer_value_score(product)
    
    # Rank products
    ranked = ranking_service.rank_products(products, sort_by="customer_value")
    
    assert len(ranked) == 3
    # Should be sorted by customer value score (highest first)
    assert ranked[0]["rank"] == 1
    assert ranked[1]["rank"] == 2
    assert ranked[2]["rank"] == 3
    
    # All should have ranking explanation
    for product in ranked:
        assert "ranking_explanation" in product
        assert "rank" in product


def test_ranking_by_price():
    """Test ranking by price."""
    products = [
        {"id": "prod1", "price": 100.0, "shipping_cost": 10.0},
        {"id": "prod2", "price": 80.0, "shipping_cost": 5.0},
        {"id": "prod3", "price": 120.0, "shipping_cost": 0.0}
    ]
    
    ranked = ranking_service.rank_products(products, sort_by="price")
    
    # Should be sorted by total cost (lowest first)
    assert ranked[0]["price"] + ranked[0]["shipping_cost"] <= ranked[1]["price"] + ranked[1]["shipping_cost"]


def test_ranking_by_rating():
    """Test ranking by rating."""
    products = [
        {"id": "prod1", "rating": 4.0, "reviews": 50},
        {"id": "prod2", "rating": 4.5, "reviews": 100},
        {"id": "prod3", "rating": 3.5, "reviews": 20}
    ]
    
    ranked = ranking_service.rank_products(products, sort_by="rating")
    
    # Should be sorted by rating (highest first)
    assert ranked[0]["rating"] >= ranked[1]["rating"]
    assert ranked[1]["rating"] >= ranked[2]["rating"]


def test_ranking_with_deals():
    """Test ranking includes deal information."""
    products = [
        {
            "id": "prod1",
            "price": 100.0,
            "shipping_cost": 0.0,
            "deal_info": {
                "is_deal": True,
                "savings_percent": 20.0,
                "deal_badge": "Save 20%"
            }
        },
        {
            "id": "prod2",
            "price": 90.0,
            "shipping_cost": 5.0,
            "deal_info": {
                "is_deal": False
            }
        }
    ]
    
    ranked = ranking_service.rank_products(products)
    
    # Product with deal should rank higher due to customer value
    assert len(ranked) == 2
    for product in ranked:
        assert "ranking_explanation" in product


def test_customer_value_calculation():
    """Test customer value score calculation."""
    product = {
        "price": 50.0,
        "shipping_cost": 5.0,
        "rating": 4.5,
        "reviews": 100,
        "deal_info": {
            "is_deal": True,
            "savings_percent": 15.0
        },
        "coupon_info": {
            "has_coupon": False
        }
    }
    
    score = customer_value_calculator.calculate_customer_value_score(product)
    
    assert 0.0 <= score <= 1.0, "Score should be between 0 and 1"
    assert "customer_value" in product
    assert "breakdown" in product["customer_value"]


def test_ranking_explanation():
    """Test ranking explanation generation."""
    product = {
        "price": 50.0,
        "shipping_cost": 0.0,
        "rating": 4.5,
        "deal_info": {
            "is_deal": True,
            "savings_percent": 20.0,
            "deal_badge": "Save 20%"
        },
        "price_comparison": {
            "retailer_count": 3,
            "savings_percent": 10.0
        }
    }
    
    explanation = ranking_service._generate_ranking_explanation(product, "customer_value")
    
    assert isinstance(explanation, str)
    assert len(explanation) > 0


def test_ranking_factors():
    """Test getting ranking factors."""
    product = {
        "price": 50.0,
        "rating": 4.5,
        "customer_value": {
            "score": 0.8
        },
        "deal_info": {
            "is_deal": True
        }
    }
    
    factors = ranking_service.get_ranking_factors(product)
    
    assert "customer_value_score" in factors
    assert "price" in factors
    assert "has_deal" in factors


if __name__ == "__main__":
    test_ranking_by_customer_value()
    test_ranking_by_price()
    test_ranking_by_rating()
    test_ranking_with_deals()
    test_customer_value_calculation()
    test_ranking_explanation()
    test_ranking_factors()
    print("All ranking tests passed!")
