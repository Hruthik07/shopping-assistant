"""Test services directly to verify they add fields."""
import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.services.price_comparison import price_comparator
from src.services.deal_detector import deal_detector
from src.services.promo_matcher import promo_matcher
from src.services.ranking_service import ranking_service


async def test_services():
    """Test services directly with sample product."""
    # Sample product
    product = {
        "id": "test_123",
        "name": "Test Product",
        "price": 50.0,
        "currency": "USD",
        "rating": 4.5,
        "reviews": 100,
        "category": "electronics",
        "source": "google_shopping",
        "availability": True
    }
    
    products = [product]
    
    print("=" * 60)
    print("TESTING SERVICES DIRECTLY")
    print("=" * 60)
    
    print(f"\nOriginal product keys: {list(product.keys())}")
    
    # Test price comparison
    print("\n1. Testing price_comparator.compare_prices...")
    products = price_comparator.compare_prices(products)
    print(f"   After price_comparison: {list(products[0].keys())}")
    print(f"   Has price_comparison: {'price_comparison' in products[0]}")
    if 'price_comparison' in products[0]:
        print(f"   price_comparison: {products[0]['price_comparison']}")
    
    # Test deal detection
    print("\n2. Testing deal_detector.detect_deals...")
    products = await deal_detector.detect_deals(products)
    print(f"   After deal_detection: {list(products[0].keys())}")
    print(f"   Has deal_info: {'deal_info' in products[0]}")
    if 'deal_info' in products[0]:
        print(f"   deal_info: {products[0]['deal_info']}")
    
    # Test promo matching
    print("\n3. Testing promo_matcher.match_promos_to_products...")
    products = await promo_matcher.match_promos_to_products(products)
    print(f"   After promo_matching: {list(products[0].keys())}")
    print(f"   Has coupon_info: {'coupon_info' in products[0]}")
    if 'coupon_info' in products[0]:
        print(f"   coupon_info: {products[0]['coupon_info']}")
    
    # Test ranking
    print("\n4. Testing ranking_service.rank_products...")
    products = ranking_service.rank_products(products, sort_by="customer_value")
    print(f"   After ranking: {list(products[0].keys())}")
    print(f"   Has customer_value: {'customer_value' in products[0]}")
    if 'customer_value' in products[0]:
        print(f"   customer_value: {products[0]['customer_value']}")
    print(f"   Has rank: {'rank' in products[0]}")
    
    print("\n" + "=" * 60)
    print("FINAL PRODUCT KEYS:")
    print("=" * 60)
    for key in sorted(products[0].keys()):
        print(f"  {key}: {type(products[0][key]).__name__}")


if __name__ == "__main__":
    asyncio.run(test_services())
