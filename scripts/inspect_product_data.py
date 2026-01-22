"""Inspect product data structure to verify deal features."""
import asyncio
import httpx
import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

BASE_URL = "http://localhost:3565"


async def inspect_product():
    """Inspect a product response to see what fields are present."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{BASE_URL}/api/chat/",
            json={
                "message": "Find me wireless headphones under $100",
                "session_id": "inspect_test"
            },
            follow_redirects=True
        )
        
        data = response.json()
        products = data.get("products", [])
        
        if products:
            print("=" * 60)
            print("FIRST PRODUCT STRUCTURE")
            print("=" * 60)
            first_product = products[0]
            
            # Print all keys
            print("\nProduct Keys:")
            for key in sorted(first_product.keys()):
                value = first_product[key]
                if isinstance(value, dict):
                    print(f"  {key}: (dict with keys: {list(value.keys())})")
                elif isinstance(value, list):
                    print(f"  {key}: (list with {len(value)} items)")
                else:
                    print(f"  {key}: {type(value).__name__}")
            
            # Check for deal-related fields
            print("\n" + "=" * 60)
            print("DEAL-RELATED FIELDS")
            print("=" * 60)
            
            deal_info = first_product.get("deal_info")
            print(f"\ndeal_info: {deal_info}")
            
            price_comparison = first_product.get("price_comparison")
            print(f"\nprice_comparison: {price_comparison}")
            
            coupon_info = first_product.get("coupon_info")
            print(f"\ncoupon_info: {coupon_info}")
            
            customer_value = first_product.get("customer_value")
            print(f"\ncustomer_value: {customer_value}")
            
            # Print full product JSON
            print("\n" + "=" * 60)
            print("FULL PRODUCT JSON (first 2000 chars)")
            print("=" * 60)
            print(json.dumps(first_product, indent=2)[:2000])


async def test_metrics_endpoint():
    """Test the metrics endpoint directly."""
    print("\n" + "=" * 60)
    print("TESTING METRICS ENDPOINT")
    print("=" * 60)
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        # Try different paths
        paths = [
            "/api/metrics/deals",
            "/api/metrics/deals/",
            "/metrics/deals",
            "/metrics/deals/"
        ]
        
        for path in paths:
            try:
                response = await client.get(f"{BASE_URL}{path}")
                print(f"\n{path}: Status {response.status_code}")
                if response.status_code == 200:
                    print(json.dumps(response.json(), indent=2))
                    return True
                else:
                    print(f"Response: {response.text[:200]}")
            except Exception as e:
                print(f"{path}: Error - {e}")
        
        return False


if __name__ == "__main__":
    asyncio.run(inspect_product())
    asyncio.run(test_metrics_endpoint())
