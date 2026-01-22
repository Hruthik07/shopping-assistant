"""Test with a completely fresh query to bypass cache."""
import asyncio
import httpx
import json
import sys
from pathlib import Path
import time

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

BASE_URL = "http://localhost:3565"


async def test_fresh_query():
    """Test with a unique query to bypass cache."""
    # Use timestamp to make query unique
    unique_query = f"Find me unique wireless earbuds under $80 - {int(time.time())}"
    
    print("=" * 60)
    print("TESTING FRESH QUERY (NO CACHE)")
    print("=" * 60)
    print(f"Query: {unique_query}\n")
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{BASE_URL}/api/chat/",
            json={
                "message": unique_query,
                "session_id": f"fresh_test_{int(time.time())}"
            },
            follow_redirects=True
        )
        
        print(f"Status: {response.status_code}\n")
        
        if response.status_code != 200:
            print(f"Error: {response.text[:500]}")
            return
        
        data = response.json()
        products = data.get("products", [])
        
        print(f"Products returned: {len(products)}\n")
        
        if products:
            first_product = products[0]
            print("=" * 60)
            print("FIRST PRODUCT FIELDS:")
            print("=" * 60)
            for key in sorted(first_product.keys()):
                value = first_product[key]
                if isinstance(value, dict):
                    print(f"  {key}: (dict) {list(value.keys())}")
                elif isinstance(value, list):
                    print(f"  {key}: (list) {len(value)} items")
                else:
                    print(f"  {key}: {value}")
            
            print("\n" + "=" * 60)
            print("DEAL FEATURES CHECK:")
            print("=" * 60)
            print(f"  deal_info: {'deal_info' in first_product}")
            if 'deal_info' in first_product:
                print(f"    {first_product['deal_info']}")
            
            print(f"  price_comparison: {'price_comparison' in first_product}")
            if 'price_comparison' in first_product:
                print(f"    {first_product['price_comparison']}")
            
            print(f"  coupon_info: {'coupon_info' in first_product}")
            if 'coupon_info' in first_product:
                print(f"    {first_product['coupon_info']}")
            
            print(f"  customer_value: {'customer_value' in first_product}")
            if 'customer_value' in first_product:
                print(f"    Score: {first_product['customer_value'].get('score', 'N/A')}")


if __name__ == "__main__":
    asyncio.run(test_fresh_query())
