"""Test script for deal-finding features."""
import asyncio
import httpx
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

BASE_URL = "http://localhost:3565"


async def test_health():
    """Test health endpoints."""
    print("=" * 60)
    print("Testing Health Endpoints")
    print("=" * 60)
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        # Test liveness
        try:
            response = await client.get(f"{BASE_URL}/api/health/liveness")
            print(f"Liveness: {response.status_code} - {response.json()}")
        except Exception as e:
            print(f"Liveness check failed: {e}")
            return False
        
        # Test readiness
        try:
            response = await client.get(f"{BASE_URL}/api/health/readiness")
            print(f"Readiness: {response.status_code} - {response.json()}")
        except Exception as e:
            print(f"Readiness check failed: {e}")
            return False
        
        # Test metrics/deals endpoint
        try:
            response = await client.get(f"{BASE_URL}/api/metrics/deals")
            print(f"Deal Metrics: {response.status_code}")
            metrics = response.json()
            print(json.dumps(metrics, indent=2))
        except Exception as e:
            print(f"Deal metrics check failed: {e}")
        
        return True


async def test_product_search_with_deals():
    """Test product search with deal finding."""
    print("\n" + "=" * 60)
    print("Testing Product Search with Deal Finding")
    print("=" * 60)
    
    query = "Find me wireless headphones under $100"
    print(f"Query: {query}\n")
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(
                f"{BASE_URL}/api/chat/",
                json={
                    "message": query,
                    "session_id": "test_session_deal_features"
                },
                follow_redirects=True
            )
            
            print(f"Status: {response.status_code}")
            data = response.json()
            
            # Check if products are returned
            products = data.get("products", [])
            print(f"\nProducts returned: {len(products)}")
            
            if products:
                # Check first product for deal features
                first_product = products[0]
                print(f"\nFirst Product: {first_product.get('name', 'N/A')}")
                
                # Check for deal info
                deal_info = first_product.get("deal_info", {})
                if deal_info:
                    print(f"  Deal Info: {deal_info.get('is_deal', False)}")
                    if deal_info.get("is_deal"):
                        print(f"  Deal Badge: {deal_info.get('deal_badge', 'N/A')}")
                        print(f"  Savings: {deal_info.get('savings_percent', 0)}%")
                
                # Check for price comparison
                price_comp = first_product.get("price_comparison", {})
                if price_comp:
                    print(f"  Price Comparison: {price_comp.get('retailer_count', 0)} retailers")
                    print(f"  Best Price: ${price_comp.get('best_price', 0):.2f}")
                
                # Check for coupon info
                coupon_info = first_product.get("coupon_info", {})
                if coupon_info:
                    print(f"  Coupon Available: {coupon_info.get('has_coupon', False)}")
                
                # Check for customer value
                customer_value = first_product.get("customer_value", {})
                if customer_value:
                    print(f"  Customer Value Score: {customer_value.get('score', 0):.3f}")
                
                # Check for ranking
                if "rank" in first_product:
                    print(f"  Rank: {first_product.get('rank')}")
                    print(f"  Ranking Explanation: {first_product.get('ranking_explanation', 'N/A')}")
            
            # Check response text for deal badges
            response_text = data.get("response", "")
            if "Save" in response_text or "Best Price" in response_text or "Coupon" in response_text:
                print("\n[OK] Deal information found in response text")
            else:
                print("\n[INFO] No deal badges in response (may be normal if no deals detected)")
            
            return True
            
        except Exception as e:
            print(f"Error testing product search: {e}")
            import traceback
            traceback.print_exc()
            return False


async def test_real_queries():
    """Test with real user queries."""
    print("\n" + "=" * 60)
    print("Testing with Real User Queries")
    print("=" * 60)
    
    queries = [
        "i need a face wash to treat darkspots, it should be with less chemicals, doctor recommended, it should be less than 30$",
        "i am having a bodypain from last 2 days, can u recommend a good massage oil to treat pain with massage oil?",
        "i need a chess to play, it should make with only wood, board should be white and brown, it should be less than 70$"
    ]
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        for i, query in enumerate(queries, 1):
            print(f"\n--- Query {i} ---")
            print(f"Query: {query[:80]}...")
            
            try:
                response = await client.post(
                    f"{BASE_URL}/api/chat/",
                    json={
                        "message": query,
                        "session_id": f"test_session_{i}"
                    },
                    follow_redirects=True
                )
                
                data = response.json()
                products = data.get("products", [])
                
                print(f"Products found: {len(products)}")
                
                # Check for deal features
                deals_found = sum(1 for p in products if p.get("deal_info", {}).get("is_deal", False))
                price_comps = sum(1 for p in products if p.get("price_comparison", {}).get("retailer_count", 0) > 1)
                coupons = sum(1 for p in products if p.get("coupon_info", {}).get("has_coupon", False))
                
                print(f"  Deals detected: {deals_found}")
                print(f"  Price comparisons: {price_comps}")
                print(f"  Coupons available: {coupons}")
                
            except Exception as e:
                print(f"Error: {e}")


async def test_metrics():
    """Test metrics endpoints."""
    print("\n" + "=" * 60)
    print("Testing Metrics Endpoints")
    print("=" * 60)
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        # Test general metrics
        try:
            response = await client.get(f"{BASE_URL}/api/metrics")
            print("General Metrics:")
            print(json.dumps(response.json(), indent=2))
        except Exception as e:
            print(f"Error getting general metrics: {e}")
        
        # Test deal metrics
        try:
            response = await client.get(f"{BASE_URL}/api/metrics/deals")
            print("\nDeal Metrics:")
            metrics = response.json()
            print(json.dumps(metrics, indent=2))
            
            # Verify metrics structure
            if "deal_detection" in metrics:
                print("\n[OK] Deal detection metrics present")
            if "price_comparison" in metrics:
                print("[OK] Price comparison metrics present")
            if "api_performance" in metrics:
                print("[OK] API performance metrics present")
                
        except Exception as e:
            print(f"Error getting deal metrics: {e}")


async def main():
    """Run all tests."""
    print("=" * 60)
    print("DEAL FEATURES TESTING")
    print("=" * 60)
    
    # Test 1: Health checks
    health_ok = await test_health()
    if not health_ok:
        print("\n[ERROR] Health checks failed. Is the server running?")
        return
    
    # Test 2: Product search with deals
    await test_product_search_with_deals()
    
    # Test 3: Real queries
    await test_real_queries()
    
    # Test 4: Metrics
    await test_metrics()
    
    print("\n" + "=" * 60)
    print("TESTING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nTesting interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
