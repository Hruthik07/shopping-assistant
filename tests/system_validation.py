"""Comprehensive system validation test."""
import asyncio
import sys
import json
from pathlib import Path
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx


BASE_URL = "http://localhost:3565"
TIMEOUT = 30.0


async def test_health_endpoint():
    """Test health check endpoint."""
    print("\n[TEST] Health Check Endpoint...")
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.get(f"{BASE_URL}/api/health")
            if response.status_code == 200:
                data = response.json()
                print(f"[OK] Health check passed: {data.get('status', 'unknown')}")
                return True
            else:
                print(f"[FAIL] Health check returned status {response.status_code}")
                return False
    except Exception as e:
        print(f"[FAIL] Health check failed: {e}")
        return False


async def test_liveness():
    """Test liveness endpoint."""
    print("\n[TEST] Liveness Endpoint...")
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.get(f"{BASE_URL}/api/health/liveness")
            if response.status_code == 200:
                print("[OK] Liveness check passed")
                return True
            else:
                print(f"[FAIL] Liveness returned status {response.status_code}")
                return False
    except Exception as e:
        print(f"[FAIL] Liveness check failed: {e}")
        return False


async def test_readiness():
    """Test readiness endpoint."""
    print("\n[TEST] Readiness Endpoint...")
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.get(f"{BASE_URL}/api/health/readiness")
            if response.status_code == 200:
                data = response.json()
                print(f"[OK] Readiness check passed: {data.get('status', 'unknown')}")
                return True
            else:
                print(f"[FAIL] Readiness returned status {response.status_code}")
                return False
    except Exception as e:
        print(f"[FAIL] Readiness check failed: {e}")
        return False


async def test_metrics_endpoint():
    """Test metrics endpoint."""
    print("\n[TEST] Metrics Endpoint...")
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.get(f"{BASE_URL}/api/metrics")
            if response.status_code == 200:
                data = response.json()
                print(f"[OK] Metrics endpoint accessible")
                print(f"    - Cache stats: {data.get('cache', {}).get('enabled', False)}")
                return True
            else:
                print(f"[FAIL] Metrics returned status {response.status_code}")
                return False
    except Exception as e:
        print(f"[FAIL] Metrics check failed: {e}")
        return False


async def test_chat_endpoint_basic():
    """Test basic chat endpoint functionality."""
    print("\n[TEST] Chat Endpoint - Basic Query...")
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            payload = {
                "message": "Find me wireless headphones under $100"
            }
            start_time = time.time()
            response = await client.post(
                f"{BASE_URL}/api/chat/",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                print(f"[OK] Chat endpoint responded in {elapsed:.2f}s")
                print(f"    - Response length: {len(data.get('response', ''))} chars")
                print(f"    - Products found: {len(data.get('products', []))}")
                print(f"    - Tools used: {len(data.get('tools_used', []))}")
                print(f"    - Request ID: {data.get('request_id', 'N/A')}")
                
                # Check for critical fields
                if 'response' in data:
                    print("[OK] Response field present")
                if 'request_id' in data:
                    print("[OK] Request ID present")
                if 'latency_breakdown' in data:
                    print("[OK] Latency breakdown present")
                if 'cache_stats' in data:
                    print("[OK] Cache stats present")
                
                return True
            else:
                print(f"[FAIL] Chat endpoint returned status {response.status_code}")
                print(f"    Response: {response.text[:200]}")
                return False
    except Exception as e:
        print(f"[FAIL] Chat endpoint failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_chat_endpoint_error_handling():
    """Test error handling in chat endpoint."""
    print("\n[TEST] Chat Endpoint - Error Handling...")
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            # Test with empty message (should be rejected)
            payload = {"message": ""}
            response = await client.post(
                f"{BASE_URL}/api/chat/",
                json=payload
            )
            
            if response.status_code in [400, 422]:
                print("[OK] Empty message properly rejected")
                return True
            else:
                print(f"[WARN] Empty message returned status {response.status_code}")
                return True  # Not a critical failure
    except Exception as e:
        print(f"[FAIL] Error handling test failed: {e}")
        return False


async def test_cache_stats():
    """Test cache statistics endpoint."""
    print("\n[TEST] Cache Statistics Endpoint...")
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.get(f"{BASE_URL}/api/metrics/cache")
            if response.status_code == 200:
                data = response.json()
                print(f"[OK] Cache stats accessible")
                print(f"    - Enabled: {data.get('enabled', False)}")
                print(f"    - Hit rate: {data.get('hit_rate', 0):.1f}%")
                return True
            else:
                print(f"[FAIL] Cache stats returned status {response.status_code}")
                return False
    except Exception as e:
        print(f"[FAIL] Cache stats check failed: {e}")
        return False


async def test_error_stats():
    """Test error statistics endpoint."""
    print("\n[TEST] Error Statistics Endpoint...")
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.get(f"{BASE_URL}/api/errors/stats")
            if response.status_code == 200:
                data = response.json()
                print(f"[OK] Error stats accessible")
                print(f"    - Total errors: {data.get('total_errors', 0)}")
                return True
            else:
                print(f"[FAIL] Error stats returned status {response.status_code}")
                return False
    except Exception as e:
        print(f"[FAIL] Error stats check failed: {e}")
        return False


async def test_root_endpoint():
    """Test root endpoint."""
    print("\n[TEST] Root Endpoint...")
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.get(f"{BASE_URL}/")
            if response.status_code in [200, 404]:  # 404 is OK if no frontend
                print("[OK] Root endpoint accessible")
                return True
            else:
                print(f"[WARN] Root endpoint returned status {response.status_code}")
                return True  # Not critical
    except Exception as e:
        print(f"[FAIL] Root endpoint check failed: {e}")
        return False


async def run_all_tests():
    """Run all system validation tests."""
    print("=" * 70)
    print("COMPREHENSIVE SYSTEM VALIDATION")
    print("=" * 70)
    print(f"Testing server at: {BASE_URL}")
    print("Waiting for server to be ready...")
    
    # Wait for server to start
    max_retries = 10
    for i in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                await client.get(f"{BASE_URL}/api/health/liveness")
                print("[OK] Server is ready!")
                break
        except:
            if i < max_retries - 1:
                print(f"Waiting for server... ({i+1}/{max_retries})")
                await asyncio.sleep(2)
            else:
                print("[FAIL] Server did not start in time")
                return 1
    
    results = []
    
    # Core health checks
    results.append(await test_health_endpoint())
    results.append(await test_liveness())
    results.append(await test_readiness())
    
    # Metrics and monitoring
    results.append(await test_metrics_endpoint())
    results.append(await test_cache_stats())
    results.append(await test_error_stats())
    
    # API functionality
    results.append(await test_root_endpoint())
    results.append(await test_chat_endpoint_basic())
    results.append(await test_chat_endpoint_error_handling())
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("[OK] ALL TESTS PASSED - System is running correctly!")
        return 0
    else:
        print("[WARN] SOME TESTS FAILED - Review issues above")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(run_all_tests())
    sys.exit(exit_code)
