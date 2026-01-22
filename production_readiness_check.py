"""Production Readiness Check - Verify all systems are working."""
import asyncio
import httpx
from src.utils.cache import cache_service
from src.analytics.langfuse_client import langfuse_client
from src.utils.config import settings
from src.analytics.logger import logger

async def check_redis():
    """Check Redis connection."""
    print("\n[1/5] Checking Redis...")
    try:
        await cache_service.connect()
        if cache_service.enabled and cache_service.redis_client:
            # Quick test
            await cache_service.set("health:check", {"status": "ok"}, ttl=10)
            result = await cache_service.get("health:check")
            await cache_service.delete("health:check")
            if result:
                print("   [OK] Redis is connected and working")
                await cache_service.disconnect()
                return True
        print("   [WARNING] Redis is not enabled")
        await cache_service.disconnect()
        return False
    except Exception as e:
        print(f"   [ERROR] Redis check failed: {e}")
        return False

def check_langfuse():
    """Check Langfuse configuration."""
    print("\n[2/5] Checking Langfuse...")
    if langfuse_client.enabled and langfuse_client.client:
        print("   [OK] Langfuse is configured and enabled")
        return True
    else:
        print("   [WARNING] Langfuse is not enabled (optional)")
        return True  # Not critical

def check_config():
    """Check critical configuration."""
    print("\n[3/5] Checking Configuration...")
    issues = []
    
    if settings.llm_provider.lower() == "anthropic":
        if not settings.anthropic_api_key:
            issues.append("ANTHROPIC_API_KEY not set")
        else:
            print("   [OK] Anthropic API key configured")
    elif settings.llm_provider.lower() == "openai":
        if not settings.openai_api_key:
            issues.append("OPENAI_API_KEY not set")
        else:
            print("   [OK] OpenAI API key configured")
    
    if not settings.serper_api_key:
        issues.append("SERPER_API_KEY not set (product search may not work)")
    else:
        print("   [OK] Serper API key configured")
    
    if issues:
        print(f"   [WARNING] Configuration issues: {', '.join(issues)}")
        return False
    return True

async def check_api_server():
    """Check if API server is running."""
    print("\n[4/7] Checking API Server...")
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get("http://localhost:3565/api/health")
            if response.status_code == 200:
                health = response.json()
                status = health.get('status', 'unknown')
                print(f"   [OK] API server is running (status: {status})")
                checks = health.get('checks', {})
                for check_name, check_data in checks.items():
                    check_status = check_data.get('status', 'unknown')
                    if check_status == 'healthy':
                        print(f"   [OK] {check_name}: {check_status}")
                    else:
                        print(f"   [WARNING] {check_name}: {check_status}")
                return True
            else:
                print(f"   [ERROR] API server returned status {response.status_code}")
                return False
    except httpx.ConnectError:
        print("   [ERROR] API server is not running")
        print("   [INFO] Start server with: python start_server.py")
        return False
    except Exception as e:
        print(f"   [ERROR] API check failed: {e}")
        return False

async def test_end_to_end():
    """Test end-to-end query processing."""
    print("\n[5/7] Testing End-to-End Query...")
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "http://localhost:3565/api/chat/",
                json={
                    "message": "Hello, can you help me find products?",
                    "session_id": "readiness-check"
                },
                headers={"Content-Type": "application/json"}
            )
            if response.status_code == 200:
                result = response.json()
                print("   [OK] End-to-end query successful")
                print(f"   [INFO] Response length: {len(result.get('response', ''))} chars")
                print(f"   [INFO] Products found: {len(result.get('products', []))}")
                print(f"   [INFO] Tools used: {', '.join(result.get('tools_used', []))}")
                return True
            else:
                print(f"   [ERROR] Query failed with status {response.status_code}")
                print(f"   [INFO] Response: {response.text[:200]}")
                return False
    except Exception as e:
        print(f"   [ERROR] End-to-end test failed: {e}")
        return False

async def check_health_endpoints():
    """Check health check endpoints."""
    print("\n[6/7] Checking Health Endpoints...")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Check liveness
            response = await client.get("http://localhost:3565/api/health/liveness")
            if response.status_code == 200:
                print("   [OK] Liveness endpoint working")
            else:
                print(f"   [ERROR] Liveness endpoint failed: {response.status_code}")
            
            # Check readiness
            response = await client.get("http://localhost:3565/api/health/readiness")
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "ready":
                    print("   [OK] Readiness endpoint working")
                else:
                    print(f"   [WARNING] Readiness endpoint: {data.get('status')}")
            else:
                print(f"   [ERROR] Readiness endpoint failed: {response.status_code}")
            
            # Check full health
            response = await client.get("http://localhost:3565/api/health")
            if response.status_code == 200:
                data = response.json()
                status = data.get("status", "unknown")
                if status == "healthy":
                    print("   [OK] Full health check passing")
                else:
                    print(f"   [WARNING] Health check: {status}")
                    checks = data.get("checks", {})
                    for check_name, check_data in checks.items():
                        check_status = check_data.get("status", "unknown")
                        if check_status != "healthy":
                            print(f"   [WARNING] {check_name}: {check_status}")
            else:
                print(f"   [ERROR] Health check failed: {response.status_code}")
    except Exception as e:
        print(f"   [ERROR] Health endpoints check failed: {e}")
        return False
    return True


async def check_metrics_endpoints():
    """Check metrics endpoints."""
    print("\n[7/7] Checking Metrics Endpoints...")
    all_ok = True
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Check metrics
            try:
                response = await client.get("http://localhost:3565/api/metrics")
                if response.status_code == 200:
                    print("   [OK] Metrics endpoint working")
                else:
                    print(f"   [ERROR] Metrics endpoint failed: {response.status_code}")
                    all_ok = False
            except Exception as e:
                print(f"   [ERROR] Metrics endpoint check failed: {str(e)}")
                all_ok = False
            
            # Check cache metrics
            try:
                response = await client.get("http://localhost:3565/api/metrics/cache")
                if response.status_code == 200:
                    print("   [OK] Cache metrics endpoint working")
                else:
                    print(f"   [ERROR] Cache metrics endpoint failed: {response.status_code}")
                    all_ok = False
            except Exception as e:
                print(f"   [ERROR] Cache metrics endpoint check failed: {str(e)}")
                all_ok = False
    except Exception as e:
        print(f"   [ERROR] Metrics endpoints check failed: {str(e)}")
        return False
    return all_ok


async def main():
    """Run all production readiness checks."""
    print("=" * 60)
    print("PRODUCTION READINESS CHECK")
    print("=" * 60)
    
    api_server_ok = await check_api_server()
    
    results = {
        "redis": await check_redis(),
        "langfuse": check_langfuse(),
        "config": check_config(),
        "api_server": api_server_ok,
        "health_endpoints": await check_health_endpoints() if api_server_ok else False,
        "metrics_endpoints": await check_metrics_endpoints() if api_server_ok else False,
        "end_to_end": await test_end_to_end() if api_server_ok else False
    }
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_passed = all(results.values())
    
    for check, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} {check.replace('_', ' ').title()}")
    
    print("\n" + "=" * 60)
    if all_passed:
        print("[SUCCESS] All checks passed! System is production-ready.")
    else:
        print("[WARNING] Some checks failed. Review issues above.")
        print("\nNext steps:")
        if not results["api_server"]:
            print("  1. Start the API server: python start_server.py")
        if not results["config"]:
            print("  2. Check your .env file for missing API keys")
        if not results["redis"]:
            print("  3. Start Redis: docker run -d -p 8971:6379 redis:latest")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())

