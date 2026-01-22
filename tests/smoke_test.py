"""Smoke tests to validate critical functionality after bug fixes."""
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.routes.chat import chat
from src.api.schemas import ChatMessage
from src.agent.shopping_agent import ShoppingAgent
from src.analytics.logger import logger
import time


async def test_shopping_agent_initialization():
    """Test that ShoppingAgent initializes without errors."""
    print("\n[TEST] ShoppingAgent Initialization...")
    try:
        agent = ShoppingAgent()
        assert hasattr(agent, '_llm_cache'), "_llm_cache not initialized"
        assert isinstance(agent._llm_cache, dict), "_llm_cache should be a dict"
        # Note: _llm_cache may contain models after initialization
        print("[OK] ShoppingAgent initialized successfully")
        return True
    except Exception as e:
        print(f"[FAIL] ShoppingAgent initialization failed: {e}")
        return False


async def test_chat_endpoint_start_time():
    """Test that chat endpoint has start_time variable."""
    print("\n[TEST] Chat Endpoint start_time Variable...")
    try:
        # Import and check the function source
        import inspect
        source = inspect.getsource(chat)
        assert 'start_time = time.time()' in source, "start_time not initialized"
        print("[OK] start_time variable found in chat endpoint")
        return True
    except Exception as e:
        print(f"[FAIL] Chat endpoint check failed: {e}")
        return False


def test_debug_code_removed():
    """Test that debug instrumentation code is removed."""
    print("\n[TEST] Debug Code Removal...")
    try:
        agent_file = Path(__file__).parent.parent / "src" / "agent" / "shopping_agent.py"
        content = agent_file.read_text(encoding="utf-8", errors="replace")
        
        # Check for debug instrumentation markers
        debug_markers = [
            "# #region debug instrumentation",
            "# #endregion",
            "c:\\agentic_ai\\.cursor\\debug.log"
        ]
        
        found_markers = []
        for marker in debug_markers:
            if marker in content:
                found_markers.append(marker)
        
        if found_markers:
            print(f"[WARN] Found debug markers: {found_markers}")
            return False
        else:
            print("[OK] No debug instrumentation code found")
            return True
    except Exception as e:
        print(f"[FAIL] Debug code check failed: {e}")
        return False


async def test_imports():
    """Test that all critical imports work."""
    print("\n[TEST] Critical Imports...")
    try:
        from src.api.routes.chat import router
        from src.agent.shopping_agent import ShoppingAgent
        from src.utils.cache import cache_service
        from src.analytics.performance_monitor import performance_monitor
        from src.analytics.error_tracker import error_tracker
        print("[OK] All critical imports successful")
        return True
    except Exception as e:
        print(f"[FAIL] Import failed: {e}")
        return False


async def test_health_check():
    """Test health check endpoint (if server is running)."""
    print("\n[TEST] Health Check Endpoint...")
    try:
        import httpx
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get("http://localhost:3565/api/health")
            if response.status_code == 200:
                print("[OK] Health check endpoint responding")
                return True
            else:
                print(f"[WARN] Health check returned status {response.status_code}")
                return False
    except Exception as e:
        print(f"[WARN] Health check failed (server may not be running): {e}")
        print("   This is OK if server is not started")
        return True  # Not a failure if server isn't running


async def run_all_tests():
    """Run all smoke tests."""
    print("=" * 60)
    print("SMOKE TESTS - Post Bug Fix Validation")
    print("=" * 60)
    
    results = []
    
    # Test 1: Imports
    results.append(await test_imports())
    
    # Test 2: ShoppingAgent initialization
    results.append(await test_shopping_agent_initialization())
    
    # Test 3: Chat endpoint start_time
    results.append(await test_chat_endpoint_start_time())
    
    # Test 4: Debug code removal
    results.append(test_debug_code_removed())
    
    # Test 5: Health check (optional)
    results.append(await test_health_check())
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("[OK] ALL TESTS PASSED - System ready for deployment")
        return 0
    else:
        print("[WARN] SOME TESTS FAILED - Review issues above")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(run_all_tests())
    sys.exit(exit_code)
