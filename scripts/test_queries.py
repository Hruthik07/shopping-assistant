"""Quick test script for the shopping assistant API."""
import sys
import io
import requests
import json
import time
from datetime import datetime

# Fix Unicode encoding for Windows
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

API_BASE = "http://localhost:3565"

# Test queries
test_queries = [
    "how are you today",
    "find me running shoes under $100",
    "show me wireless headphones",
    "I need a laptop for students under $500",
    "best skincare products"
]

def test_query(query: str, session_id: str = None):
    """Test a single query."""
    print(f"\n{'='*70}")
    print(f"Query: {query}")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    try:
        response = requests.post(
            f"{API_BASE}/api/chat/",
            json={
                "message": query,
                "session_id": session_id,
                "persona": "friendly",
                "tone": "warm"
            },
            timeout=120
        )
        
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            print(f"[OK] Success ({elapsed:.2f}s)")
            print(f"Response preview: {data.get('response', '')[:200]}...")
            print(f"Products found: {len(data.get('products', []))}")
            print(f"Tools used: {data.get('tools_used', [])}")
            if 'latency_breakdown' in data:
                print(f"Latency breakdown: {json.dumps(data['latency_breakdown'], indent=2)}")
            return data.get('session_id'), True
        else:
            print(f"[ERROR] {response.status_code}: {response.text[:200]}")
            return session_id, False
            
    except requests.exceptions.Timeout:
        print(f"[TIMEOUT] After {elapsed:.2f}s")
        return session_id, False
    except Exception as e:
        print(f"[ERROR] {str(e)[:200]}")
        return session_id, False

def main():
    """Run test queries."""
    print("\n" + "="*70)
    print("SHOPPING ASSISTANT API TEST")
    print("="*70)
    print(f"Testing server at: {API_BASE}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check health first
    try:
        health = requests.get(f"{API_BASE}/api/health", timeout=5)
        if health.status_code == 200:
            health_data = health.json()
            print(f"\n[OK] Server health: {health_data.get('status')}")
            print(f"  LLM Provider: {health_data.get('checks', {}).get('llm_provider', {}).get('provider', 'unknown')}")
        else:
            print(f"\n[ERROR] Server health check failed: {health.status_code}")
            return
    except Exception as e:
        print(f"\n[ERROR] Cannot connect to server: {e}")
        print("Please make sure the server is running at http://localhost:3565")
        return
    
    # Run test queries
    session_id = None
    results = []
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n[{i}/{len(test_queries)}]")
        session_id, success = test_query(query, session_id)
        results.append(success)
        time.sleep(1)  # Small delay between queries
    
    # Summary
    print(f"\n{'='*70}")
    print("TEST SUMMARY")
    print(f"{'='*70}")
    print(f"Total queries: {len(test_queries)}")
    print(f"Successful: {sum(results)}")
    print(f"Failed: {len(results) - sum(results)}")
    print(f"Success rate: {sum(results)/len(results)*100:.1f}%")

if __name__ == "__main__":
    main()
