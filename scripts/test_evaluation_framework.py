"""Test the evaluation framework (DeepEval + IR Metrics)."""
import asyncio
from evaluation.unified_evaluator import UnifiedEvaluator
from src.analytics.logger import logger

async def test_evaluation():
    """Test the evaluation framework with a simple query."""
    print("=" * 60)
    print("Evaluation Framework Test")
    print("=" * 60)
    
    evaluator = UnifiedEvaluator(base_url="http://localhost:3565")
    
    # Test query
    test_query = "Find wireless headphones under $100"
    session_id = "test-eval-session"
    
    print(f"\nTest Query: '{test_query}'")
    print(f"Session ID: {session_id}")
    print("\nRunning evaluation...")
    
    try:
        # Run evaluation
        result = await evaluator.evaluate_query(
            query=test_query,
            session_id=session_id,
            evaluation_types=["llm_quality", "performance"]  # Skip retrieval (needs labels)
        )
        
        print("\n" + "=" * 60)
        print("Evaluation Results")
        print("=" * 60)
        
        # Display results
        if "deepeval" in result:
            deepeval = result["deepeval"]
            print(f"\n[DeepEval] Enabled: {deepeval.get('enabled', False)}")
            if deepeval.get("enabled"):
                print(f"[DeepEval] Overall Score: {deepeval.get('overall_score', 0):.3f}")
                print(f"[DeepEval] All Passed: {deepeval.get('all_passed', False)}")
                if "metrics" in deepeval:
                    print("\nDeepEval Metrics:")
                    for metric_name, metric_data in deepeval["metrics"].items():
                        score = metric_data.get("score", 0)
                        passed = metric_data.get("passed", False)
                        status = "[PASS]" if passed else "[FAIL]"
                        print(f"   {status} {metric_name}: {score:.3f}")
            else:
                print(f"[DeepEval] Error: {deepeval.get('error', 'Unknown error')}")
        
        # Performance metrics
        if "performance" in result:
            perf = result["performance"]
            print(f"\n[Performance] Total Time: {perf.get('total_time', 0):.3f}s")
            print(f"[Performance] TTFT: {perf.get('ttft', 0):.3f}s")
        
        # API Response
        if "api_response" in result:
            api = result["api_response"]
            print(f"\n[API] Response Length: {len(api.get('response', ''))} chars")
            print(f"[API] Products Found: {len(api.get('products', []))}")
            print(f"[API] Tools Used: {', '.join(api.get('tools_used', []))}")
        
        print("\n[SUCCESS] Evaluation framework is working!")
        
    except Exception as e:
        print(f"\n[ERROR] Evaluation failed: {e}")
        print("\nNote: Make sure the server is running on http://localhost:3565")
        print("      Run: python start_server.py")
        logger.error(f"Evaluation test failed: {e}", exc_info=True)
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    asyncio.run(test_evaluation())

