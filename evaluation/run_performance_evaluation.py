"""Run performance evaluation with multiple queries."""
import asyncio
import httpx
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from evaluation.latency_dashboard import LatencyDashboard
from src.analytics.latency_tracker import latency_tracker


class PerformanceEvaluator:
    """Evaluate system performance with test queries."""
    
    def __init__(self, base_url: str = "http://localhost:3565"):
        self.base_url = base_url
        self.results: List[Dict[str, Any]] = []
    
    async def run_query(self, query: str, session_id: str = None) -> Dict[str, Any]:
        """Run a single query and collect metrics."""
        if session_id is None:
            session_id = f"eval-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/chat/",
                    json={
                        "message": query,
                        "session_id": session_id
                    },
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                result = response.json()
                
                elapsed = asyncio.get_event_loop().time() - start_time
                
                return {
                    "query": query,
                    "session_id": session_id,
                    "success": True,
                    "response_time": elapsed,
                    "response_length": len(result.get("response", "")),
                    "products_found": len(result.get("products", [])),
                    "tools_used": result.get("tools_used", []),
                    "intent": result.get("intent", "unknown"),
                    "request_id": result.get("request_id"),
                    "latency_breakdown": result.get("latency_breakdown") or {},
                    "error": None
                }
        except Exception as e:
            elapsed = asyncio.get_event_loop().time() - start_time
            return {
                "query": query,
                "session_id": session_id,
                "success": False,
                "response_time": elapsed,
                "error": str(e)
            }
    
    async def run_evaluation(self, queries: List[str]) -> Dict[str, Any]:
        """Run evaluation with multiple queries."""
        print(f"\n{'='*60}")
        print("PERFORMANCE EVALUATION")
        print(f"{'='*60}\n")
        
        print(f"Running {len(queries)} queries...\n")
        
        for i, query in enumerate(queries, 1):
            print(f"[{i}/{len(queries)}] Query: {query}")
            result = await self.run_query(query, session_id=f"eval-{i}")
            self.results.append(result)
            
            if result["success"]:
                print(f"  [OK] Success - {result['response_time']:.2f}s")
                print(f"  Products: {result['products_found']}, Tools: {result['tools_used']}")
            else:
                print(f"  [FAIL] Failed - {result['error']}")
            print()
        
        return self.generate_report()
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        successful = [r for r in self.results if r["success"]]
        failed = [r for r in self.results if not r["success"]]
        
        if successful:
            response_times = [r["response_time"] for r in successful]
            avg_response_time = sum(response_times) / len(response_times)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
        else:
            avg_response_time = min_response_time = max_response_time = 0
        
        # Get latency component stats
        latency_stats = latency_tracker.get_all_stats()
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_queries": len(self.results),
                "successful": len(successful),
                "failed": len(failed),
                "success_rate": len(successful) / len(self.results) if self.results else 0,
                "avg_response_time": avg_response_time,
                "min_response_time": min_response_time,
                "max_response_time": max_response_time
            },
            "query_results": self.results,
            "latency_breakdown": latency_stats,
            "recommendations": self._generate_recommendations(successful, latency_stats)
        }
        
        return report
    
    def _generate_recommendations(
        self,
        successful: List[Dict],
        latency_stats: Dict[str, Dict[str, float]]
    ) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        if not successful:
            return ["No successful queries to analyze"]
        
        avg_time = sum(r["response_time"] for r in successful) / len(successful)
        
        if avg_time > 3.0:
            recommendations.append(f"Average response time ({avg_time:.2f}s) is high - consider optimization")
        
        # Check component bottlenecks
        if latency_stats.get("serper_api_call", {}).get("mean", 0) > 1.0:
            recommendations.append("Serper API calls are slow - implement caching")
        
        if latency_stats.get("llm_processing", {}).get("mean", 0) > 2.0:
            recommendations.append("LLM processing is slow - consider reducing context size")
        
        if latency_stats.get("rag_retrieval", {}).get("mean", 0) > 0.5:
            recommendations.append("RAG retrieval is slow - optimize vector search")
        
        if not recommendations:
            recommendations.append("Performance looks good! No major issues detected.")
        
        return recommendations
    
    def print_report(self, report: Dict[str, Any]):
        """Print formatted report."""
        print(f"\n{'='*60}")
        print("PERFORMANCE EVALUATION REPORT")
        print(f"{'='*60}\n")
        
        summary = report["summary"]
        print("--- Summary ---")
        print(f"Total Queries: {summary['total_queries']}")
        print(f"Successful: {summary['successful']} ({summary['success_rate']*100:.1f}%)")
        print(f"Failed: {summary['failed']}")
        print(f"\nResponse Times:")
        print(f"  Average: {summary['avg_response_time']:.3f}s")
        print(f"  Min: {summary['min_response_time']:.3f}s")
        print(f"  Max: {summary['max_response_time']:.3f}s")
        
        print(f"\n--- Query Results ---")
        for i, result in enumerate(report["query_results"], 1):
            status = "[OK]" if result["success"] else "[FAIL]"
            print(f"\n{i}. {status} {result['query']}")
            if result["success"]:
                print(f"   Time: {result['response_time']:.3f}s")
                print(f"   Products: {result['products_found']}")
                print(f"   Tools: {result.get('tools_used', [])}")
                print(f"   Intent: {result.get('intent', 'unknown')}")
            else:
                print(f"   Error: {result.get('error', 'Unknown')}")
        
        print(f"\n--- Component Latency Breakdown ---")
        latency_stats = report["latency_breakdown"]
        if latency_stats:
            for component, stats in latency_stats.items():
                print(f"\n{component}:")
                print(f"  Count: {stats.get('count', 0)}")
                print(f"  Mean: {stats.get('mean', 0):.3f}s")
                print(f"  P95: {stats.get('p95', 0):.3f}s")
                print(f"  P99: {stats.get('p99', 0):.3f}s")
        else:
            print("No latency data collected yet")
        
        print(f"\n--- Recommendations ---")
        for i, rec in enumerate(report["recommendations"], 1):
            print(f"{i}. {rec}")
        
        print(f"\n{'='*60}\n")


async def main():
    """Run performance evaluation."""
    # Test queries - diverse set
    test_queries = [
        "Find me wireless headphones",
        "What's the price of laptops with 16GB RAM?",
        "Show me top 5 AI Engineering books",
        "Find face cream for dark spots under $25",
        "Search for gaming laptops with NVIDIA GPU",
        "Tell me about the best selling products"
    ]
    
    evaluator = PerformanceEvaluator()
    report = await evaluator.run_evaluation(test_queries)
    evaluator.print_report(report)
    
    # Print detailed latency breakdown from query results
    successful_results = [r for r in report["query_results"] if r["success"]]
    if successful_results:
        print("\n" + "="*60)
        print("DETAILED LATENCY BREAKDOWN (Per Query)")
        print("="*60)
        for i, result in enumerate(successful_results, 1):
            if result.get("latency_breakdown"):
                print(f"\nQuery {i}: {result['query'][:60]}")
                print(f"  Total: {result['response_time']:.3f}s")
                for component, latency in sorted(result["latency_breakdown"].items(), key=lambda x: x[1], reverse=True):
                    if component != "total":
                        percentage = (latency / result["latency_breakdown"].get("total", 1)) * 100
                        print(f"  {component}: {latency:.3f}s ({percentage:.1f}%)")
        
        # Aggregate component latencies
        print("\n" + "="*60)
        print("AGGREGATED COMPONENT LATENCIES")
        print("="*60)
        component_times = {}
        for result in successful_results:
            if result.get("latency_breakdown"):
                for component, latency in result["latency_breakdown"].items():
                    if component != "total":
                        if component not in component_times:
                            component_times[component] = []
                        component_times[component].append(latency)
        
        for component, times in sorted(component_times.items(), key=lambda x: sum(x[1])/len(x[1]), reverse=True):
            avg = sum(times) / len(times)
            print(f"{component}:")
            print(f"  Average: {avg:.3f}s")
            print(f"  Min: {min(times):.3f}s, Max: {max(times):.3f}s")
            print(f"  Count: {len(times)}")
    
    # Also print latency dashboard (if data available)
    print("\n" + "="*60)
    print("LATENCY DASHBOARD")
    print("="*60)
    dashboard = LatencyDashboard()
    dashboard.print_report()
    
    # Save report to file
    report_file = f"evaluation/performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n[FILE] Report saved to: {report_file}")


if __name__ == "__main__":
    asyncio.run(main())

