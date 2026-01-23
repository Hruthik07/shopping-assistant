"""Load testing suite for capacity planning."""

import asyncio
import httpx
import time
from typing import List, Dict, Any
from statistics import mean, median
from collections import defaultdict
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analytics.logger import logger


class LoadTester:
    """Load testing tool for the shopping assistant API."""

    def __init__(self, base_url: str = "http://localhost:3565"):
        """Initialize load tester.

        Args:
            base_url: API server base URL
        """
        self.base_url = base_url
        self.test_queries = [
            "Find me wireless headphones under $100",
            "What are the best laptops for gaming?",
            "Show me running shoes between $50 and $90",
            "Find me a fitness tracker",
            "I need a water bottle",
        ]

    async def run_single_request(
        self, query: str, session_id: str, client: httpx.AsyncClient
    ) -> Dict[str, Any]:
        """Run a single request and measure latency.

        Args:
            query: Test query
            session_id: Session ID
            client: HTTP client

        Returns:
            Request result dictionary
        """
        start_time = time.time()

        try:
            response = await client.post(
                f"{self.base_url}/api/chat/",
                json={"message": query, "session_id": session_id},
                headers={"Content-Type": "application/json"},
                timeout=60.0,
            )

            elapsed = time.time() - start_time

            if response.status_code == 200:
                result = response.json()
                return {
                    "success": True,
                    "latency": elapsed,
                    "status_code": response.status_code,
                    "response_length": len(result.get("response", "")),
                    "products_count": len(result.get("products", [])),
                    "cached": result.get("cached", False),
                }
            else:
                return {
                    "success": False,
                    "latency": elapsed,
                    "status_code": response.status_code,
                    "error": response.text[:200],
                }
        except httpx.TimeoutException:
            return {"success": False, "latency": time.time() - start_time, "error": "timeout"}
        except Exception as e:
            return {"success": False, "latency": time.time() - start_time, "error": str(e)}

    async def run_concurrent_load(
        self, concurrent_users: int, requests_per_user: int = 1, ramp_up_seconds: int = 0
    ) -> Dict[str, Any]:
        """Run load test with concurrent users.

        Args:
            concurrent_users: Number of concurrent users
            requests_per_user: Number of requests per user
            ramp_up_seconds: Seconds to ramp up to full load

        Returns:
            Load test results
        """
        print(f"\n{'='*70}")
        print(f"LOAD TEST: {concurrent_users} concurrent users, {requests_per_user} requests each")
        print(f"{'='*70}\n")

        results: List[Dict[str, Any]] = []
        start_time = time.time()

        async def user_simulation(user_id: int):
            """Simulate a single user making requests."""
            async with httpx.AsyncClient(timeout=60.0) as client:
                user_results = []
                session_id = f"load_test_user_{user_id}"

                for req_num in range(requests_per_user):
                    # Select query (round-robin)
                    query = self.test_queries[req_num % len(self.test_queries)]

                    # Ramp up delay
                    if ramp_up_seconds > 0:
                        delay = (user_id * ramp_up_seconds) / concurrent_users
                        await asyncio.sleep(delay)

                    result = await self.run_single_request(query, session_id, client)
                    result["user_id"] = user_id
                    result["request_num"] = req_num
                    user_results.append(result)
                    results.append(result)

                return user_results

        # Run concurrent users
        tasks = [user_simulation(i) for i in range(concurrent_users)]
        await asyncio.gather(*tasks)

        total_time = time.time() - start_time

        # Analyze results
        successful = [r for r in results if r.get("success")]
        failed = [r for r in results if not r.get("success")]

        latencies = [r["latency"] for r in successful] if successful else []

        analysis = {
            "concurrent_users": concurrent_users,
            "requests_per_user": requests_per_user,
            "total_requests": len(results),
            "successful_requests": len(successful),
            "failed_requests": len(failed),
            "success_rate": len(successful) / len(results) * 100 if results else 0,
            "total_time": total_time,
            "requests_per_second": len(results) / total_time if total_time > 0 else 0,
            "latency": {
                "min": min(latencies) if latencies else 0,
                "max": max(latencies) if latencies else 0,
                "mean": mean(latencies) if latencies else 0,
                "median": median(latencies) if latencies else 0,
                "p95": self._percentile(latencies, 95) if latencies else 0,
                "p99": self._percentile(latencies, 99) if latencies else 0,
            },
            "errors": self._analyze_errors(failed),
            "cache_stats": self._analyze_cache(successful),
        }

        return analysis

    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]

    def _analyze_errors(self, failed: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze error patterns."""
        error_types = defaultdict(int)
        for result in failed:
            error = result.get("error", "unknown")
            error_types[error] += 1

        return {"total_errors": len(failed), "error_types": dict(error_types)}

    def _analyze_cache(self, successful: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze cache performance."""
        cached = sum(1 for r in successful if r.get("cached", False))
        total = len(successful)

        return {
            "cache_hits": cached,
            "cache_misses": total - cached,
            "cache_hit_rate": (cached / total * 100) if total > 0 else 0,
        }

    async def run_scenario(
        self, scenario_name: str, concurrent_users: int, requests_per_user: int = 1
    ) -> Dict[str, Any]:
        """Run a specific load test scenario.

        Args:
            scenario_name: Name of the scenario
            concurrent_users: Number of concurrent users
            requests_per_user: Requests per user

        Returns:
            Scenario results
        """
        print(f"\nRunning scenario: {scenario_name}")
        results = await self.run_concurrent_load(
            concurrent_users=concurrent_users, requests_per_user=requests_per_user
        )

        print(f"\nResults for {scenario_name}:")
        print(f"  Success Rate: {results['success_rate']:.1f}%")
        print(f"  Requests/sec: {results['requests_per_second']:.2f}")
        print(f"  Avg Latency: {results['latency']['mean']:.2f}s")
        print(f"  P95 Latency: {results['latency']['p95']:.2f}s")
        print(f"  P99 Latency: {results['latency']['p99']:.2f}s")
        print(f"  Cache Hit Rate: {results['cache_stats']['cache_hit_rate']:.1f}%")

        return results

    async def run_capacity_test(self) -> Dict[str, Any]:
        """Run capacity test with increasing load.

        Returns:
            Capacity test results
        """
        print("\n" + "=" * 70)
        print("CAPACITY TEST - Increasing Load")
        print("=" * 70)

        scenarios = [
            ("Light Load", 10, 1),
            ("Medium Load", 25, 2),
            ("Heavy Load", 50, 2),
            ("Stress Test", 100, 1),
        ]

        all_results = {}

        for scenario_name, users, requests in scenarios:
            try:
                results = await self.run_scenario(scenario_name, users, requests)
                all_results[scenario_name] = results

                # Wait between scenarios
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"Error in scenario {scenario_name}: {e}")
                all_results[scenario_name] = {"error": str(e)}

        # Generate summary
        summary = {
            "scenarios": all_results,
            "recommendations": self._generate_recommendations(all_results),
        }

        return summary

    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate capacity planning recommendations.

        Args:
            results: Scenario results

        Returns:
            List of recommendations
        """
        recommendations = []

        for scenario_name, scenario_results in results.items():
            if "error" in scenario_results:
                continue

            success_rate = scenario_results.get("success_rate", 100)
            p95_latency = scenario_results.get("latency", {}).get("p95", 0)

            if success_rate < 95:
                recommendations.append(
                    f"{scenario_name}: Success rate {success_rate:.1f}% is below 95% threshold"
                )

            if p95_latency > 20:
                recommendations.append(
                    f"{scenario_name}: P95 latency {p95_latency:.1f}s exceeds 20s target"
                )

        if not recommendations:
            recommendations.append("All scenarios passed performance thresholds")

        return recommendations


async def main():
    """Run load tests."""
    tester = LoadTester()

    # Run capacity test
    capacity_results = await tester.run_capacity_test()

    print("\n" + "=" * 70)
    print("CAPACITY TEST SUMMARY")
    print("=" * 70)

    for scenario, results in capacity_results.get("scenarios", {}).items():
        if "error" not in results:
            print(f"\n{scenario}:")
            print(f"  Success: {results['success_rate']:.1f}%")
            print(f"  Throughput: {results['requests_per_second']:.2f} req/s")
            print(f"  P95 Latency: {results['latency']['p95']:.2f}s")

    print("\nRecommendations:")
    for rec in capacity_results.get("recommendations", []):
        print(f"  - {rec}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
