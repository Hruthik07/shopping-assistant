"""Performance test suite for latency and throughput."""
import asyncio
import httpx
from typing import List, Dict, Any
from statistics import mean, median, stdev
from src.analytics.logger import logger


class PerformanceTestSuite:
    """Test suite for performance evaluation."""
    
    def __init__(self, base_url: str = "http://localhost:3565"):
        """Initialize performance test suite.
        
        Args:
            base_url: Base URL of the API server
        """
        self.base_url = base_url
    
    async def test_latency(
        self,
        query: str,
        session_id: str = None,
        iterations: int = 1
    ) -> Dict[str, Any]:
        """Test query latency.
        
        Args:
            query: Test query
            session_id: Optional session ID
            iterations: Number of iterations to run
        
        Returns:
            Dictionary with latency metrics
        """
        latencies = []
        ttfts = []
        
        for i in range(iterations):
            start_time = asyncio.get_event_loop().time()
            
            try:
                async with httpx.AsyncClient(timeout=120.0) as client:
                    response = await client.post(
                        f"{self.base_url}/api/chat/",
                        json={"message": query, "session_id": session_id},
                        headers={"Content-Type": "application/json"}
                    )
                    response.raise_for_status()
                    result = response.json()
                
                elapsed = asyncio.get_event_loop().time() - start_time
                latencies.append(elapsed)
                
                latency_breakdown = result.get("latency_breakdown", {})
                ttft = latency_breakdown.get("ttft", 0)
                if ttft > 0:
                    ttfts.append(ttft)
                
            except Exception as e:
                logger.error(f"Latency test iteration {i+1} failed: {e}")
                elapsed = asyncio.get_event_loop().time() - start_time
                latencies.append(elapsed)
        
        return {
            "query": query,
            "iterations": iterations,
            "total_time": {
                "mean": mean(latencies) if latencies else 0,
                "median": median(latencies) if latencies else 0,
                "min": min(latencies) if latencies else 0,
                "max": max(latencies) if latencies else 0,
                "std": stdev(latencies) if len(latencies) > 1 else 0
            },
            "ttft": {
                "mean": mean(ttfts) if ttfts else 0,
                "median": median(ttfts) if ttfts else 0,
                "min": min(ttfts) if ttfts else 0,
                "max": max(ttfts) if ttfts else 0,
                "std": stdev(ttfts) if len(ttfts) > 1 else 0
            },
            "passed": mean(latencies) < 5.0 if latencies else False  # Target: < 5 seconds
        }
    
    async def test_throughput(
        self,
        queries: List[str],
        concurrent: int = 5
    ) -> Dict[str, Any]:
        """Test system throughput with concurrent requests.
        
        Args:
            queries: List of test queries
            concurrent: Number of concurrent requests
        
        Returns:
            Dictionary with throughput metrics
        """
        start_time = asyncio.get_event_loop().time()
        
        async def run_query(query: str):
            try:
                async with httpx.AsyncClient(timeout=120.0) as client:
                    response = await client.post(
                        f"{self.base_url}/api/chat/",
                        json={"message": query},
                        headers={"Content-Type": "application/json"}
                    )
                    response.raise_for_status()
                    return True
            except Exception as e:
                logger.error(f"Throughput test query failed: {e}")
                return False
        
        # Run queries in batches
        results = []
        for i in range(0, len(queries), concurrent):
            batch = queries[i:i+concurrent]
            batch_results = await asyncio.gather(*[run_query(q) for q in batch])
            results.extend(batch_results)
        
        elapsed = asyncio.get_event_loop().time() - start_time
        successful = sum(1 for r in results if r)
        
        return {
            "total_queries": len(queries),
            "concurrent": concurrent,
            "successful": successful,
            "failed": len(queries) - successful,
            "total_time": elapsed,
            "queries_per_second": len(queries) / elapsed if elapsed > 0 else 0,
            "success_rate": successful / len(queries) if queries else 0,
            "passed": (successful / len(queries)) >= 0.95 if queries else False  # 95% success rate
        }
    
    async def test_component_breakdown(
        self,
        query: str,
        session_id: str = None
    ) -> Dict[str, Any]:
        """Test latency breakdown by component.
        
        Args:
            query: Test query
            session_id: Optional session ID
        
        Returns:
            Dictionary with component-level latency breakdown
        """
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/chat/",
                    json={"message": query, "session_id": session_id},
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                result = response.json()
            
            latency_breakdown = result.get("latency_breakdown", {})
            
            # Identify bottlenecks
            bottlenecks = []
            total_time = sum(latency_breakdown.values())
            
            for component, time in latency_breakdown.items():
                percentage = (time / total_time * 100) if total_time > 0 else 0
                if percentage > 30:  # Component taking > 30% of total time
                    bottlenecks.append({
                        "component": component,
                        "time": time,
                        "percentage": percentage
                    })
            
            return {
                "query": query,
                "latency_breakdown": latency_breakdown,
                "total_time": total_time,
                "bottlenecks": bottlenecks,
                "passed": len(bottlenecks) == 0  # No major bottlenecks
            }
            
        except Exception as e:
            logger.error(f"Component breakdown test failed: {e}")
            return {
                "query": query,
                "error": str(e),
                "passed": False
            }

