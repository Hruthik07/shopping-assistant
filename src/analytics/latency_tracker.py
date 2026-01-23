"""Detailed latency tracking for performance optimization."""

import time
from typing import Dict, Any, Optional, List
from contextlib import contextmanager
from collections import defaultdict
from src.analytics.logger import logger
import statistics
import uuid


class LatencyTracker:
    """Track latency at component level."""

    def __init__(self):
        self.component_times: Dict[str, List[float]] = defaultdict(list)
        self.request_times: Dict[str, Dict[str, float]] = {}
        self.current_request_id: Optional[str] = None

    @contextmanager
    def track_component(self, component_name: str, request_id: Optional[str] = None):
        """Context manager to track component latency."""
        start = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start
            self.component_times[component_name].append(elapsed)

            if request_id:
                if request_id not in self.request_times:
                    self.request_times[request_id] = {}
                self.request_times[request_id][component_name] = elapsed

            logger.debug(f"{component_name} took {elapsed:.3f}s")

    def get_component_stats(self, component_name: str) -> Dict[str, float]:
        """Get statistics for a component."""
        times = self.component_times.get(component_name, [])
        if not times:
            return {}

        return {
            "count": len(times),
            "mean": statistics.mean(times),
            "median": statistics.median(times),
            "p50": statistics.median(times),
            "p95": self._percentile(times, 95),
            "p99": self._percentile(times, 99),
            "min": min(times),
            "max": max(times),
            "std": statistics.stdev(times) if len(times) > 1 else 0.0,
        }

    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all components."""
        return {
            component: self.get_component_stats(component)
            for component in self.component_times.keys()
        }

    def get_request_breakdown(self, request_id: str) -> Dict[str, float]:
        """Get latency breakdown for a specific request."""
        return self.request_times.get(request_id, {})

    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]

    def reset(self):
        """Reset all tracking data."""
        self.component_times.clear()
        self.request_times.clear()

    def generate_request_id(self) -> str:
        """Generate a unique request ID."""
        return str(uuid.uuid4())

    def track_ttft(self, request_id: str, ttft: float):
        """Track Time To First Token (TTFT)."""
        if request_id not in self.request_times:
            self.request_times[request_id] = {}
        self.request_times[request_id]["ttft"] = ttft

        # Also track in component times for aggregate stats
        self.component_times["ttft"].append(ttft)
        logger.debug(f"TTFT for {request_id}: {ttft:.3f}s")


# Global latency tracker
latency_tracker = LatencyTracker()
