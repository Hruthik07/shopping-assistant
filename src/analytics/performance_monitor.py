"""Performance monitoring and regression detection."""

from typing import Dict, Any, List, Optional
from collections import deque
import time
import statistics

from src.analytics.logger import logger


class PerformanceMonitor:
    """Monitor performance metrics and detect regressions."""

    def __init__(self, window_size: int = 100):
        """Initialize performance monitor.

        Args:
            window_size: Number of recent requests to keep in memory
        """
        self.window_size = window_size
        self.latency_history: deque = deque(maxlen=window_size)
        self.request_times: Dict[str, float] = {}
        self.baseline_p50: Optional[float] = None
        self.baseline_p95: Optional[float] = None
        self.baseline_p99: Optional[float] = None

    def record_latency(self, latency: float, request_id: Optional[str] = None):
        """Record a latency measurement.

        Args:
            latency: Latency in seconds
            request_id: Optional request ID for tracking
        """
        self.latency_history.append(latency)

        if request_id:
            self.request_times[request_id] = latency

        # Update baseline if we have enough data
        if len(self.latency_history) >= 50 and self.baseline_p50 is None:
            self._update_baseline()

    def _update_baseline(self):
        """Update baseline percentiles from current history."""
        if len(self.latency_history) < 10:
            return

        sorted_latencies = sorted(self.latency_history)
        n = len(sorted_latencies)

        self.baseline_p50 = sorted_latencies[int(n * 0.50)]
        self.baseline_p95 = sorted_latencies[int(n * 0.95)]
        self.baseline_p99 = sorted_latencies[int(n * 0.99)]

        logger.info(
            f"Performance baseline updated: P50={self.baseline_p50:.2f}s, "
            f"P95={self.baseline_p95:.2f}s, P99={self.baseline_p99:.2f}s"
        )

    def get_percentiles(self) -> Dict[str, float]:
        """Get current latency percentiles.

        Returns:
            Dictionary with P50, P95, P99 latencies
        """
        if len(self.latency_history) < 10:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "sample_size": len(self.latency_history)}

        sorted_latencies = sorted(self.latency_history)
        n = len(sorted_latencies)

        percentiles = {
            "p50": sorted_latencies[int(n * 0.50)],
            "p95": sorted_latencies[int(n * 0.95)],
            "p99": sorted_latencies[int(n * 0.99)],
            "min": sorted_latencies[0],
            "max": sorted_latencies[-1],
            "mean": statistics.mean(sorted_latencies),
            "average": statistics.mean(sorted_latencies),  # Alias for mean
            "sample_size": n,
        }

        # Export to CloudWatch (async, non-blocking)
        try:
            from src.analytics.cloudwatch_exporter import cloudwatch_exporter
            import asyncio

            # Schedule async export (fire and forget) - safer event loop handling
            try:
                loop = asyncio.get_running_loop()
                asyncio.create_task(cloudwatch_exporter.export_latency_metrics(percentiles))
            except RuntimeError:
                # No running event loop, try to get existing loop
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.create_task(cloudwatch_exporter.export_latency_metrics(percentiles))
                    else:
                        asyncio.run(cloudwatch_exporter.export_latency_metrics(percentiles))
                except Exception:
                    pass  # Skip export if event loop handling fails
        except Exception:
            pass  # Graceful degradation if CloudWatch unavailable

        return percentiles

    def check_regression(
        self, threshold_p95: Optional[float] = None, threshold_p99: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """Check for performance regressions.

        Args:
            threshold_p95: Optional P95 threshold in seconds
            threshold_p99: Optional P99 threshold in seconds

        Returns:
            Alert dictionary if regression detected, None otherwise
        """
        if len(self.latency_history) < 10:
            return None

        percentiles = self.get_percentiles()
        alerts = []

        # Check against baseline if available
        if self.baseline_p95 and percentiles["p95"] > self.baseline_p95 * 1.2:
            alerts.append(
                {
                    "metric": "p95_latency",
                    "baseline": self.baseline_p95,
                    "current": percentiles["p95"],
                    "increase_percent": (
                        (percentiles["p95"] - self.baseline_p95) / self.baseline_p95
                    )
                    * 100,
                    "severity": (
                        "high" if percentiles["p95"] > self.baseline_p95 * 1.5 else "medium"
                    ),
                }
            )

        if self.baseline_p99 and percentiles["p99"] > self.baseline_p99 * 1.2:
            alerts.append(
                {
                    "metric": "p99_latency",
                    "baseline": self.baseline_p99,
                    "current": percentiles["p99"],
                    "increase_percent": (
                        (percentiles["p99"] - self.baseline_p99) / self.baseline_p99
                    )
                    * 100,
                    "severity": (
                        "high" if percentiles["p99"] > self.baseline_p99 * 1.5 else "medium"
                    ),
                }
            )

        # Check against absolute thresholds
        if threshold_p95 and percentiles["p95"] > threshold_p95:
            alerts.append(
                {
                    "metric": "p95_latency_threshold",
                    "threshold": threshold_p95,
                    "current": percentiles["p95"],
                    "exceeded_by": percentiles["p95"] - threshold_p95,
                    "severity": "high",
                }
            )

        if threshold_p99 and percentiles["p99"] > threshold_p99:
            alerts.append(
                {
                    "metric": "p99_latency_threshold",
                    "threshold": threshold_p99,
                    "current": percentiles["p99"],
                    "exceeded_by": percentiles["p99"] - threshold_p99,
                    "severity": "high",
                }
            )

        if alerts:
            return {
                "alert": True,
                "timestamp": time.time(),
                "alerts": alerts,
                "current_percentiles": percentiles,
            }

        return None

    def get_slow_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get slowest queries from recent history.

        Args:
            limit: Number of slow queries to return

        Returns:
            List of slow query information
        """
        # This would need to be enhanced to track query text
        # For now, return latency information
        sorted_latencies = sorted(self.latency_history, reverse=True)

        return [{"latency": lat, "rank": i + 1} for i, lat in enumerate(sorted_latencies[:limit])]

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics.

        Returns:
            Dictionary with performance stats
        """
        percentiles = self.get_percentiles()

        return {
            "percentiles": percentiles,
            "baseline": {
                "p50": self.baseline_p50,
                "p95": self.baseline_p95,
                "p99": self.baseline_p99,
            },
            "sample_size": len(self.latency_history),
            "window_size": self.window_size,
        }


# Global performance monitor instance
performance_monitor = PerformanceMonitor()
