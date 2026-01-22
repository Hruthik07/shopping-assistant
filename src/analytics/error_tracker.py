"""Error tracking and alerting."""
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import time

from src.analytics.logger import logger


class ErrorTracker:
    """Track errors and generate alerts."""
    
    def __init__(self):
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.error_history: list = []
        self.alert_thresholds = {
            "rate_limit": 5,  # Alert if 5+ rate limit errors in 1 minute
            "timeout": 3,  # Alert if 3+ timeouts in 1 minute
            "llm_error": 3,  # Alert if 3+ LLM errors in 1 minute
            "database_error": 2,  # Alert if 2+ DB errors in 1 minute
            "cache_error": 5,  # Alert if 5+ cache errors in 1 minute (less critical)
        }
        self.window_seconds = 60  # 1 minute window
    
    def record_error(
        self,
        error_type: str,
        error_message: str,
        context: Optional[Dict[str, Any]] = None
    ):
        """Record an error and check for alerts."""
        timestamp = time.time()
        
        # Add to history
        error_entry = {
            "timestamp": timestamp,
            "type": error_type,
            "message": error_message,
            "context": context or {}
        }
        self.error_history.append(error_entry)
        
        # Increment count
        self.error_counts[error_type] += 1
        
        # Clean old history (keep last hour)
        cutoff = timestamp - 3600
        self.error_history = [e for e in self.error_history if e["timestamp"] > cutoff]
        
        # Check for alerts
        self._check_alerts(error_type, timestamp)
        
        # Log error with context
        logger.error(
            f"Error recorded: {error_type} - {error_message}",
            extra={"error_type": error_type, "context": context}
        )
    
    def _check_alerts(self, error_type: str, current_time: float):
        """Check if error threshold is exceeded and alert."""
        threshold = self.alert_thresholds.get(error_type)
        if not threshold:
            return
        
        # Count errors in the last window
        window_start = current_time - self.window_seconds
        recent_errors = [
            e for e in self.error_history
            if e["type"] == error_type and e["timestamp"] > window_start
        ]
        
        if len(recent_errors) >= threshold:
            logger.warning(
                f"ALERT: {error_type} threshold exceeded - {len(recent_errors)} errors in last {self.window_seconds}s",
                extra={
                    "error_type": error_type,
                    "count": len(recent_errors),
                    "threshold": threshold,
                    "window_seconds": self.window_seconds
                }
            )
    
    def get_error_stats(self, window_seconds: int = 300) -> Dict[str, Any]:
        """Get error statistics for the last window."""
        cutoff = time.time() - window_seconds
        recent_errors = [e for e in self.error_history if e["timestamp"] > cutoff]
        
        error_types = defaultdict(int)
        for error in recent_errors:
            error_types[error["type"]] += 1
        
        stats = {
            "window_seconds": window_seconds,
            "total_errors": len(recent_errors),
            "error_types": dict(error_types),
            "error_rate": len(recent_errors) / (window_seconds / 60) if window_seconds > 0 else 0  # errors per minute
        }
        
        # Export to CloudWatch (async, non-blocking)
        try:
            from src.analytics.cloudwatch_exporter import cloudwatch_exporter
            import asyncio
            # Schedule async export (fire and forget) - safer event loop handling
            try:
                loop = asyncio.get_running_loop()
                asyncio.create_task(cloudwatch_exporter.export_error_metrics(stats))
            except RuntimeError:
                # No running event loop, try to get existing loop
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.create_task(cloudwatch_exporter.export_error_metrics(stats))
                    else:
                        asyncio.run(cloudwatch_exporter.export_error_metrics(stats))
                except Exception:
                    pass  # Skip export if event loop handling fails
        except Exception:
            pass  # Graceful degradation if CloudWatch unavailable
        
        return stats
    
    def get_recent_errors(self, limit: int = 10) -> list:
        """Get most recent errors."""
        return sorted(
            self.error_history,
            key=lambda x: x["timestamp"],
            reverse=True
        )[:limit]


# Global error tracker instance
error_tracker = ErrorTracker()
