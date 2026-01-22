"""Metrics endpoint for monitoring and observability."""
from fastapi import APIRouter
from typing import Dict, Any, Optional
import time

from src.analytics.logger import logger
from src.analytics.latency_tracker import latency_tracker
from src.analytics.cost_tracker import cost_tracker
from src.analytics.performance_monitor import performance_monitor
from src.analytics.tracker import tracker
from src.utils.cache import cache_service

router = APIRouter(prefix="/api/metrics", tags=["metrics"])


@router.get("")
async def get_metrics() -> Dict[str, Any]:
    """Get Prometheus-style metrics for monitoring."""
    metrics = {
        "timestamp": time.time(),
        "cache": {},
        "latency": {},
        "system": {}
    }
    
    # Cache metrics
    try:
        cache_stats = cache_service.get_cache_stats()
        metrics["cache"] = {
            "hits": cache_stats.get("hits", 0),
            "misses": cache_stats.get("misses", 0),
            "sets": cache_stats.get("sets", 0),
            "errors": cache_stats.get("errors", 0),
            "total_requests": cache_stats.get("total_requests", 0),
            "hit_rate": cache_stats.get("hit_rate", 0.0),
            "enabled": cache_service.enabled
        }
        
        # Get Redis stats if available
        try:
            redis_stats = await cache_service.get_stats()
            if isinstance(redis_stats, dict) and "redis" in redis_stats:
                metrics["cache"]["redis"] = redis_stats["redis"]
        except Exception as e:
            logger.debug(f"Could not get Redis stats: {e}")
    except Exception as e:
        logger.warning(f"Error getting cache metrics: {e}")
        metrics["cache"] = {"error": str(e)}
    
    # Latency metrics (if available from tracker)
    try:
        # Get recent latency statistics
        # Note: latency_tracker may need to expose aggregated stats
        metrics["latency"] = {
            "tracker_available": hasattr(latency_tracker, 'get_stats'),
            "note": "Detailed latency stats available via Langfuse"
        }
    except Exception as e:
        logger.debug(f"Error getting latency metrics: {e}")
        metrics["latency"] = {"error": str(e)}
    
    # System metrics
    try:
        import psutil
        process = psutil.Process()
        metrics["system"] = {
            "cpu_percent": process.cpu_percent(interval=0.1),
            "memory_mb": process.memory_info().rss / 1024 / 1024,
            "threads": process.num_threads(),
            "open_files": len(process.open_files()) if hasattr(process, 'open_files') else 0
        }
    except ImportError:
        metrics["system"] = {
            "note": "psutil not available - install for detailed system metrics"
        }
    except Exception as e:
        logger.debug(f"Error getting system metrics: {e}")
        metrics["system"] = {"error": str(e)}
    
    return metrics


@router.get("/cache")
async def get_cache_metrics() -> Dict[str, Any]:
    """Get detailed cache metrics."""
    try:
        internal_stats = cache_service.get_cache_stats()
        redis_stats = await cache_service.get_stats()
        
        return {
            "internal": internal_stats,
            "redis": redis_stats.get("redis") if isinstance(redis_stats, dict) else None,
            "enabled": cache_service.enabled
        }
    except Exception as e:
        logger.error(f"Error getting cache metrics: {e}")
        return {
            "error": str(e),
            "enabled": cache_service.enabled
        }


@router.get("/latency")
async def get_latency_metrics() -> Dict[str, Any]:
    """Get latency metrics summary."""
    try:
        # Get performance monitor stats
        perf_stats = performance_monitor.get_stats()
        percentiles = performance_monitor.get_percentiles()
        
        # Check for regressions
        regression_alert = performance_monitor.check_regression(
            threshold_p95=20.0,  # Alert if P95 > 20s
            threshold_p99=30.0   # Alert if P99 > 30s
        )
        
        return {
            "percentiles": percentiles,
            "baseline": perf_stats.get("baseline", {}),
            "sample_size": perf_stats.get("sample_size", 0),
            "regression_alert": regression_alert,
            "note": "Detailed latency stats also available via Langfuse"
        }
    except Exception as e:
        logger.error(f"Error getting latency metrics: {e}")
        return {
            "error": str(e),
            "note": "Detailed latency metrics are tracked via Langfuse"
        }


@router.get("/cost")
async def get_cost_metrics(
    days: int = 7,
    model: Optional[str] = None
) -> Dict[str, Any]:
    """Get cost metrics and statistics.
    
    Args:
        days: Number of days to look back (default: 7)
        model: Optional model filter
    """
    try:
        stats = cost_tracker.get_cost_stats(days=days, model=model)
        recent_costs = cost_tracker.get_recent_costs(limit=10)
        
        result = {
            "statistics": stats,
            "recent_costs": recent_costs,
            "all_time": {
                "total_cost": round(cost_tracker.total_cost, 4),
                "total_requests": cost_tracker.request_count,
                "total_input_tokens": cost_tracker.total_input_tokens,
                "total_output_tokens": cost_tracker.total_output_tokens
            }
        }
        
        # Add Bedrock costs if enabled
        try:
            from src.analytics.bedrock_cost_tracker import bedrock_cost_tracker
            if bedrock_cost_tracker.enabled:
                bedrock_costs = await bedrock_cost_tracker.get_bedrock_costs()
                result["bedrock"] = bedrock_costs
        except Exception as e:
            logger.debug(f"Could not get Bedrock costs: {e}")
        
        return result
    except Exception as e:
        logger.error(f"Error getting cost metrics: {e}")
        return {
            "error": str(e)
        }


@router.get("/deals")
async def get_deal_metrics() -> Dict[str, Any]:
    """Get deal detection and price comparison metrics."""
    try:
        metrics = tracker.get_metrics()
        
        return {
            "deal_detection": {
                "deals_detected": metrics.get("deals_detected", 0),
                "total_products_analyzed": metrics.get("total_products_analyzed", 0),
                "deal_detection_rate": round(metrics.get("deal_detection_rate", 0.0), 2),
                "average_savings_percent": round(metrics.get("average_savings_percent", 0.0), 2),
                "total_savings_amount": round(metrics.get("total_savings_amount", 0.0), 2)
            },
            "price_comparison": {
                "price_comparisons": metrics.get("price_comparisons", 0),
                "products_with_multiple_retailers": metrics.get("products_with_multiple_retailers", 0),
                "average_retailers_per_product": round(metrics.get("average_retailers_per_product", 0.0), 2),
                "average_price_difference": round(metrics.get("average_price_difference", 0.0), 2)
            },
            "api_performance": {
                "total_api_calls": metrics.get("api_calls_total", 0),
                "successful_calls": metrics.get("api_calls_successful", 0),
                "failed_calls": metrics.get("api_calls_failed", 0),
                "success_rate": round(metrics.get("api_success_rate", 0.0), 2)
            }
        }
    except Exception as e:
        logger.error(f"Error getting deal metrics: {e}")
        return {
            "error": str(e)
        }
