"""CloudWatch metrics exporter for AWS Bedrock deployment."""

import time
from typing import Dict, Any, List, Optional
from collections import defaultdict
from datetime import datetime
import asyncio

from src.analytics.logger import logger
from src.utils.config import settings


class CloudWatchExporter:
    """Export application metrics to AWS CloudWatch."""

    def __init__(self):
        """Initialize CloudWatch exporter."""
        self.enabled = getattr(settings, "cloudwatch_enabled", False)
        self.namespace = getattr(settings, "cloudwatch_namespace", "ShoppingAssistant/Application")
        self.region = getattr(settings, "aws_region", "us-east-1")
        self.client = None
        self.metric_buffer: List[Dict[str, Any]] = []
        self.buffer_lock = asyncio.Lock()
        self.last_flush = time.time()
        self.flush_interval = 60  # Flush every 60 seconds

        if self.enabled:
            self._initialize_client()

    def _initialize_client(self):
        """Initialize boto3 CloudWatch client."""
        try:
            import boto3
            from botocore.exceptions import ClientError, NoCredentialsError

            self.boto3 = boto3
            self.ClientError = ClientError
            self.NoCredentialsError = NoCredentialsError

            # Initialize CloudWatch client
            self.client = boto3.client("cloudwatch", region_name=self.region)
            logger.info(f"CloudWatch exporter initialized for region: {self.region}")
        except ImportError:
            logger.warning(
                "boto3 not installed. CloudWatch export disabled. Install with: pip install boto3"
            )
            self.enabled = False
        except Exception as e:
            logger.warning(
                f"Failed to initialize CloudWatch client: {e}. CloudWatch export disabled."
            )
            self.enabled = False

    async def put_metric(
        self,
        metric_name: str,
        value: float,
        unit: str = "None",
        dimensions: Optional[Dict[str, str]] = None,
        timestamp: Optional[datetime] = None,
    ):
        """Put a metric to CloudWatch (buffered).

        Args:
            metric_name: Name of the metric
            value: Metric value
            unit: CloudWatch unit (Count, Seconds, Percent, None, etc.)
            dimensions: Optional dimensions (key-value pairs)
            timestamp: Optional timestamp (defaults to now)
        """
        if not self.enabled or not self.client:
            return

        metric_data = {
            "MetricName": metric_name,
            "Value": value,
            "Unit": unit,
            "Timestamp": timestamp or datetime.utcnow(),
        }

        if dimensions:
            metric_data["Dimensions"] = [{"Name": k, "Value": v} for k, v in dimensions.items()]

        async with self.buffer_lock:
            self.metric_buffer.append(metric_data)

        # Auto-flush if buffer is large or time elapsed
        current_time = time.time()
        if len(self.metric_buffer) >= 20 or (current_time - self.last_flush) >= self.flush_interval:
            await self.flush()

    async def flush(self):
        """Flush buffered metrics to CloudWatch."""
        if not self.enabled or not self.client:
            return

        async with self.buffer_lock:
            if not self.metric_buffer:
                return

            # CloudWatch allows max 20 metrics per PutMetricData call
            batches = [
                self.metric_buffer[i : i + 20] for i in range(0, len(self.metric_buffer), 20)
            ]

            for batch in batches:
                try:
                    self.client.put_metric_data(Namespace=self.namespace, MetricData=batch)
                    logger.debug(f"Sent {len(batch)} metrics to CloudWatch")
                except self.NoCredentialsError:
                    logger.warning("AWS credentials not found. CloudWatch export disabled.")
                    self.enabled = False
                    break
                except self.ClientError as e:
                    error_code = e.response.get("Error", {}).get("Code", "")
                    if error_code == "AccessDenied":
                        logger.warning("CloudWatch access denied. Check IAM permissions.")
                    else:
                        logger.warning(f"CloudWatch PutMetricData failed: {e}")
                    # Don't disable on transient errors
                except Exception as e:
                    logger.warning(f"Unexpected error sending metrics to CloudWatch: {e}")

            self.metric_buffer.clear()
            self.last_flush = time.time()

    async def export_cache_metrics(self, cache_stats: Dict[str, Any]):
        """Export cache metrics to CloudWatch.

        Args:
            cache_stats: Cache statistics dictionary
        """
        if not self.enabled:
            return

        try:
            # Cache hit rate (percentage)
            hit_rate = cache_stats.get("hit_rate", 0.0)
            await self.put_metric(
                "CacheHitRate",
                hit_rate * 100,  # Convert to percentage
                unit="Percent",
                dimensions={"MetricType": "Cache"},
            )

            # Cache hits (count)
            hits = cache_stats.get("hits", 0)
            await self.put_metric(
                "CacheHits", float(hits), unit="Count", dimensions={"MetricType": "Cache"}
            )

            # Cache misses (count)
            misses = cache_stats.get("misses", 0)
            await self.put_metric(
                "CacheMisses", float(misses), unit="Count", dimensions={"MetricType": "Cache"}
            )

        except Exception as e:
            logger.debug(f"Error exporting cache metrics: {e}")

    async def export_latency_metrics(self, percentiles: Dict[str, float]):
        """Export latency metrics to CloudWatch.

        Args:
            percentiles: Dictionary with P50, P95, P99 latencies
        """
        if not self.enabled:
            return

        try:
            # P50 latency
            if "p50" in percentiles:
                await self.put_metric(
                    "ApplicationLatency",
                    percentiles["p50"],
                    unit="Seconds",
                    dimensions={"Percentile": "P50"},
                )

            # P95 latency
            if "p95" in percentiles:
                await self.put_metric(
                    "ApplicationLatency",
                    percentiles["p95"],
                    unit="Seconds",
                    dimensions={"Percentile": "P95"},
                )

            # P99 latency
            if "p99" in percentiles:
                await self.put_metric(
                    "ApplicationLatency",
                    percentiles["p99"],
                    unit="Seconds",
                    dimensions={"Percentile": "P99"},
                )

            # Average latency
            if "mean" in percentiles or "average" in percentiles:
                avg = percentiles.get("mean") or percentiles.get("average", 0)
                await self.put_metric(
                    "ApplicationLatency", avg, unit="Seconds", dimensions={"Percentile": "Average"}
                )

        except Exception as e:
            logger.debug(f"Error exporting latency metrics: {e}")

    async def export_error_metrics(self, error_stats: Dict[str, Any]):
        """Export error metrics to CloudWatch.

        Args:
            error_stats: Error statistics dictionary
        """
        if not self.enabled:
            return

        try:
            # Total error rate
            total_errors = error_stats.get("total_errors", 0)
            error_rate = error_stats.get("error_rate", 0.0)

            await self.put_metric(
                "ErrorCount", float(total_errors), unit="Count", dimensions={"MetricType": "Errors"}
            )

            await self.put_metric(
                "ErrorRate", error_rate, unit="Count/Second", dimensions={"MetricType": "Errors"}
            )

            # Error rates by type
            error_types = error_stats.get("error_types", {})
            for error_type, count in error_types.items():
                await self.put_metric(
                    "ErrorCount",
                    float(count),
                    unit="Count",
                    dimensions={"MetricType": "Errors", "ErrorType": error_type},
                )

        except Exception as e:
            logger.debug(f"Error exporting error metrics: {e}")

    async def export_system_metrics(self, system_stats: Dict[str, Any]):
        """Export system metrics to CloudWatch.

        Args:
            system_stats: System statistics dictionary
        """
        if not self.enabled:
            return

        try:
            # CPU usage
            if "cpu_percent" in system_stats:
                await self.put_metric(
                    "SystemCPU",
                    system_stats["cpu_percent"],
                    unit="Percent",
                    dimensions={"MetricType": "System"},
                )

            # Memory usage
            if "memory_mb" in system_stats:
                await self.put_metric(
                    "SystemMemory",
                    system_stats["memory_mb"],
                    unit="Megabytes",
                    dimensions={"MetricType": "System"},
                )

        except Exception as e:
            logger.debug(f"Error exporting system metrics: {e}")

    async def export_evaluation_metrics(
        self, ir_metrics: Optional[Dict[str, float]] = None, deepeval_score: Optional[float] = None
    ):
        """Export evaluation metrics to CloudWatch.

        Args:
            ir_metrics: IR metrics dictionary (Precision, Recall, NDCG, etc.)
            deepeval_score: DeepEval overall score
        """
        if not self.enabled:
            return

        try:
            # IR Metrics
            if ir_metrics:
                for metric_name, value in ir_metrics.items():
                    if isinstance(value, (int, float)):
                        await self.put_metric(
                            f"IR{metric_name}",
                            float(value),
                            unit="None",
                            dimensions={"MetricType": "Evaluation", "EvaluationType": "IR"},
                        )

            # DeepEval score
            if deepeval_score is not None:
                await self.put_metric(
                    "DeepEvalScore",
                    float(deepeval_score),
                    unit="None",
                    dimensions={"MetricType": "Evaluation", "EvaluationType": "DeepEval"},
                )

        except Exception as e:
            logger.debug(f"Error exporting evaluation metrics: {e}")


# Global CloudWatch exporter instance
cloudwatch_exporter = CloudWatchExporter()
