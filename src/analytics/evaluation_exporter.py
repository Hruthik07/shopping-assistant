"""Export evaluation metrics to CloudWatch."""

from typing import Dict, Any, Optional
import time

from src.analytics.logger import logger
from src.analytics.cloudwatch_exporter import cloudwatch_exporter


class EvaluationExporter:
    """Export evaluation results (IR metrics, DeepEval scores) to CloudWatch."""

    def __init__(self):
        """Initialize evaluation exporter."""
        self.enabled = cloudwatch_exporter.enabled

    async def export_ir_metrics(
        self,
        ir_metrics: Dict[str, float],
        query: Optional[str] = None,
        trace_id: Optional[str] = None,
    ):
        """Export IR metrics to CloudWatch.

        Args:
            ir_metrics: Dictionary with IR metric scores (e.g., {"Precision@5": 0.85, "Recall@10": 0.92})
            query: Optional query text for context
            trace_id: Optional Langfuse trace ID for linking
        """
        if not self.enabled:
            return

        try:
            dimensions = {"MetricType": "Evaluation", "EvaluationType": "IR"}

            if trace_id:
                dimensions["TraceID"] = trace_id[:50]  # Truncate for CloudWatch

            # Export each IR metric
            for metric_name, value in ir_metrics.items():
                if isinstance(value, (int, float)):
                    await cloudwatch_exporter.put_metric(
                        metric_name=f"IR{metric_name}",
                        value=float(value),
                        unit="None",
                        dimensions=dimensions,
                    )

            logger.debug(f"Exported {len(ir_metrics)} IR metrics to CloudWatch")

        except Exception as e:
            logger.debug(f"Error exporting IR metrics to CloudWatch: {e}")

    async def export_deepeval_scores(
        self, deepeval_results: Dict[str, Any], trace_id: Optional[str] = None
    ):
        """Export DeepEval scores to CloudWatch.

        Args:
            deepeval_results: Dictionary with DeepEval results
                - "overall_score": float (average score)
                - "metrics": Dict[str, Dict] with individual metric scores
            trace_id: Optional Langfuse trace ID for linking
        """
        if not self.enabled:
            return

        try:
            dimensions = {"MetricType": "Evaluation", "EvaluationType": "DeepEval"}

            if trace_id:
                dimensions["TraceID"] = trace_id[:50]

            # Export overall score
            overall_score = deepeval_results.get("overall_score")
            if overall_score is not None:
                await cloudwatch_exporter.put_metric(
                    metric_name="DeepEvalOverallScore",
                    value=float(overall_score),
                    unit="None",
                    dimensions=dimensions,
                )

            # Export individual metric scores (summary)
            metrics = deepeval_results.get("metrics", {})
            if metrics:
                # Calculate pass rate
                passed_count = sum(
                    1 for m in metrics.values() if isinstance(m, dict) and m.get("passed", False)
                )
                pass_rate = passed_count / len(metrics) if metrics else 0.0

                await cloudwatch_exporter.put_metric(
                    metric_name="DeepEvalPassRate",
                    value=pass_rate * 100,  # Convert to percentage
                    unit="Percent",
                    dimensions=dimensions,
                )

                # Export average score across all metrics
                scores = [
                    m.get("score", 0.0)
                    for m in metrics.values()
                    if isinstance(m, dict) and "score" in m
                ]
                if scores:
                    avg_score = sum(scores) / len(scores)
                    await cloudwatch_exporter.put_metric(
                        metric_name="DeepEvalAverageScore",
                        value=float(avg_score),
                        unit="None",
                        dimensions=dimensions,
                    )

            logger.debug("Exported DeepEval scores to CloudWatch")

        except Exception as e:
            logger.debug(f"Error exporting DeepEval scores to CloudWatch: {e}")

    async def export_evaluation_latency(self, latency: float, evaluation_type: str = "complete"):
        """Export evaluation latency to CloudWatch.

        Args:
            latency: Evaluation latency in seconds
            evaluation_type: Type of evaluation (e.g., "complete", "ir", "deepeval")
        """
        if not self.enabled:
            return

        try:
            await cloudwatch_exporter.put_metric(
                metric_name="EvaluationLatency",
                value=float(latency),
                unit="Seconds",
                dimensions={"MetricType": "Evaluation", "EvaluationType": evaluation_type},
            )
        except Exception as e:
            logger.debug(f"Error exporting evaluation latency: {e}")

    async def export_batch_evaluation_results(
        self, results: Dict[str, Any], trace_ids: Optional[Dict[str, str]] = None
    ):
        """Export batch evaluation results to CloudWatch.

        Args:
            results: Dictionary with batch evaluation results
                - "ir_metrics": Dict with aggregated IR metrics
                - "deepeval": Dict with aggregated DeepEval scores
                - "latency": float (total evaluation latency)
            trace_ids: Optional mapping of query -> trace_id
        """
        if not self.enabled:
            return

        try:
            # Export aggregated IR metrics
            ir_metrics = results.get("ir_metrics", {})
            if ir_metrics:
                await self.export_ir_metrics(ir_metrics)

            # Export aggregated DeepEval scores
            deepeval = results.get("deepeval", {})
            if deepeval:
                await self.export_deepeval_scores(deepeval)

            # Export evaluation latency
            latency = results.get("latency")
            if latency:
                await self.export_evaluation_latency(latency, "complete")

            logger.debug("Exported batch evaluation results to CloudWatch")

        except Exception as e:
            logger.debug(f"Error exporting batch evaluation results: {e}")


# Global evaluation exporter instance
evaluation_exporter = EvaluationExporter()
