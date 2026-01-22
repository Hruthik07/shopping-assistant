"""Generate CloudWatch dashboard JSON for unified monitoring."""
import json
from typing import Dict, Any, Optional
from datetime import datetime

from src.utils.config import settings


class CloudWatchDashboardGenerator:
    """Generate CloudWatch dashboard JSON configuration."""
    
    def __init__(self):
        """Initialize dashboard generator."""
        self.namespace = getattr(settings, 'cloudwatch_namespace', 'ShoppingAssistant/Application')
        self.region = getattr(settings, 'aws_region', 'us-east-1')
        self.langfuse_url = getattr(settings, 'langfuse_host', 'https://cloud.langfuse.com')
        self.langfuse_project = getattr(settings, 'langfuse_project_name', 'shopping-assistant')
    
    def generate_dashboard(self) -> Dict[str, Any]:
        """Generate complete CloudWatch dashboard JSON.
        
        Returns:
            Dictionary with dashboard configuration
        """
        dashboard = {
            "widgets": []
        }
        
        # Row 1: Bedrock Metrics
        dashboard["widgets"].extend(self._bedrock_metrics_widgets())
        
        # Row 2: Application Metrics
        dashboard["widgets"].extend(self._application_metrics_widgets())
        
        # Row 3: Cache & Performance
        dashboard["widgets"].extend(self._cache_performance_widgets())
        
        # Row 4: Evaluation Metrics
        dashboard["widgets"].extend(self._evaluation_metrics_widgets())
        
        # Row 5: Cost & Errors
        dashboard["widgets"].extend(self._cost_error_widgets())
        
        # Row 6: Links & Status
        dashboard["widgets"].extend(self._links_status_widgets())
        
        return dashboard
    
    def _bedrock_metrics_widgets(self) -> list:
        """Generate Bedrock metrics widgets."""
        widgets = []
        
        # Bedrock Invocation Count
        widgets.append({
            "type": "metric",
            "properties": {
                "metrics": [
                    ["AWS/Bedrock", "Invocations", {"stat": "Sum"}]
                ],
                "period": 300,
                "stat": "Sum",
                "region": self.region,
                "title": "Bedrock Invocations",
                "view": "timeSeries",
                "yAxis": {
                    "left": {"label": "Count"}
                }
            }
        })
        
        # Bedrock Latency
        widgets.append({
            "type": "metric",
            "properties": {
                "metrics": [
                    ["AWS/Bedrock", "ModelLatency", {"stat": "Average", "label": "Average"}],
                    ["...", {"stat": "p50", "label": "P50"}],
                    ["...", {"stat": "p95", "label": "P95"}],
                    ["...", {"stat": "p99", "label": "P99"}]
                ],
                "period": 300,
                "stat": "Average",
                "region": self.region,
                "title": "Bedrock Latency",
                "view": "timeSeries",
                "yAxis": {
                    "left": {"label": "Seconds"}
                }
            }
        })
        
        # Bedrock Error Rate
        widgets.append({
            "type": "metric",
            "properties": {
                "metrics": [
                    ["AWS/Bedrock", "ModelInvocationErrors", {"stat": "Sum"}],
                    [".", "ModelInvocation4XXErrors", {"stat": "Sum"}],
                    [".", "ModelInvocation5XXErrors", {"stat": "Sum"}]
                ],
                "period": 300,
                "stat": "Sum",
                "region": self.region,
                "title": "Bedrock Errors",
                "view": "timeSeries",
                "yAxis": {
                    "left": {"label": "Count"}
                }
            }
        })
        
        return widgets
    
    def _application_metrics_widgets(self) -> list:
        """Generate application metrics widgets."""
        widgets = []
        
        # Application Latency
        widgets.append({
            "type": "metric",
            "properties": {
                "metrics": [
                    [self.namespace, "ApplicationLatency", {"stat": "Average", "dimensions": {"Percentile": "Average"}, "label": "Average"}],
                    ["...", {"stat": "Average", "dimensions": {"Percentile": "P50"}, "label": "P50"}],
                    ["...", {"stat": "Average", "dimensions": {"Percentile": "P95"}, "label": "P95"}],
                    ["...", {"stat": "Average", "dimensions": {"Percentile": "P99"}, "label": "P99"}]
                ],
                "period": 300,
                "stat": "Average",
                "region": self.region,
                "title": "Application Latency",
                "view": "timeSeries",
                "yAxis": {
                    "left": {"label": "Seconds"}
                }
            }
        })
        
        # Request Count
        widgets.append({
            "type": "metric",
            "properties": {
                "metrics": [
                    [self.namespace, "RequestCount", {"stat": "Sum"}]
                ],
                "period": 300,
                "stat": "Sum",
                "region": self.region,
                "title": "Request Count",
                "view": "timeSeries",
                "yAxis": {
                    "left": {"label": "Count"}
                }
            }
        })
        
        return widgets
    
    def _cache_performance_widgets(self) -> list:
        """Generate cache and performance widgets."""
        widgets = []
        
        # Cache Hit Rate
        widgets.append({
            "type": "metric",
            "properties": {
                "metrics": [
                    [self.namespace, "CacheHitRate", {"stat": "Average", "dimensions": {"MetricType": "Cache"}}]
                ],
                "period": 300,
                "stat": "Average",
                "region": self.region,
                "title": "Cache Hit Rate",
                "view": "timeSeries",
                "yAxis": {
                    "left": {"label": "Percent", "min": 0, "max": 100}
                }
            }
        })
        
        # Cache Hits/Misses
        widgets.append({
            "type": "metric",
            "properties": {
                "metrics": [
                    [self.namespace, "CacheHits", {"stat": "Sum", "dimensions": {"MetricType": "Cache"}, "label": "Hits"}],
                    [".", "CacheMisses", {"stat": "Sum", "dimensions": {"MetricType": "Cache"}, "label": "Misses"}]
                ],
                "period": 300,
                "stat": "Sum",
                "region": self.region,
                "title": "Cache Hits vs Misses",
                "view": "timeSeries",
                "yAxis": {
                    "left": {"label": "Count"}
                }
            }
        })
        
        return widgets
    
    def _evaluation_metrics_widgets(self) -> list:
        """Generate evaluation metrics widgets."""
        widgets = []
        
        # IR Metrics Summary (Number widgets)
        widgets.append({
            "type": "metric",
            "properties": {
                "metrics": [
                    [self.namespace, "IRPrecision@5", {"stat": "Average", "dimensions": {"MetricType": "Evaluation", "EvaluationType": "IR"}}]
                ],
                "period": 3600,
                "stat": "Average",
                "region": self.region,
                "title": "Precision@5",
                "view": "singleValue"
            }
        })
        
        widgets.append({
            "type": "metric",
            "properties": {
                "metrics": [
                    [self.namespace, "IRNDCG@10", {"stat": "Average", "dimensions": {"MetricType": "Evaluation", "EvaluationType": "IR"}}]
                ],
                "period": 3600,
                "stat": "Average",
                "region": self.region,
                "title": "NDCG@10",
                "view": "singleValue"
            }
        })
        
        widgets.append({
            "type": "metric",
            "properties": {
                "metrics": [
                    [self.namespace, "DeepEvalOverallScore", {"stat": "Average", "dimensions": {"MetricType": "Evaluation", "EvaluationType": "DeepEval"}}]
                ],
                "period": 3600,
                "stat": "Average",
                "region": self.region,
                "title": "DeepEval Score",
                "view": "singleValue"
            }
        })
        
        return widgets
    
    def _cost_error_widgets(self) -> list:
        """Generate cost and error widgets."""
        widgets = []
        
        # Error Rate
        widgets.append({
            "type": "metric",
            "properties": {
                "metrics": [
                    [self.namespace, "ErrorCount", {"stat": "Sum", "dimensions": {"MetricType": "Errors"}}]
                ],
                "period": 300,
                "stat": "Sum",
                "region": self.region,
                "title": "Error Count",
                "view": "timeSeries",
                "yAxis": {
                    "left": {"label": "Count"}
                }
            }
        })
        
        # Cost (if Cost Explorer metrics available)
        widgets.append({
            "type": "text",
            "properties": {
                "markdown": "## Cost Metrics\n\nCost tracking via Cost Explorer API.\n\nView in AWS Cost Explorer: [Open Cost Explorer](https://console.aws.amazon.com/cost-management/home)"
            }
        })
        
        return widgets
    
    def _links_status_widgets(self) -> list:
        """Generate links and status widgets."""
        widgets = []
        
        # Links to external tools
        langfuse_link = f"{self.langfuse_url}/project/{self.langfuse_project}/traces"
        widgets.append({
            "type": "text",
            "properties": {
                "markdown": f"## Monitoring Links\n\n- [Langfuse Traces]({langfuse_link})\n- [Confident AI](https://app.confident-ai.com)\n- [CloudWatch Metrics](https://{self.region}.console.aws.amazon.com/cloudwatch/home?region={self.region}#metricsV2:graph=~();namespace={self.namespace})"
            }
        })
        
        return widgets
    
    def save_dashboard_json(self, filepath: str = "monitoring/cloudwatch_dashboard.json"):
        """Save dashboard JSON to file.
        
        Args:
            filepath: Path to save dashboard JSON
        """
        dashboard = self.generate_dashboard()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(dashboard, f, indent=2)
        
        return filepath


# Global dashboard generator instance
dashboard_generator = CloudWatchDashboardGenerator()
