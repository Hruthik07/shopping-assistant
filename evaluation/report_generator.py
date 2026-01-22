"""Generate comprehensive evaluation reports."""
import json
import csv
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from evaluation.unified_evaluator import UnifiedEvaluator
from src.analytics.logger import logger


class ReportGenerator:
    """Generate evaluation reports in multiple formats."""
    
    def __init__(self, output_dir: str = "evaluation/reports"):
        """Initialize report generator.
        
        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_json_report(
        self,
        results: Dict[str, Any],
        filename: Optional[str] = None
    ) -> str:
        """Generate JSON report.
        
        Args:
            results: Evaluation results dictionary
            filename: Optional filename (default: timestamp-based)
        
        Returns:
            Path to generated report
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_report_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"JSON report saved to: {filepath}")
        return str(filepath)
    
    def generate_csv_report(
        self,
        results: List[Dict[str, Any]],
        filename: Optional[str] = None
    ) -> str:
        """Generate CSV report from results list.
        
        Args:
            results: List of evaluation result dictionaries
            filename: Optional filename (default: timestamp-based)
        
        Returns:
            Path to generated report
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_report_{timestamp}.csv"
        
        filepath = self.output_dir / filename
        
        if not results:
            logger.warning("No results to export to CSV")
            return str(filepath)
        
        # Detect MCP result rows (different shape than UnifiedEvaluator results)
        is_mcp_rows = bool(results) and isinstance(results[0], dict) and (
            "avg_latency" in results[0] and "avg_tool_selection_score" in results[0] and "runs" in results[0]
        )

        rows = []
        for result in results:
            if is_mcp_rows:
                row = {
                    "id": result.get("id", ""),
                    "query": result.get("query", ""),
                    "iterations": result.get("iterations", ""),
                    "passed": result.get("passed", False),
                    "pass_rate": result.get("pass_rate", 0.0),
                    "avg_latency": result.get("avg_latency", 0.0),
                    "avg_tool_selection_score": result.get("avg_tool_selection_score", 0.0),
                    "avg_price_compliance_rate": result.get("avg_price_compliance_rate", 0.0),
                    "avg_url_valid_rate": result.get("avg_url_valid_rate", 0.0),
                    "faithfulness_failures": result.get("faithfulness_failures", 0),
                }
                rows.append(row)
                continue

            # Default: UnifiedEvaluator row format
            row = {
                "query": result.get("query", ""),
                "session_id": result.get("session_id", ""),
                "timestamp": result.get("timestamp", ""),
                "success": result.get("success", False)
            }
            
            # Add DeepEval metrics
            deepeval = result.get("deepeval", {})
            if deepeval.get("enabled"):
                row["deepeval_overall_score"] = deepeval.get("overall_score", 0)
                row["deepeval_all_passed"] = deepeval.get("all_passed", False)
                metrics = deepeval.get("metrics", {})
                for metric_name, metric_data in metrics.items():
                    if isinstance(metric_data, dict):
                        row[f"deepeval_{metric_name}_score"] = metric_data.get("score", 0) if metric_data.get("score") is not None else "N/A"
                        row[f"deepeval_{metric_name}_passed"] = metric_data.get("passed", False) if metric_data.get("passed") is not None else "N/A"
                        row[f"deepeval_{metric_name}_status"] = metric_data.get("status", "unknown")
            
            # Add IR metrics
            ir_metrics = result.get("ir_metrics", {})
            if ir_metrics:
                row["ir_metrics_status"] = ir_metrics.get("status", "unknown")
                if ir_metrics.get("status") == "success":
                    for k in [1, 3, 5, 10]:
                        row[f"ir_precision@{k}"] = ir_metrics.get(f"precision@{k}", "N/A")
                        row[f"ir_recall@{k}"] = ir_metrics.get(f"recall@{k}", "N/A")
                        row[f"ir_ndcg@{k}"] = ir_metrics.get(f"ndcg@{k}", "N/A")
                    row["ir_mrr"] = ir_metrics.get("mrr", "N/A")
                    row["ir_map"] = ir_metrics.get("map", "N/A")
                    row["ir_context_precision"] = ir_metrics.get("context_precision", "N/A")
                else:
                    row["ir_metrics_note"] = ir_metrics.get("note", "")
            
            # Add performance metrics
            performance = result.get("performance", {})
            if performance:
                row["total_time"] = performance.get("total_time", 0)
                row["ttft"] = performance.get("ttft", 0)
                row["products_count"] = performance.get("products_count", 0)
            
            rows.append(row)
        
        # Write CSV
        if rows:
            fieldnames = rows[0].keys()
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        
        logger.info(f"CSV report saved to: {filepath}")
        return str(filepath)
    
    def generate_html_dashboard(
        self,
        results: Dict[str, Any],
        filename: Optional[str] = None
    ) -> str:
        """Generate HTML dashboard report.
        
        Args:
            results: Evaluation results dictionary
            filename: Optional filename (default: timestamp-based)
        
        Returns:
            Path to generated report
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_dashboard_{timestamp}.html"
        
        filepath = self.output_dir / filename
        
        # Generate HTML
        html = self._generate_html_content(results)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)
        
        logger.info(f"HTML dashboard saved to: {filepath}")
        return str(filepath)
    
    def _generate_html_content(self, results: Dict[str, Any]) -> str:
        """Generate HTML content for dashboard."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Evaluation Report - {timestamp}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 2px solid #007bff;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
        }}
        .metric-card {{
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 4px;
            padding: 15px;
            margin: 10px 0;
        }}
        .metric-name {{
            font-weight: bold;
            color: #007bff;
        }}
        .score {{
            font-size: 24px;
            font-weight: bold;
            color: #28a745;
        }}
        .score.low {{
            color: #dc3545;
        }}
        .score.medium {{
            color: #ffc107;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #dee2e6;
        }}
        th {{
            background: #007bff;
            color: white;
        }}
        tr:hover {{
            background: #f8f9fa;
        }}
        .passed {{
            color: #28a745;
            font-weight: bold;
        }}
        .failed {{
            color: #dc3545;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Evaluation Report</h1>
        <p><strong>Generated:</strong> {timestamp}</p>
        
        <h2>Summary</h2>
        <div class="metric-card">
            <div class="metric-name">Total Queries</div>
            <div class="score">{results.get('total_queries', 0)}</div>
        </div>
        <div class="metric-card">
            <div class="metric-name">Successful</div>
            <div class="score">{results.get('successful', 0)}</div>
        </div>
        <div class="metric-card">
            <div class="metric-name">Failed</div>
            <div class="score low">{results.get('failed', 0)}</div>
        </div>
"""
        
        # Add aggregate metrics
        aggregates = results.get("aggregate_metrics", {})
        if aggregates:
            html += "<h2>Aggregate Metrics</h2>"
            
            if "deepeval" in aggregates:
                deepeval = aggregates["deepeval"]
                html += f"""
        <div class="metric-card">
            <div class="metric-name">DeepEval Average Score</div>
            <div class="score">{deepeval.get('average_score', 0):.3f}</div>
            <div>Pass Rate: {deepeval.get('pass_rate', 0):.1%}</div>
        </div>
"""
            
            if "ir_metrics" in aggregates:
                ir = aggregates["ir_metrics"]
                html += f"""
        <div class="metric-card">
            <div class="metric-name">IR Metrics Average</div>
            <div>Precision@5: {ir.get('avg_precision@5', 0):.3f}</div>
            <div>Recall@5: {ir.get('avg_recall@5', 0):.3f}</div>
            <div>NDCG@5: {ir.get('avg_ndcg@5', 0):.3f}</div>
        </div>
"""
            
            if "performance" in aggregates:
                perf = aggregates["performance"]
                html += f"""
        <div class="metric-card">
            <div class="metric-name">Average Total Time</div>
            <div class="score">{perf.get('average_total_time', 0):.2f}s</div>
            <div>Average TTFT: {perf.get('average_ttft', 0):.2f}s</div>
        </div>
"""
            
            if "mcp" in aggregates:
                mcp = aggregates["mcp"]
                html += f"""
        <div class="metric-card">
            <div class="metric-name">MCP Pass Rate</div>
            <div class="score">{mcp.get('pass_rate', 0):.3f}</div>
            <div>Avg Latency: {mcp.get('avg_latency', 0):.3f}s</div>
            <div>Avg Tool Selection Score: {mcp.get('avg_tool_selection_score', 0):.3f}</div>
            <div>Avg Price Compliance: {mcp.get('avg_price_compliance_rate', 0):.3f}</div>
            <div>Avg URL Valid Rate: {mcp.get('avg_url_valid_rate', 0):.3f}</div>
        </div>
"""
        
        # Add detailed results table
        detailed_results = results.get("results", [])
        if detailed_results:
            # Special-case MCP results (different schema)
            if results.get("evaluation_types") == ["mcp"]:
                html += "<h2>Detailed Results</h2><table><thead><tr><th>ID</th><th>Query</th><th>Passed</th><th>Pass Rate</th><th>Avg Latency</th><th>Tool Score</th><th>Price Compliance</th><th>URL Valid</th><th>Faithfulness Failures</th></tr></thead><tbody>"
                for r in detailed_results[:50]:
                    q = (r.get("query", "") or "")[:70]
                    passed = bool(r.get("passed"))
                    html += f"""
                    <tr>
                        <td>{r.get('id','')}</td>
                        <td>{q}</td>
                        <td class="{'passed' if passed else 'failed'}">{'✓' if passed else '✗'}</td>
                        <td>{r.get('pass_rate', 0):.3f}</td>
                        <td>{r.get('avg_latency', 0):.3f}s</td>
                        <td>{r.get('avg_tool_selection_score', 0):.3f}</td>
                        <td>{r.get('avg_price_compliance_rate', 0):.3f}</td>
                        <td>{r.get('avg_url_valid_rate', 0):.3f}</td>
                        <td>{r.get('faithfulness_failures', 0)}</td>
                    </tr>
"""
                html += "</tbody></table>"
            else:
                html += "<h2>Detailed Results</h2><table><thead><tr><th>Query</th><th>Success</th><th>DeepEval Score</th><th>IR Status</th><th>Total Time</th><th>TTFT</th></tr></thead><tbody>"
                
                for result in detailed_results[:50]:  # Limit to 50 rows
                    query = result.get("query", "")[:50]
                    success = "✓" if result.get("success") else "✗"
                    deepeval_score = result.get("deepeval", {}).get("overall_score", 0)
                    ir_status = result.get("ir_metrics", {}).get("status", "N/A")
                    total_time = result.get("performance", {}).get("total_time", 0)
                    ttft = result.get("performance", {}).get("ttft", 0)
                    
                    html += f"""
                    <tr>
                        <td>{query}</td>
                        <td class="{'passed' if result.get('success') else 'failed'}">{success}</td>
                        <td>{deepeval_score:.3f}</td>
                        <td>{ir_status}</td>
                        <td>{total_time:.2f}s</td>
                        <td>{ttft:.2f}s</td>
                    </tr>
"""
                
                html += "</tbody></table>"
        
        html += """
    </div>
</body>
</html>
"""
        return html
    
    def generate_comparison_report(
        self,
        baseline_results: Dict[str, Any],
        current_results: Dict[str, Any],
        filename: Optional[str] = None
    ) -> str:
        """Generate comparison report between baseline and current results.
        
        Args:
            baseline_results: Baseline evaluation results
            current_results: Current evaluation results
            filename: Optional filename
        
        Returns:
            Path to generated report
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"comparison_report_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        comparison = {
            "baseline": baseline_results,
            "current": current_results,
            "comparison": self._compare_results(baseline_results, current_results),
            "timestamp": datetime.now().isoformat()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Comparison report saved to: {filepath}")
        return str(filepath)
    
    def _compare_results(
        self,
        baseline: Dict[str, Any],
        current: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare baseline and current results."""
        comparison = {}
        
        # Compare DeepEval scores
        baseline_deepeval = baseline.get("aggregate_metrics", {}).get("deepeval", {})
        current_deepeval = current.get("aggregate_metrics", {}).get("deepeval", {})
        
        if baseline_deepeval and current_deepeval:
            baseline_score = baseline_deepeval.get("average_score", 0)
            current_score = current_deepeval.get("average_score", 0)
            comparison["deepeval"] = {
                "baseline": baseline_score,
                "current": current_score,
                "change": current_score - baseline_score,
                "change_percent": ((current_score - baseline_score) / baseline_score * 100) if baseline_score > 0 else 0
            }
        
        # Compare performance
        baseline_perf = baseline.get("aggregate_metrics", {}).get("performance", {})
        current_perf = current.get("aggregate_metrics", {}).get("performance", {})
        
        if baseline_perf and current_perf:
            baseline_time = baseline_perf.get("average_total_time", 0)
            current_time = current_perf.get("average_total_time", 0)
            comparison["performance"] = {
                "baseline": baseline_time,
                "current": current_time,
                "change": current_time - baseline_time,
                "change_percent": ((current_time - baseline_time) / baseline_time * 100) if baseline_time > 0 else 0
            }
        
        return comparison

