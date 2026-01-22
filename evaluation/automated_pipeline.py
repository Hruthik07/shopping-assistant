"""Automated evaluation pipeline for continuous quality monitoring."""
import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import httpx

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from evaluation.run_complete_evaluation import CompleteEvaluator
from evaluation.unified_evaluator import UnifiedEvaluator
from evaluation.ir_metrics import IRMetrics
from src.analytics.logger import logger
from src.analytics.cost_tracker import cost_tracker


class AutomatedEvaluationPipeline:
    """Automated evaluation pipeline with baseline comparison and alerting."""
    
    def __init__(
        self,
        base_url: str = "http://localhost:3565",
        baseline_file: Optional[str] = None
    ):
        """Initialize automated evaluation pipeline.
        
        Args:
            base_url: API server base URL
            baseline_file: Path to baseline evaluation results
        """
        self.base_url = base_url
        self.baseline_file = baseline_file or "evaluation/baseline_metrics.json"
        self.results_dir = Path("evaluation/automated_results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.baseline_metrics: Optional[Dict[str, Any]] = None
        self.load_baseline()
    
    def load_baseline(self):
        """Load baseline metrics for comparison."""
        try:
            baseline_path = Path(self.baseline_file)
            if baseline_path.exists():
                with open(baseline_path, 'r', encoding='utf-8') as f:
                    self.baseline_metrics = json.load(f)
                logger.info(f"Loaded baseline metrics from {self.baseline_file}")
            else:
                logger.warning(f"Baseline file not found: {self.baseline_file}")
        except Exception as e:
            logger.error(f"Error loading baseline: {e}")
    
    def save_baseline(self, metrics: Dict[str, Any]):
        """Save current metrics as new baseline.
        
        Args:
            metrics: Current evaluation metrics
        """
        try:
            baseline_path = Path(self.baseline_file)
            baseline_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(baseline_path, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2)
            
            self.baseline_metrics = metrics
            logger.info(f"Saved new baseline to {self.baseline_file}")
        except Exception as e:
            logger.error(f"Error saving baseline: {e}")
    
    async def run_evaluation(
        self,
        queries: List[str],
        evaluation_types: List[str] = ["latency", "ir_metrics"]
    ) -> Dict[str, Any]:
        """Run evaluation and return results.
        
        Args:
            queries: List of test queries
            evaluation_types: Types of evaluation to run
            
        Returns:
            Dictionary with evaluation results
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results = {
            "timestamp": timestamp,
            "evaluation_types": evaluation_types,
            "queries": queries,
            "results": {}
        }
        
        # Run complete evaluation (latency + IR metrics)
        if "latency" in evaluation_types or "ir_metrics" in evaluation_types:
            try:
                evaluator = CompleteEvaluator(base_url=self.base_url)
                start_time = asyncio.get_event_loop().time()
                await evaluator.run_complete_evaluation(queries)
                evaluation_latency = asyncio.get_event_loop().time() - start_time
                
                # Export evaluation latency to CloudWatch
                try:
                    from src.analytics.evaluation_exporter import evaluation_exporter
                    await evaluation_exporter.export_evaluation_latency(evaluation_latency, "complete")
                except Exception:
                    pass  # Graceful degradation
                
                # Extract metrics from evaluator
                results["results"]["latency"] = {
                    "average": evaluator.query_results[0].get("response_time", 0) if evaluator.query_results else 0,
                    "queries": len(evaluator.query_results),
                    "successful": sum(1 for q in evaluator.query_results if q.get("success", False))
                }
                
                # IR metrics would be in the evaluator's results
                results["results"]["ir_metrics"] = {
                    "note": "IR metrics calculated in complete evaluation"
                }
            except Exception as e:
                logger.error(f"Error running complete evaluation: {e}")
                results["results"]["latency"] = {"error": str(e)}
        
        # Run LLM quality evaluation (if enabled and API key available)
        if "llm_quality" in evaluation_types:
            try:
                unified_evaluator = UnifiedEvaluator(base_url=self.base_url)
                llm_results = await unified_evaluator.evaluate_batch(
                    queries=queries,
                    evaluation_types=["llm_quality"]
                )
                results["results"]["llm_quality"] = llm_results
            except Exception as e:
                logger.warning(f"LLM quality evaluation failed (may need API key): {e}")
                results["results"]["llm_quality"] = {"error": str(e)}
        
        # Add cost metrics
        cost_stats = cost_tracker.get_cost_stats(days=1)
        results["results"]["cost"] = {
            "daily_cost": cost_stats.get("total_cost", 0),
            "requests": cost_stats.get("request_count", 0),
            "average_cost_per_request": cost_stats.get("average_cost_per_request", 0)
        }
        
        return results
    
    def compare_with_baseline(self, current_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare current results with baseline and detect regressions.
        
        Args:
            current_results: Current evaluation results
            
        Returns:
            Dictionary with comparison and alerts
        """
        if not self.baseline_metrics:
            return {
                "baseline_available": False,
                "message": "No baseline available for comparison"
            }
        
        comparison = {
            "baseline_available": True,
            "regressions": [],
            "improvements": [],
            "stable": []
        }
        
        # Compare latency
        current_latency = current_results.get("results", {}).get("latency", {}).get("average", 0)
        baseline_latency = self.baseline_metrics.get("results", {}).get("latency", {}).get("average", 0)
        
        if baseline_latency > 0:
            latency_change = ((current_latency - baseline_latency) / baseline_latency) * 100
            
            if latency_change > 20:  # 20% increase is regression
                comparison["regressions"].append({
                    "metric": "latency",
                    "baseline": baseline_latency,
                    "current": current_latency,
                    "change_percent": round(latency_change, 2),
                    "severity": "high" if latency_change > 50 else "medium"
                })
            elif latency_change < -10:  # 10% decrease is improvement
                comparison["improvements"].append({
                    "metric": "latency",
                    "baseline": baseline_latency,
                    "current": current_latency,
                    "change_percent": round(latency_change, 2)
                })
            else:
                comparison["stable"].append({
                    "metric": "latency",
                    "value": current_latency,
                    "change_percent": round(latency_change, 2)
                })
        
        # Compare cost
        current_cost = current_results.get("results", {}).get("cost", {}).get("daily_cost", 0)
        baseline_cost = self.baseline_metrics.get("results", {}).get("cost", {}).get("daily_cost", 0)
        
        if baseline_cost > 0:
            cost_change = ((current_cost - baseline_cost) / baseline_cost) * 100
            
            if cost_change > 30:  # 30% increase is regression
                comparison["regressions"].append({
                    "metric": "cost",
                    "baseline": baseline_cost,
                    "current": current_cost,
                    "change_percent": round(cost_change, 2),
                    "severity": "high" if cost_change > 50 else "medium"
                })
            elif cost_change < -10:  # 10% decrease is improvement
                comparison["improvements"].append({
                    "metric": "cost",
                    "baseline": baseline_cost,
                    "current": current_cost,
                    "change_percent": round(cost_change, 2)
                })
        
        return comparison
    
    def generate_alert(self, comparison: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate alert if regressions detected.
        
        Args:
            comparison: Comparison results from compare_with_baseline
            
        Returns:
            Alert dictionary if regressions found, None otherwise
        """
        regressions = comparison.get("regressions", [])
        
        if not regressions:
            return None
        
        high_severity = [r for r in regressions if r.get("severity") == "high"]
        medium_severity = [r for r in regressions if r.get("severity") == "medium"]
        
        return {
            "alert": True,
            "timestamp": datetime.now().isoformat(),
            "high_severity_count": len(high_severity),
            "medium_severity_count": len(medium_severity),
            "regressions": regressions,
            "message": f"ALERT: {len(regressions)} metric regression(s) detected"
        }
    
    async def run_automated_evaluation(
        self,
        queries: Optional[List[str]] = None,
        save_results: bool = True,
        update_baseline: bool = False
    ) -> Dict[str, Any]:
        """Run automated evaluation with baseline comparison.
        
        Args:
            queries: Optional list of queries (uses default if not provided)
            save_results: Whether to save results to file
            update_baseline: Whether to update baseline with current results
            
        Returns:
            Complete evaluation report with comparison
        """
        if queries is None:
            # Default test queries
            queries = [
                "Find me wireless headphones under $100",
                "What are the best laptops for gaming?",
                "Show me running shoes between $50 and $90"
            ]
        
        logger.info(f"Starting automated evaluation with {len(queries)} queries")
        
        # Run evaluation
        results = await self.run_evaluation(
            queries=queries,
            evaluation_types=["latency", "ir_metrics"]
        )
        
        # Compare with baseline
        comparison = self.compare_with_baseline(results)
        results["comparison"] = comparison
        
        # Generate alert if needed
        alert = self.generate_alert(comparison)
        if alert:
            results["alert"] = alert
            logger.warning(f"ALERT: {alert['message']}")
        
        # Save results
        if save_results:
            timestamp = results["timestamp"]
            results_file = self.results_dir / f"evaluation_{timestamp}.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {results_file}")
            results["results_file"] = str(results_file)
        
        # Update baseline if requested
        if update_baseline:
            self.save_baseline(results)
        
        return results
    
    def generate_trend_report(self, days: int = 7) -> Dict[str, Any]:
        """Generate trend report from historical evaluations.
        
        Args:
            days: Number of days to look back
            
        Returns:
            Trend report dictionary
        """
        cutoff_date = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
        
        # Find all evaluation files
        evaluation_files = sorted(
            self.results_dir.glob("evaluation_*.json"),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        trends = {
            "period_days": days,
            "evaluations": [],
            "latency_trend": [],
            "cost_trend": []
        }
        
        for eval_file in evaluation_files:
            try:
                # Check if file is within date range
                file_date = eval_file.stem.split("_")[1] if "_" in eval_file.stem else ""
                if file_date < cutoff_date:
                    continue
                
                with open(eval_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                trends["evaluations"].append({
                    "timestamp": data.get("timestamp"),
                    "latency": data.get("results", {}).get("latency", {}).get("average", 0),
                    "cost": data.get("results", {}).get("cost", {}).get("daily_cost", 0)
                })
                
                trends["latency_trend"].append(data.get("results", {}).get("latency", {}).get("average", 0))
                trends["cost_trend"].append(data.get("results", {}).get("cost", {}).get("daily_cost", 0))
                
            except Exception as e:
                logger.debug(f"Error reading evaluation file {eval_file}: {e}")
        
        # Calculate trend statistics
        if trends["latency_trend"]:
            trends["latency_stats"] = {
                "min": min(trends["latency_trend"]),
                "max": max(trends["latency_trend"]),
                "average": sum(trends["latency_trend"]) / len(trends["latency_trend"]),
                "trend": "improving" if trends["latency_trend"][0] < trends["latency_trend"][-1] else "degrading"
            }
        
        if trends["cost_trend"]:
            trends["cost_stats"] = {
                "min": min(trends["cost_trend"]),
                "max": max(trends["cost_trend"]),
                "average": sum(trends["cost_trend"]) / len(trends["cost_trend"]),
                "trend": "increasing" if trends["cost_trend"][0] < trends["cost_trend"][-1] else "decreasing"
            }
        
        return trends


async def main():
    """Run automated evaluation (can be called from scheduler)."""
    pipeline = AutomatedEvaluationPipeline()
    
    results = await pipeline.run_automated_evaluation(
        save_results=True,
        update_baseline=False  # Set to True to update baseline
    )
    
    print("\n" + "="*70)
    print("AUTOMATED EVALUATION RESULTS")
    print("="*70)
    print(f"Timestamp: {results['timestamp']}")
    print(f"Queries: {len(results['queries'])}")
    
    if "alert" in results:
        print(f"\n⚠️  ALERT: {results['alert']['message']}")
        for regression in results['alert']['regressions']:
            print(f"  - {regression['metric']}: {regression['change_percent']:.1f}% change")
    else:
        print("\n✅ No regressions detected")
    
    if "comparison" in results:
        comp = results["comparison"]
        if comp.get("improvements"):
            print("\n📈 Improvements:")
            for improvement in comp["improvements"]:
                print(f"  - {improvement['metric']}: {improvement['change_percent']:.1f}% improvement")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    asyncio.run(main())
