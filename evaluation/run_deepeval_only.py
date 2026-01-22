"""Run only DeepEval LLM quality tests."""
import asyncio
import json
import sys
import io
from pathlib import Path
from evaluation.unified_evaluator import UnifiedEvaluator
from evaluation.report_generator import ReportGenerator
from src.analytics.logger import logger

# Fix Unicode encoding for Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


async def run_deepeval_evaluation(
    dataset_path: str = "evaluation/datasets/complex_queries.json",
    base_url: str = "http://localhost:3565",
    output_dir: str = "evaluation/reports"
):
    """Run DeepEval evaluation only.
    
    Args:
        dataset_path: Path to queries JSON file
        base_url: Base URL of API server
        output_dir: Directory to save reports
    """
    print(f"\n{'='*70}")
    print("Running DeepEval LLM Quality Evaluation")
    print(f"{'='*70}\n")
    
    # Load queries
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
        queries = [q.get("query", "") for q in dataset.get("queries", [])]
    
    print(f"Queries: {len(queries)}")
    print(f"Base URL: {base_url}\n")
    
    for i, query in enumerate(queries, 1):
        print(f"Query {i}/{len(queries)}: {query}")
    
    print("\n" + "="*70)
    print("Starting Evaluation...")
    print("="*70 + "\n")
    
    # Initialize evaluator
    evaluator = UnifiedEvaluator(base_url=base_url)
    
    # Run evaluation with only LLM quality metrics
    results = await evaluator.evaluate_batch(
        queries=queries,
        evaluation_types=["llm_quality"]  # Only DeepEval metrics
    )
    
    # Print results summary
    print("\n" + "="*70)
    print("Evaluation Results Summary")
    print("="*70 + "\n")
    
    if "aggregate_metrics" in results and "deepeval" in results["aggregate_metrics"]:
        deepeval = results["aggregate_metrics"]["deepeval"]
        print(f"Average Score: {deepeval.get('average_score', 0):.3f}")
        print(f"Pass Rate: {deepeval.get('pass_rate', 0):.1%}")
        print(f"Total Metrics: {deepeval.get('total_metrics', 0)}")
        print(f"Passed Metrics: {deepeval.get('passed_metrics', 0)}")
        
        # Print per-metric scores
        if "metric_scores" in deepeval:
            print("\nPer-Metric Scores:")
            print("-" * 70)
            for metric_name, score_data in deepeval["metric_scores"].items():
                score = score_data.get("score", 0)
                passed = score_data.get("passed", False)
                status = "[PASS]" if passed else "[FAIL]"
                print(f"  {metric_name:30s} {score:.3f}  {status}")
    
    # Print per-query results
    if "results" in results:
        print("\n" + "="*70)
        print("Per-Query Results")
        print("="*70 + "\n")
        
        for i, result in enumerate(results["results"], 1):
            query = result.get("query", "Unknown")
            deepeval = result.get("deepeval", {})
            
            if deepeval.get("enabled"):
                avg_score = deepeval.get("average_score", 0)
                passed = deepeval.get("passed", False)
                status = "[PASS]" if passed else "[FAIL]"
                
                print(f"Query {i}: {query[:60]}...")
                print(f"  Score: {avg_score:.3f}  {status}")
                
                # Show individual metric scores
                if "metrics" in deepeval:
                    for metric_name, metric_data in deepeval["metrics"].items():
                        if isinstance(metric_data, dict):
                            score = metric_data.get("score")
                            if score is None:
                                score = 0
                            passed = metric_data.get("passed", False)
                            print(f"    - {metric_name}: {score:.3f} {'[PASS]' if passed else '[FAIL]'}")
                print()
            else:
                print(f"Query {i}: {query[:60]}...")
                print(f"  DeepEval disabled or error: {deepeval.get('error', 'Unknown')}\n")
    
    # Generate reports
    print("="*70)
    print("Generating Reports...")
    print("="*70 + "\n")
    
    report_generator = ReportGenerator(output_dir=output_dir)
    
    # JSON report
    json_path = report_generator.generate_json_report(results)
    print(f"  [OK] JSON report: {json_path}")
    
    # CSV report
    if "results" in results and isinstance(results["results"], list):
        csv_path = report_generator.generate_csv_report(results["results"])
        print(f"  [OK] CSV report: {csv_path}")
    
    # HTML dashboard
    html_path = report_generator.generate_html_dashboard(results)
    print(f"  [OK] HTML dashboard: {html_path}")
    
    print("\n" + "="*70)
    print("Evaluation Complete!")
    print("="*70 + "\n")
    
    return results


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run DeepEval LLM quality evaluation only"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="evaluation/datasets/complex_queries.json",
        help="Path to queries JSON file (default: evaluation/datasets/complex_queries.json)"
    )
    
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:3565",
        help="Base URL of API server (default: http://localhost:3565)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation/reports",
        help="Output directory for reports (default: evaluation/reports)"
    )
    
    args = parser.parse_args()
    
    try:
        results = asyncio.run(run_deepeval_evaluation(
            dataset_path=args.dataset,
            base_url=args.base_url,
            output_dir=args.output
        ))
        
        # Exit with error code if evaluation failed
        if results and "aggregate_metrics" in results:
            deepeval = results["aggregate_metrics"].get("deepeval", {})
            if deepeval.get("pass_rate", 0) < 0.5:  # Less than 50% pass rate
                print("Warning: Low pass rate detected!")
                sys.exit(1)
        
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
