"""Command-line interface for running evaluations."""
import asyncio
import argparse
import json
import sys
from pathlib import Path
from typing import List
from evaluation.unified_evaluator import UnifiedEvaluator
from evaluation.report_generator import ReportGenerator
from evaluation.test_suites.llm_quality_tests import LLMQualityTestSuite
from evaluation.test_suites.retrieval_tests import RetrievalTestSuite
from evaluation.test_suites.performance_tests import PerformanceTestSuite
from evaluation.test_suites.system_tests import SystemTestSuite
from evaluation.test_suites.mcp_tests import MCPTestSuite
from src.analytics.logger import logger


async def run_evaluation(
    evaluation_type: str,
    queries: List[str] = None,
    dataset_path: str = None,
    output_dir: str = "evaluation/reports",
    base_url: str = "http://localhost:3565",
    iterations: int = 1
):
    """Run evaluation based on type.
    
    Args:
        evaluation_type: Type of evaluation (llm, retrieval, performance, system, mcp, all)
        queries: List of queries to evaluate
        dataset_path: Path to dataset JSON file
        output_dir: Directory to save reports
        base_url: Base URL of API server
        iterations: Number of iterations for performance tests
    """
    evaluator = UnifiedEvaluator(base_url=base_url)
    report_generator = ReportGenerator(output_dir=output_dir)
    
    # Load queries if dataset provided
    if dataset_path:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
            if queries is None:
                queries = [q.get("query", "") for q in dataset.get("queries", [])]
    
    # Default queries if none provided
    if queries is None:
        queries = [
            "Find me wireless headphones under $100",
            "What are the best laptops for gaming?",
            "Show me running shoes between $50 and $90"
        ]
    
    print(f"\n{'='*70}")
    print(f"Running {evaluation_type.upper()} Evaluation")
    print(f"{'='*70}\n")
    print(f"Queries: {len(queries)}")
    print(f"Base URL: {base_url}\n")
    
    results = None
    
    if evaluation_type == "llm" or evaluation_type == "all":
        print("Running LLM Quality Evaluation...")
        results = await evaluator.evaluate_batch(
            queries=queries,
            evaluation_types=["llm_quality"]
        )
        print(f"[OK] LLM Quality Evaluation Complete")
        print(f"  Average Score: {results.get('aggregate_metrics', {}).get('deepeval', {}).get('average_score', 0):.3f}\n")
    
    if evaluation_type == "retrieval" or evaluation_type == "all":
        print("Running Retrieval Quality Evaluation...")
        print("  Note: Retrieval evaluation requires relevance labels")
        print("  Use evaluate_ir_metrics.py for full IR evaluation\n")
        # IR metrics require labeled data - skip for now
        if results is None:
            results = {"note": "IR metrics require relevance labels"}
    
    if evaluation_type == "performance" or evaluation_type == "all":
        print("Running Performance Evaluation...")
        perf_suite = PerformanceTestSuite(base_url=base_url)
        
        perf_results = []
        for query in queries[:5]:  # Limit to 5 queries for performance
            result = await perf_suite.test_latency(query, iterations=iterations)
            perf_results.append(result)
        
        if results is None:
            results = {}
        results["performance"] = perf_results
        print(f"[OK] Performance Evaluation Complete\n")
    
    if evaluation_type == "mcp":
        print("Running MCP (Tool Layer) Evaluation...")
        # MCP evaluation requires a dataset with expectations
        if not dataset_path:
            dataset_path = "evaluation/datasets/mcp_eval_queries.json"
            print(f"  Using default MCP dataset: {dataset_path}")
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        mcp_suite = MCPTestSuite(base_url=base_url)
        results = await mcp_suite.evaluate_dataset(dataset=dataset, iterations=iterations)
        print(f"[OK] MCP Evaluation Complete\n")
    
    if evaluation_type == "system" or evaluation_type == "all":
        print("Running Full System Evaluation...")
        system_suite = SystemTestSuite(base_url=base_url)
        system_results = await system_suite.run_full_system_evaluation(queries)
        
        if results is None:
            results = system_results
        else:
            results["system"] = system_results
        print(f"[OK] System Evaluation Complete\n")
    
    if evaluation_type == "all":
        print("Running Comprehensive Evaluation...")
        results = await evaluator.evaluate_batch(
            queries=queries,
            evaluation_types=["full_system"]
        )
        print(f"[OK] Comprehensive Evaluation Complete\n")
    
    # Generate reports
    if results:
        print("Generating Reports...")
        
        # JSON report
        json_path = report_generator.generate_json_report(results)
        print(f"  [OK] JSON report: {json_path}")
        
        # CSV report (if results list available)
        if "results" in results and isinstance(results["results"], list):
            csv_path = report_generator.generate_csv_report(results["results"])
            print(f"  [OK] CSV report: {csv_path}")
        
        # HTML dashboard
        html_path = report_generator.generate_html_dashboard(results)
        print(f"  [OK] HTML dashboard: {html_path}")
        
        print(f"\n{'='*70}")
        print("Evaluation Complete!")
        print(f"{'='*70}\n")
        
        # Print summary
        if "aggregate_metrics" in results:
            aggregates = results["aggregate_metrics"]
            if "deepeval" in aggregates:
                deepeval = aggregates["deepeval"]
                print(f"DeepEval Average Score: {deepeval.get('average_score', 0):.3f}")
                print(f"Pass Rate: {deepeval.get('pass_rate', 0):.1%}")
            
            if "performance" in aggregates:
                perf = aggregates["performance"]
                print(f"Average Total Time: {perf.get('average_total_time', 0):.2f}s")
                print(f"Average TTFT: {perf.get('average_ttft', 0):.2f}s")
    else:
        print("No results to report")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive evaluation of the shopping assistant"
    )
    
    parser.add_argument(
        "--type",
        choices=["llm", "retrieval", "performance", "system", "mcp", "all"],
        default="all",
        help="Type of evaluation to run (default: all)"
    )
    
    parser.add_argument(
        "--queries",
        type=str,
        help="Path to JSON file with test queries"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        help="Path to dataset JSON file (alternative to --queries)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation/reports",
        help="Output directory for reports (default: evaluation/reports)"
    )
    
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:3565",
        help="Base URL of API server (default: http://localhost:3565)"
    )
    
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of iterations for performance tests (default: 1)"
    )
    
    args = parser.parse_args()
    
    # Determine queries source
    queries_path = args.queries or args.dataset
    
    try:
        asyncio.run(run_evaluation(
            evaluation_type=args.type,
            queries=None,  # Will be loaded from file if provided
            dataset_path=queries_path,
            output_dir=args.output,
            base_url=args.base_url,
            iterations=args.iterations
        ))
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

