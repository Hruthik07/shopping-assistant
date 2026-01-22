"""Unified evaluation orchestrator combining all evaluation types."""
import asyncio
import httpx
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import json

from evaluation.deepeval_integration import deepeval_evaluator
from evaluation.langfuse_integration import langfuse_exporter
from evaluation.ir_metrics import IRMetrics
from evaluation.llm_quality_metrics import LLMQualityMetrics
from src.analytics.latency_tracker import latency_tracker
from src.analytics.logger import logger


class UnifiedEvaluator:
    """Main orchestrator for comprehensive evaluation."""
    
    def __init__(self, base_url: str = "http://localhost:3565"):
        """Initialize unified evaluator.
        
        Args:
            base_url: Base URL of the API server
        """
        self.base_url = base_url
        self.llm_quality_metrics = LLMQualityMetrics()
        self.results: List[Dict[str, Any]] = []
    
    async def evaluate_query(
        self,
        query: str,
        session_id: Optional[str] = None,
        evaluation_types: Optional[List[str]] = None,
        expected_output: Optional[str] = None,
        relevant_product_ids: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Evaluate a single query with specified evaluation types.
        
        Args:
            query: User query
            session_id: Optional session ID
            evaluation_types: List of evaluation types to run:
                - "llm_quality": DeepEval LLM quality metrics
                - "retrieval": IR metrics (requires relevance labels)
                - "performance": Latency and performance metrics
                - "full_system": All metrics combined
        
        Returns:
            Dictionary with evaluation results
        """
        # Input validation
        if not query or not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        
        if evaluation_types is None:
            evaluation_types = ["full_system"]
        
        if not isinstance(evaluation_types, list) or not evaluation_types:
            raise ValueError("evaluation_types must be a non-empty list")
        
        # Validate evaluation types
        valid_types = ["llm_quality", "retrieval", "performance", "full_system"]
        invalid_types = [t for t in evaluation_types if t not in valid_types]
        if invalid_types:
            raise ValueError(f"Invalid evaluation types: {invalid_types}. Valid types: {valid_types}")
        
        if session_id is None:
            session_id = f"eval-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        if not isinstance(session_id, str):
            raise ValueError("session_id must be a string")
        
        start_time = asyncio.get_event_loop().time()
        results = {
            "query": query,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "evaluation_types": evaluation_types
        }
        
        try:
            # Run query through API (increased timeout for DeepEval evaluation)
            async with httpx.AsyncClient(timeout=180.0) as client:
                try:
                    response = await client.post(
                        f"{self.base_url}/api/chat/",
                        json={
                            "message": query,
                            "session_id": session_id
                        },
                        headers={"Content-Type": "application/json"}
                    )
                    response.raise_for_status()
                    api_result = response.json()
                    
                    # Validate API response
                    if not isinstance(api_result, dict):
                        raise ValueError("API response is not a dictionary")
                    
                except httpx.TimeoutException:
                    raise TimeoutError(f"API request timed out after 180s for query: {query[:50]}...")
                except httpx.HTTPStatusError as e:
                    raise ValueError(f"API returned error status {e.response.status_code}: {e.response.text[:200]}")
                except httpx.RequestError as e:
                    raise ConnectionError(f"Failed to connect to API: {str(e)}")
            
            elapsed = asyncio.get_event_loop().time() - start_time
            
            # Extract data
            response_text = api_result.get("response", "")
            products = api_result.get("products", [])
            tools_used = api_result.get("tools_used", [])
            latency_breakdown = api_result.get("latency_breakdown", {})
            langfuse_trace_id = api_result.get("langfuse_trace_id")
            
            # Run evaluations based on types
            if "llm_quality" in evaluation_types or "full_system" in evaluation_types:
                # DeepEval metrics - format product context efficiently to reduce costs
                # Reduced from 10 to 5 products and truncated descriptions to save tokens
                context = []
                for p in products[:5]:  # Reduced to top 5 products to save costs
                    product_text = f"Product: {p.get('name', 'Unknown')}"
                    if p.get('price'):
                        product_text += f". Price: ${p.get('price', '')}"
                    if p.get('brand'):
                        product_text += f". Brand: {p.get('brand', '')}"
                    # Truncate description to 50 chars (was 200) to reduce token usage
                    if p.get('description'):
                        product_text += f". {p.get('description', '')[:50]}"
                    context.append(product_text)
                
                # Only pass context if we have products
                deepeval_results = await deepeval_evaluator.evaluate_query(
                    query=query,
                    response=response_text,
                    context=context if context else None,
                    expected_output=expected_output
                )
                results["deepeval"] = deepeval_results
                
                # Export to Langfuse if trace ID available
                if langfuse_trace_id and deepeval_results.get("enabled"):
                    langfuse_exporter.export_evaluation_results(
                        trace_id=langfuse_trace_id,
                        evaluation_results=deepeval_results,
                        source="deepeval"
                    )
                
                # Legacy LLM quality metrics (fallback)
                legacy_quality = self.llm_quality_metrics.evaluate_response(
                    query=query,
                    response=response_text,
                    products=products,
                    tools_used=tools_used
                )
                results["llm_quality_legacy"] = legacy_quality
            
            if "retrieval" in evaluation_types or "full_system" in evaluation_types:
                # IR metrics (requires relevance labels)
                if products:
                    # Check if relevance labels are provided
                    if relevant_product_ids and len(relevant_product_ids) > 0:
                        # Calculate IR metrics (includes all metrics: Precision@K, Recall@K, NDCG@K, MRR, MAP, Context Precision)
                        try:
                            ir_scores = IRMetrics.evaluate_retrieval(
                                query=query,
                                retrieved_products=products,
                                relevant_products=relevant_product_ids,
                                k_values=[1, 3, 5, 10]
                            )
                            ir_scores["status"] = "success"
                            results["ir_metrics"] = ir_scores
                        except Exception as e:
                            logger.error(f"Failed to calculate IR metrics: {e}")
                            results["ir_metrics"] = {
                                "status": "error",
                                "error": str(e),
                                "retrieved_count": len(products)
                            }
                    else:
                        # Mark as requires_labels
                        retrieved_ids = [p.get("id") or p.get("metadata", {}).get("id", str(i)) for i, p in enumerate(products)]
                        results["ir_metrics"] = {
                            "status": "requires_labels",
                            "retrieved_count": len(retrieved_ids),
                            "note": "IR metrics require relevance labels (relevant_product_ids) to calculate Precision@K, Recall@K, NDCG@K, MRR, MAP, and Context Precision"
                        }
                else:
                    results["ir_metrics"] = {
                        "status": "no_products",
                        "retrieved_count": 0,
                        "note": "No products retrieved for IR evaluation"
                    }
            
            if "performance" in evaluation_types or "full_system" in evaluation_types:
                # Performance metrics
                results["performance"] = {
                    "total_time": elapsed,
                    "latency_breakdown": latency_breakdown,
                    "ttft": latency_breakdown.get("ttft", 0),
                    "response_length": len(response_text),
                    "products_count": len(products),
                    "tools_used": tools_used
                }
            
            results["success"] = True
            results["error"] = None
            
        except Exception as e:
            elapsed = asyncio.get_event_loop().time() - start_time
            logger.error(f"Evaluation failed for query: {query[:50]}... Error: {e}", exc_info=True)
            results["success"] = False
            results["error"] = str(e)
            results["total_time"] = elapsed
        
        self.results.append(results)
        return results
    
    async def evaluate_batch(
        self,
        queries: List[str],
        evaluation_types: Optional[List[str]] = None,
        max_concurrent: int = 5,
        timeout_per_query: float = 300.0  # Increased to 5 minutes for DeepEval metrics
    ) -> Dict[str, Any]:
        """Evaluate a batch of queries with concurrency control and timeout.
        
        Args:
            queries: List of queries to evaluate
            evaluation_types: List of evaluation types to run
            max_concurrent: Maximum concurrent evaluations (default: 5)
            timeout_per_query: Timeout per query in seconds (default: 180.0)
        
        Returns:
            Dictionary with aggregate results
        """
        if evaluation_types is None:
            evaluation_types = ["full_system"]
        
        # Validate inputs
        if not queries or not isinstance(queries, list):
            raise ValueError("queries must be a non-empty list")
        
        if not all(isinstance(q, str) and q.strip() for q in queries):
            raise ValueError("All queries must be non-empty strings")
        
        logger.info(f"Starting batch evaluation of {len(queries)} queries (max concurrent: {max_concurrent})")
        
        # Run evaluations with concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def evaluate_with_semaphore(query: str):
            async with semaphore:
                try:
                    return await asyncio.wait_for(
                        self.evaluate_query(query, evaluation_types=evaluation_types),
                        timeout=timeout_per_query
                    )
                except asyncio.TimeoutError:
                    logger.error(f"Query evaluation timed out: {query[:50]}...")
                    return {
                        "query": query,
                        "success": False,
                        "error": f"Evaluation timed out after {timeout_per_query}s"
                    }
                except Exception as e:
                    logger.error(f"Query evaluation failed: {query[:50]}... Error: {e}")
                    return {
                        "query": query,
                        "success": False,
                        "error": str(e)
                    }
        
        tasks = [evaluate_with_semaphore(q) for q in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and validate results
        valid_results = []
        failed_results = []
        exceptions = []
        
        for r in results:
            if isinstance(r, Exception):
                exceptions.append(r)
                logger.error(f"Evaluation raised exception: {r}")
            elif isinstance(r, dict):
                if r.get("success"):
                    valid_results.append(r)
                else:
                    failed_results.append(r)
            else:
                logger.warning(f"Unexpected result type: {type(r)}")
                failed_results.append({
                    "success": False,
                    "error": f"Unexpected result type: {type(r)}"
                })
        
        if exceptions:
            logger.warning(f"{len(exceptions)} evaluations raised exceptions")
        
        # Calculate aggregates
        aggregate = self._calculate_aggregate_metrics(valid_results, evaluation_types)
        
        return {
            "total_queries": len(queries),
            "successful": len(valid_results),
            "failed": len(failed_results),
            "evaluation_types": evaluation_types,
            "aggregate_metrics": aggregate,
            "results": valid_results,
            "failures": failed_results
        }
    
    def _calculate_aggregate_metrics(
        self,
        results: List[Dict[str, Any]],
        evaluation_types: List[str]
    ) -> Dict[str, Any]:
        """Calculate aggregate metrics from results."""
        if not results:
            return {}
        
        aggregates = {}
        
        # DeepEval aggregates
        if "llm_quality" in evaluation_types or "full_system" in evaluation_types:
            deepeval_scores = [
                r.get("deepeval", {}).get("overall_score", 0.0)
                for r in results
                if r.get("deepeval", {}).get("enabled")
            ]
            if deepeval_scores:
                aggregates["deepeval"] = {
                    "average_score": sum(deepeval_scores) / len(deepeval_scores),
                    "min_score": min(deepeval_scores),
                    "max_score": max(deepeval_scores),
                    "pass_rate": sum(1 for s in deepeval_scores if s >= 0.7) / len(deepeval_scores)
                }
        
        # IR metrics aggregates
        if "retrieval" in evaluation_types or "full_system" in evaluation_types:
            ir_successful = [
                r.get("ir_metrics", {})
                for r in results
                if r.get("ir_metrics", {}).get("status") == "success"
            ]
            if ir_successful:
                precision_scores = [m.get("precision@5", 0) for m in ir_successful if "precision@5" in m]
                recall_scores = [m.get("recall@5", 0) for m in ir_successful if "recall@5" in m]
                ndcg_scores = [m.get("ndcg@5", 0) for m in ir_successful if "ndcg@5" in m]
                mrr_scores = [m.get("mrr", 0) for m in ir_successful if "mrr" in m]
                map_scores = [m.get("map", 0) for m in ir_successful if "map" in m]
                
                aggregates["ir_metrics"] = {}
                if precision_scores:
                    aggregates["ir_metrics"]["avg_precision@5"] = sum(precision_scores) / len(precision_scores)
                if recall_scores:
                    aggregates["ir_metrics"]["avg_recall@5"] = sum(recall_scores) / len(recall_scores)
                if ndcg_scores:
                    aggregates["ir_metrics"]["avg_ndcg@5"] = sum(ndcg_scores) / len(ndcg_scores)
                if mrr_scores:
                    aggregates["ir_metrics"]["avg_mrr"] = sum(mrr_scores) / len(mrr_scores)
                if map_scores:
                    aggregates["ir_metrics"]["avg_map"] = sum(map_scores) / len(map_scores)
        
        # Performance aggregates
        if "performance" in evaluation_types or "full_system" in evaluation_types:
            total_times = [r.get("performance", {}).get("total_time", 0) for r in results if r.get("performance")]
            ttfts = [
                r.get("performance", {}).get("ttft", 0)
                for r in results
                if r.get("performance", {}).get("ttft", 0) > 0
            ]
            
            if total_times:
                aggregates["performance"] = {
                    "average_total_time": sum(total_times) / len(total_times),
                    "min_total_time": min(total_times),
                    "max_total_time": max(total_times),
                    "average_ttft": sum(ttfts) / len(ttfts) if ttfts else 0,
                    "min_ttft": min(ttfts) if ttfts else 0,
                    "max_ttft": max(ttfts) if ttfts else 0
                }
        
        return aggregates
    
    def reset(self):
        """Reset evaluation results."""
        self.results.clear()
    
    def get_results(self) -> List[Dict[str, Any]]:
        """Get all evaluation results."""
        return self.results.copy()

