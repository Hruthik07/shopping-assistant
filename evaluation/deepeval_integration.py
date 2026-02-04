"""DeepEval integration for LLM quality evaluation."""
import asyncio
import os
from typing import Dict, Any, List, Optional

# Conditional import - deepeval may not be compatible with all LangChain versions
try:
    from deepeval import evaluate
    from deepeval.test_case import LLMTestCase
    from deepeval.metrics import (
        FaithfulnessMetric,  # Similar to Groundedness
        AnswerRelevancyMetric,
        ContextualRelevancyMetric,
        ContextualPrecisionMetric,
        SummarizationMetric,
        BiasMetric,
        HallucinationMetric,  # Similar to Coherence/Tone
        ToxicityMetric  # Safety metric
    )
    DEEPEVAL_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    DEEPEVAL_AVAILABLE = False
    evaluate = None
    LLMTestCase = None
    FaithfulnessMetric = None
    AnswerRelevancyMetric = None
    ContextualRelevancyMetric = None
    ContextualPrecisionMetric = None
    SummarizationMetric = None
    BiasMetric = None
    HallucinationMetric = None
    ToxicityMetric = None

from evaluation.deepeval_config import deepeval_config, METRIC_THRESHOLDS
from src.analytics.logger import logger


class DeepEvalEvaluator:
    """DeepEval-based evaluator for LLM quality metrics."""
    
    def __init__(self):
        """Initialize DeepEval evaluator."""
        if not DEEPEVAL_AVAILABLE:
            self.enabled = False
            logger.warning("DeepEval is not available (import error). DeepEval features will be disabled.")
            return
        
        self.enabled = deepeval_config.is_enabled()
        if not self.enabled:
            logger.warning("DeepEval is disabled or not configured")
    
    async def evaluate_query(
        self,
        query: str,
        response: str,
        context: Optional[List[str]] = None,
        expected_output: Optional[str] = None
    ) -> Dict[str, Any]:
        """Evaluate a single query-response pair.
        
        Args:
            query: User query
            response: LLM response
            context: Retrieved context (for RAG metrics)
            expected_output: Expected output (optional, for comparison)
        
        Returns:
            Dictionary with metric scores and pass/fail status
        """
        if not DEEPEVAL_AVAILABLE or not self.enabled:
            return {"enabled": False, "error": "DeepEval not available or not enabled"}
        
        try:
            # Format context properly for DeepEval
            # DeepEval expects context as a list of strings
            formatted_context = []
            if context:
                if isinstance(context, list):
                    # If already a list of strings, use as-is
                    formatted_context = [str(c) for c in context if c and str(c).strip()]
                else:
                    # If single string or other format, convert to list
                    formatted_context = [str(context)] if str(context).strip() else []
            
            # Create test case with comprehensive error handling
            try:
                # Validate inputs
                if not query or not isinstance(query, str):
                    raise ValueError("Query must be a non-empty string")
                if not response or not isinstance(response, str):
                    raise ValueError("Response must be a non-empty string")
                
                # DeepEval LLMTestCase accepts both 'context' and 'retrieval_context'
                # Use 'retrieval_context' for RAG metrics, 'context' for general metrics
                test_case_params = {
                    "input": query,
                    "actual_output": response,
                }
                
                # Add expected_output if provided
                if expected_output:
                    if not isinstance(expected_output, str):
                        logger.warning(f"Expected output is not a string, converting: {type(expected_output)}")
                        expected_output = str(expected_output)
                    test_case_params["expected_output"] = expected_output
                
                # Add context/retrieval_context if available
                if formatted_context:
                    # Use retrieval_context for RAG-specific metrics
                    test_case_params["retrieval_context"] = formatted_context
                    # Also set context for metrics that need it (like Hallucination)
                    test_case_params["context"] = formatted_context
                
                test_case = LLMTestCase(**test_case_params)
            except (ValueError, TypeError) as validation_err:
                logger.error(f"Input validation failed for LLMTestCase: {validation_err}")
                return {
                    "enabled": True,
                    "error": f"Input validation failed: {str(validation_err)}",
                    "overall_score": 0.0,
                    "all_passed": False
                }
            except Exception as test_case_err:
                logger.error(f"Failed to create LLMTestCase: {test_case_err}", exc_info=True)
                return {
                    "enabled": True,
                    "error": f"Test case creation failed: {str(test_case_err)}",
                    "overall_score": 0.0,
                    "all_passed": False
                }
            
            # Initialize metrics
            # NOTE: Each metric makes 2-5 LLM calls internally, so 8 metrics = 16-40 API calls!
            # To reduce costs, you can comment out expensive metrics or use DEEPEVAL_METRICS env var
            metrics_list = []
            results = {}
            
            # Get which metrics to run from environment (default: all)
            enabled_metrics = os.getenv("DEEPEVAL_METRICS", "all").lower().split(",")
            run_all = "all" in enabled_metrics or len(enabled_metrics) == 0
            
            # Faithfulness (similar to Groundedness - checks if response is faithful to context)
            if context and (run_all or "faithfulness" in enabled_metrics):
                faithfulness = FaithfulnessMetric(threshold=METRIC_THRESHOLDS.get("faithfulness", 0.7))
                metrics_list.append(("faithfulness", faithfulness, None))
            
            # Answer Relevancy (ESSENTIAL - most important metric)
            if run_all or "answer_relevancy" in enabled_metrics:
                answer_relevancy = AnswerRelevancyMetric(threshold=METRIC_THRESHOLDS["answer_relevancy"])
                metrics_list.append(("answer_relevancy", answer_relevancy, None))
            
            # Contextual Relevancy (for RAG) - EXPENSIVE (3-5 calls, 120s timeout)
            if context and (run_all or "contextual_relevancy" in enabled_metrics):
                contextual_relevancy = ContextualRelevancyMetric(threshold=METRIC_THRESHOLDS["contextual_relevancy"])
                metrics_list.append(("contextual_relevancy", contextual_relevancy, 120.0))  # 120s timeout
            
            # Contextual Precision (for RAG) - keep in list but mark as optional
            if context and (run_all or "contextual_precision" in enabled_metrics):
                contextual_precision = ContextualPrecisionMetric(threshold=METRIC_THRESHOLDS["contextual_precision"])
                metrics_list.append(("contextual_precision", contextual_precision, expected_output is not None))  # Skip if no expected_output
            
            # Summarization (completeness) - EXPENSIVE
            if run_all or "summarization" in enabled_metrics:
                summarization = SummarizationMetric(threshold=METRIC_THRESHOLDS["summarization"])
                metrics_list.append(("summarization", summarization, None))
            
            # Bias
            if run_all or "bias" in enabled_metrics:
                bias = BiasMetric(threshold=METRIC_THRESHOLDS["bias"])
                metrics_list.append(("bias", bias, None))
            
            # Hallucination (checks for made-up information) - requires context
            if context and (run_all or "hallucination" in enabled_metrics):
                hallucination = HallucinationMetric(threshold=METRIC_THRESHOLDS.get("hallucination", 0.7))
                metrics_list.append(("hallucination", hallucination, None))
            
            # Toxicity (safety check) - ESSENTIAL for safety
            if run_all or "toxicity" in enabled_metrics:
                toxicity = ToxicityMetric(threshold=METRIC_THRESHOLDS.get("toxicity", 0.8))
                metrics_list.append(("toxicity", toxicity, None))
            
            # Run evaluation with timeout handling
            for metric_name, metric, timeout_or_skip in metrics_list:
                # Check if metric should be skipped
                if isinstance(timeout_or_skip, bool) and not timeout_or_skip:
                    results[metric_name] = {
                        "score": None,
                        "passed": None,
                        "status": "skipped",
                        "reason": "expected_output not provided"
                    }
                    logger.debug(f"Skipping {metric_name} - required input not provided")
                    continue
                
                try:
                    # Run metric with timeout if specified
                    if isinstance(timeout_or_skip, (int, float)) and timeout_or_skip > 0:
                        # Run synchronous metric.measure() in executor with timeout
                        loop = asyncio.get_event_loop()
                        try:
                            await asyncio.wait_for(
                                loop.run_in_executor(None, metric.measure, test_case),
                                timeout=timeout_or_skip
                            )
                        except asyncio.TimeoutError:
                            logger.warning(f"Timeout ({timeout_or_skip}s) evaluating {metric_name}")
                            results[metric_name] = {
                                "score": 0.0,
                                "passed": False,
                                "status": "timeout",
                                "error": f"Metric evaluation exceeded {timeout_or_skip}s timeout"
                            }
                            continue
                    else:
                        # Run synchronously for fast metrics
                        metric.measure(test_case)
                    
                    results[metric_name] = {
                        "score": metric.score,
                        "passed": metric.success,
                        "status": "success",
                        "reason": metric.reason if hasattr(metric, "reason") else None
                    }
                except Exception as e:
                    logger.error(f"Failed to evaluate {metric_name}: {e}")
                    results[metric_name] = {
                        "score": 0.0,
                        "passed": False,
                        "status": "error",
                        "error": str(e)[:200]  # Truncate long error messages
                    }
            
            # Calculate overall score (only from successful metrics)
            scores = [r["score"] for r in results.values() if r.get("status") == "success" and "score" in r and r["score"] is not None]
            overall_score = sum(scores) / len(scores) if scores else 0.0
            
            # Check if all successful metrics passed (skip skipped/timeout/error metrics)
            successful_metrics = [r for r in results.values() if r.get("status") == "success"]
            all_passed = all(r.get("passed", False) for r in successful_metrics) if successful_metrics else False
            
            deepeval_result = {
                "enabled": True,
                "overall_score": round(overall_score, 3),
                "all_passed": all_passed,
                "metrics": results,
                "thresholds": METRIC_THRESHOLDS
            }
            
            # Export to CloudWatch (async, non-blocking)
            try:
                from src.analytics.evaluation_exporter import evaluation_exporter
                # Schedule async export (fire and forget)
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.create_task(evaluation_exporter.export_deepeval_scores(deepeval_result))
                except RuntimeError:
                    # No event loop, skip export
                    pass
            except Exception:
                pass  # Graceful degradation if CloudWatch unavailable
            
            return deepeval_result
            
        except Exception as e:
            logger.error(f"DeepEval evaluation failed: {e}", exc_info=True)
            return {
                "enabled": True,
                "error": str(e),
                "overall_score": 0.0,
                "all_passed": False
            }
    
    async def evaluate_batch(
        self,
        test_cases: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Evaluate a batch of test cases.
        
        Args:
            test_cases: List of dicts with keys: query, response, context (optional), expected_output (optional)
        
        Returns:
            Dictionary with aggregate results
        """
        if not DEEPEVAL_AVAILABLE or not self.enabled:
            return {"enabled": False, "error": "DeepEval not available or not enabled"}
        
        results = []
        for test_case in test_cases:
            result = await self.evaluate_query(
                query=test_case.get("query", ""),
                response=test_case.get("response", ""),
                context=test_case.get("context"),
                expected_output=test_case.get("expected_output")
            )
            results.append(result)
        
        # Calculate aggregate metrics
        overall_scores = [r.get("overall_score", 0.0) for r in results if r.get("enabled")]
        avg_score = sum(overall_scores) / len(overall_scores) if overall_scores else 0.0
        pass_rate = sum(1 for r in results if r.get("all_passed", False)) / len(results) if results else 0.0
        
        batch_result = {
            "enabled": True,
            "total_cases": len(test_cases),
            "average_score": round(avg_score, 3),
            "pass_rate": round(pass_rate, 3),
            "results": results,
            "overall_score": round(avg_score, 3)  # For CloudWatch export
        }
        
        # Export batch results to CloudWatch
        try:
            from src.analytics.evaluation_exporter import evaluation_exporter
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(evaluation_exporter.export_deepeval_scores(batch_result))
            except RuntimeError:
                pass
        except Exception:
            pass  # Graceful degradation
        
        return batch_result


# Global evaluator instance
deepeval_evaluator = DeepEvalEvaluator()

