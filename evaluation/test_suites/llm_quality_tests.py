"""LLM quality test suite using DeepEval."""
import pytest
import asyncio
from typing import List, Dict, Any
from evaluation.deepeval_integration import deepeval_evaluator
from evaluation.unified_evaluator import UnifiedEvaluator


class LLMQualityTestSuite:
    """Test suite for LLM quality evaluation."""
    
    def __init__(self, base_url: str = "http://localhost:3565"):
        """Initialize test suite.
        
        Args:
            base_url: Base URL of the API server
        """
        self.evaluator = UnifiedEvaluator(base_url=base_url)
    
    async def test_relevance(self, query: str, response: str, context: List[str] = None) -> Dict[str, Any]:
        """Test response relevance to query."""
        result = await deepeval_evaluator.evaluate_query(
            query=query,
            response=response,
            context=context
        )
        return {
            "test": "relevance",
            "query": query,
            "result": result,
            "passed": result.get("metrics", {}).get("answerrelevancy", {}).get("passed", False)
        }
    
    async def test_completeness(self, query: str, response: str) -> Dict[str, Any]:
        """Test response completeness."""
        result = await deepeval_evaluator.evaluate_query(
            query=query,
            response=response
        )
        return {
            "test": "completeness",
            "query": query,
            "result": result,
            "passed": result.get("metrics", {}).get("summarization", {}).get("passed", False)
        }
    
    async def test_coherence(self, query: str, response: str) -> Dict[str, Any]:
        """Test response coherence."""
        result = await deepeval_evaluator.evaluate_query(
            query=query,
            response=response
        )
        return {
            "test": "coherence",
            "query": query,
            "result": result,
            "passed": result.get("metrics", {}).get("coherence", {}).get("passed", False)
        }
    
    async def test_bias(self, query: str, response: str) -> Dict[str, Any]:
        """Test for bias in response."""
        result = await deepeval_evaluator.evaluate_query(
            query=query,
            response=response
        )
        return {
            "test": "bias",
            "query": query,
            "result": result,
            "passed": result.get("metrics", {}).get("bias", {}).get("passed", False)
        }
    
    async def test_tone(self, query: str, response: str) -> Dict[str, Any]:
        """Test response tone appropriateness."""
        result = await deepeval_evaluator.evaluate_query(
            query=query,
            response=response
        )
        return {
            "test": "tone",
            "query": query,
            "result": result,
            "passed": result.get("metrics", {}).get("tone", {}).get("passed", False)
        }
    
    async def run_full_suite(self, queries: List[str]) -> Dict[str, Any]:
        """Run full LLM quality test suite.
        
        Args:
            queries: List of test queries
        
        Returns:
            Dictionary with test results
        """
        results = await self.evaluator.evaluate_batch(
            queries=queries,
            evaluation_types=["llm_quality"]
        )
        return results


# Pytest integration
@pytest.mark.asyncio
async def test_llm_quality_relevance():
    """Pytest test for LLM relevance."""
    suite = LLMQualityTestSuite()
    result = await suite.test_relevance(
        query="Find me wireless headphones under $100",
        response="Here are some wireless headphones under $100..."
    )
    assert result["passed"], f"Relevance test failed: {result['result']}"


@pytest.mark.asyncio
async def test_llm_quality_completeness():
    """Pytest test for LLM completeness."""
    suite = LLMQualityTestSuite()
    result = await suite.test_completeness(
        query="What are the best laptops?",
        response="Here are the best laptops..."
    )
    assert result["passed"], f"Completeness test failed: {result['result']}"

