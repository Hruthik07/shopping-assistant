"""Full system integration test suite."""
import asyncio
import httpx
from typing import List, Dict, Any
from evaluation.unified_evaluator import UnifiedEvaluator
from src.analytics.logger import logger


class SystemTestSuite:
    """Test suite for full system integration evaluation."""
    
    def __init__(self, base_url: str = "http://localhost:3565"):
        """Initialize system test suite.
        
        Args:
            base_url: Base URL of the API server
        """
        self.base_url = base_url
        self.evaluator = UnifiedEvaluator(base_url=base_url)
    
    async def test_multi_turn_conversation(
        self,
        conversation_turns: List[str],
        session_id: str = None
    ) -> Dict[str, Any]:
        """Test multi-turn conversation handling.
        
        Args:
            conversation_turns: List of queries in conversation order
            session_id: Optional session ID
        
        Returns:
            Dictionary with test results
        """
        if session_id is None:
            session_id = f"system-test-{asyncio.get_event_loop().time()}"
        
        responses = []
        context_maintained = True
        
        for i, query in enumerate(conversation_turns):
            try:
                async with httpx.AsyncClient(timeout=120.0) as client:
                    response = await client.post(
                        f"{self.base_url}/api/chat/",
                        json={"message": query, "session_id": session_id},
                        headers={"Content-Type": "application/json"}
                    )
                    response.raise_for_status()
                    result = response.json()
                
                response_text = result.get("response", "")
                responses.append({
                    "turn": i + 1,
                    "query": query,
                    "response": response_text[:200],  # First 200 chars
                    "products_count": len(result.get("products", [])),
                    "tools_used": result.get("tools_used", [])
                })
                
                # Check if context is maintained (simple heuristic: response mentions previous query terms)
                if i > 0:
                    prev_query_terms = set(conversation_turns[i-1].lower().split())
                    response_lower = response_text.lower()
                    mentioned_terms = sum(1 for term in prev_query_terms if len(term) > 3 and term in response_lower)
                    if mentioned_terms == 0 and len(prev_query_terms) > 2:
                        context_maintained = False
                
            except Exception as e:
                logger.error(f"Multi-turn test turn {i+1} failed: {e}")
                responses.append({
                    "turn": i + 1,
                    "query": query,
                    "error": str(e)
                })
                context_maintained = False
        
        return {
            "test": "multi_turn_conversation",
            "session_id": session_id,
            "turns": len(conversation_turns),
            "context_maintained": context_maintained,
            "responses": responses,
            "passed": context_maintained and len(responses) == len(conversation_turns)
        }
    
    async def test_tool_usage(
        self,
        query: str,
        expected_tools: List[str],
        session_id: str = None
    ) -> Dict[str, Any]:
        """Test that correct tools are used.
        
        Args:
            query: Test query
            expected_tools: List of expected tool names
            session_id: Optional session ID
        
        Returns:
            Dictionary with test results
        """
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/chat/",
                    json={"message": query, "session_id": session_id},
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                result = response.json()
            
            tools_used = result.get("tools_used", [])
            expected_set = set(expected_tools)
            used_set = set(tools_used)
            
            # Check if expected tools were used
            missing_tools = expected_set - used_set
            unexpected_tools = used_set - expected_set
            
            return {
                "test": "tool_usage",
                "query": query,
                "expected_tools": expected_tools,
                "tools_used": tools_used,
                "missing_tools": list(missing_tools),
                "unexpected_tools": list(unexpected_tools),
                "passed": len(missing_tools) == 0
            }
            
        except Exception as e:
            logger.error(f"Tool usage test failed: {e}")
            return {
                "test": "tool_usage",
                "query": query,
                "error": str(e),
                "passed": False
            }
    
    async def test_end_to_end_workflow(
        self,
        workflow_queries: List[Dict[str, Any]],
        session_id: str = None
    ) -> Dict[str, Any]:
        """Test end-to-end workflow.
        
        Args:
            workflow_queries: List of dicts with keys: query, expected_products (optional), expected_tools (optional)
            session_id: Optional session ID
        
        Returns:
            Dictionary with test results
        """
        if session_id is None:
            session_id = f"workflow-test-{asyncio.get_event_loop().time()}"
        
        results = []
        all_passed = True
        
        for i, workflow_step in enumerate(workflow_queries):
            query = workflow_step["query"]
            expected_products = workflow_step.get("expected_products", [])
            expected_tools = workflow_step.get("expected_tools", [])
            
            try:
                async with httpx.AsyncClient(timeout=120.0) as client:
                    response = await client.post(
                        f"{self.base_url}/api/chat/",
                        json={"message": query, "session_id": session_id},
                        headers={"Content-Type": "application/json"}
                    )
                    response.raise_for_status()
                    result = response.json()
                
                products = result.get("products", [])
                tools_used = result.get("tools_used", [])
                
                # Check expectations
                step_passed = True
                issues = []
                
                if expected_products:
                    product_names = [p.get("name", "").lower() for p in products]
                    missing = [ep for ep in expected_products if ep.lower() not in " ".join(product_names)]
                    if missing:
                        step_passed = False
                        issues.append(f"Missing expected products: {missing}")
                
                if expected_tools:
                    missing_tools = set(expected_tools) - set(tools_used)
                    if missing_tools:
                        step_passed = False
                        issues.append(f"Missing expected tools: {list(missing_tools)}")
                
                if not step_passed:
                    all_passed = False
                
                results.append({
                    "step": i + 1,
                    "query": query,
                    "passed": step_passed,
                    "products_count": len(products),
                    "tools_used": tools_used,
                    "issues": issues
                })
                
            except Exception as e:
                logger.error(f"Workflow test step {i+1} failed: {e}")
                results.append({
                    "step": i + 1,
                    "query": query,
                    "error": str(e),
                    "passed": False
                })
                all_passed = False
        
        return {
            "test": "end_to_end_workflow",
            "session_id": session_id,
            "steps": len(workflow_queries),
            "all_passed": all_passed,
            "results": results
        }
    
    async def run_full_system_evaluation(
        self,
        test_queries: List[str]
    ) -> Dict[str, Any]:
        """Run full system evaluation.
        
        Args:
            test_queries: List of test queries
        
        Returns:
            Dictionary with comprehensive evaluation results
        """
        results = await self.evaluator.evaluate_batch(
            queries=test_queries,
            evaluation_types=["full_system"]
        )
        return results

