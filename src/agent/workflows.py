"""Multi-step agent workflows."""
from typing import Dict, Any, List, Optional
from src.analytics.logger import logger


class WorkflowManager:
    """Manage multi-step agent workflows."""
    
    def create_search_workflow(self, query: str) -> Dict[str, Any]:
        """Create a product search workflow."""
        return {
            "type": "search",
            "steps": [
                {"action": "understand_query", "input": query},
                {"action": "api_search", "input": query},
                {"action": "filter_results", "input": None},
                {"action": "format_response", "input": None}
            ],
            "current_step": 0,
            "state": {}
        }
    
    def create_question_workflow(self, query: str, product_id: Optional[str] = None) -> Dict[str, Any]:
        """Create a question-answering workflow."""
        return {
            "type": "question",
            "steps": [
                {"action": "understand_query", "input": query},
                {"action": "api_retrieve_context", "input": {"query": query, "product_id": product_id}},
                {"action": "web_search_if_needed", "input": query},
                {"action": "synthesize_answer", "input": None}
            ],
            "current_step": 0,
            "state": {}
        }
    
    def create_comparison_workflow(self, products: List[str]) -> Dict[str, Any]:
        """Create a product comparison workflow."""
        return {
            "type": "compare",
            "steps": [
                {"action": "get_product_details", "input": products},
                {"action": "compare_features", "input": None},
                {"action": "compare_prices", "input": None},
                {"action": "generate_comparison", "input": None}
            ],
            "current_step": 0,
            "state": {}
        }
    
    def execute_step(self, workflow: Dict[str, Any], step_data: Any = None) -> Dict[str, Any]:
        """Execute the next step in workflow."""
        if workflow["current_step"] >= len(workflow["steps"]):
            return {
                "complete": True,
                "result": workflow.get("state", {}).get("result")
            }
        
        step = workflow["steps"][workflow["current_step"]]
        action = step["action"]
        input_data = step_data or step.get("input")
        
        # Update workflow state
        workflow["current_step"] += 1
        
        return {
            "action": action,
            "input": input_data,
            "workflow": workflow,
            "complete": workflow["current_step"] >= len(workflow["steps"])
        }


# Global workflow manager
workflow_manager = WorkflowManager()

