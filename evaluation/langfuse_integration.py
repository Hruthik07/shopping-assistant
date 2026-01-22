"""Langfuse integration for evaluation export and tracking."""
from typing import Dict, Any, List, Optional
from langfuse import Langfuse
from src.analytics.langfuse_client import langfuse_client
from src.utils.config import settings
from src.analytics.logger import logger


class LangfuseEvaluationExporter:
    """Export evaluation results to Langfuse for tracking."""
    
    def __init__(self):
        """Initialize Langfuse evaluation exporter."""
        self.enabled = langfuse_client.enabled
        self.client = langfuse_client.client if langfuse_client.enabled else None
    
    def create_dataset(self, dataset_name: str, description: Optional[str] = None) -> Optional[Any]:
        """Create a Langfuse dataset for evaluation.
        
        Args:
            dataset_name: Name of the dataset
            description: Optional description
        
        Returns:
            Dataset instance or None if disabled
        """
        if not self.enabled or not self.client:
            return None
        
        try:
            dataset = self.client.create_dataset(
                name=dataset_name,
                description=description or f"Evaluation dataset: {dataset_name}"
            )
            logger.info(f"Created Langfuse dataset: {dataset_name}")
            return dataset
        except Exception as e:
            logger.error(f"Failed to create Langfuse dataset: {e}")
            return None
    
    def add_item_to_dataset(
        self,
        dataset_name: str,
        input: Dict[str, Any],
        expected_output: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Add an item to a Langfuse dataset.
        
        Args:
            dataset_name: Name of the dataset
            input: Input data (query, context, etc.)
            expected_output: Expected output (optional)
            metadata: Additional metadata
        
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled or not self.client:
            return False
        
        try:
            self.client.create_dataset_item(
                dataset_name=dataset_name,
                input=input,
                expected_output=expected_output,
                metadata=metadata or {}
            )
            return True
        except Exception as e:
            logger.error(f"Failed to add item to Langfuse dataset: {e}")
            return False
    
    def create_evaluation(
        self,
        trace_id: str,
        name: str,
        score: float,
        comment: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Create an evaluation score for a trace.
        
        Args:
            trace_id: Langfuse trace ID
            name: Evaluation name (e.g., "relevance", "completeness")
            score: Score value (0-1)
            comment: Optional comment
            metadata: Additional metadata
        
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled or not self.client:
            return False
        
        try:
            langfuse_client.score(
                trace_id=trace_id,
                name=name,
                value=score,
                comment=comment
            )
            return True
        except Exception as e:
            logger.error(f"Failed to create Langfuse evaluation: {e}")
            return False
    
    def export_evaluation_results(
        self,
        trace_id: str,
        evaluation_results: Dict[str, Any],
        source: str = "deepeval"
    ) -> bool:
        """Export evaluation results to Langfuse.
        
        Args:
            trace_id: Langfuse trace ID
            evaluation_results: Dictionary with evaluation metrics
            source: Source of evaluation (e.g., "deepeval", "ir_metrics")
        
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled or not trace_id:
            return False
        
        try:
            # Export individual metric scores
            metrics = evaluation_results.get("metrics", {})
            for metric_name, metric_data in metrics.items():
                if isinstance(metric_data, dict) and "score" in metric_data:
                    self.create_evaluation(
                        trace_id=trace_id,
                        name=f"{source}_{metric_name}",
                        score=metric_data["score"],
                        comment=metric_data.get("reason"),
                        metadata={"source": source, "passed": metric_data.get("passed", False)}
                    )
            
            # Export overall score
            if "overall_score" in evaluation_results:
                self.create_evaluation(
                    trace_id=trace_id,
                    name=f"{source}_overall",
                    score=evaluation_results["overall_score"],
                    comment=f"Overall {source} evaluation score",
                    metadata={"source": source, "all_passed": evaluation_results.get("all_passed", False)}
                )
            
            # Flush to ensure data is sent
            langfuse_client.flush()
            return True
            
        except Exception as e:
            logger.error(f"Failed to export evaluation results to Langfuse: {e}")
            return False
    
    def get_trace_evaluations(self, trace_id: str) -> List[Dict[str, Any]]:
        """Get all evaluations for a trace.
        
        Args:
            trace_id: Langfuse trace ID
        
        Returns:
            List of evaluation dictionaries
        """
        if not self.enabled or not self.client:
            return []
        
        try:
            # Note: This would require Langfuse API access
            # For now, return empty list as this is a read operation
            # that would need to be implemented via Langfuse API
            return []
        except Exception as e:
            logger.error(f"Failed to get trace evaluations: {e}")
            return []


# Global exporter instance
langfuse_exporter = LangfuseEvaluationExporter()

