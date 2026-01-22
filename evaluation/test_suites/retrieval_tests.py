"""Retrieval quality test suite using IR metrics."""
from typing import List, Dict, Any, Set
from evaluation.ir_metrics import IRMetrics

# Optional import - may fail if RAG components are missing
try:
    from evaluation.evaluate_ir_metrics import evaluate_retrieval_quality
    RETRIEVAL_EVAL_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    RETRIEVAL_EVAL_AVAILABLE = False
    evaluate_retrieval_quality = None

from src.analytics.logger import logger


class RetrievalTestSuite:
    """Test suite for retrieval quality evaluation."""
    
    def __init__(self):
        """Initialize retrieval test suite."""
        self.ir_metrics = IRMetrics()
    
    def test_precision_at_k(
        self,
        retrieved: List[str],
        relevant: Set[str],
        k: int = 5
    ) -> Dict[str, Any]:
        """Test Precision@K.
        
        Args:
            retrieved: List of retrieved product IDs
            relevant: Set of relevant product IDs
            k: Number of top results to consider
        
        Returns:
            Dictionary with test results
        """
        precision = self.ir_metrics.precision_at_k(retrieved, relevant, k)
        passed = precision >= 0.65  # Target threshold
        
        return {
            "test": f"precision@{k}",
            "score": precision,
            "passed": passed,
            "threshold": 0.65,
            "retrieved_count": len(retrieved),
            "relevant_count": len(relevant)
        }
    
    def test_recall_at_k(
        self,
        retrieved: List[str],
        relevant: Set[str],
        k: int = 10
    ) -> Dict[str, Any]:
        """Test Recall@K.
        
        Args:
            retrieved: List of retrieved product IDs
            relevant: Set of relevant product IDs
            k: Number of top results to consider
        
        Returns:
            Dictionary with test results
        """
        recall = self.ir_metrics.recall_at_k(retrieved, relevant, k)
        passed = recall >= 0.60  # Target threshold
        
        return {
            "test": f"recall@{k}",
            "score": recall,
            "passed": passed,
            "threshold": 0.60,
            "retrieved_count": len(retrieved),
            "relevant_count": len(relevant)
        }
    
    def test_ndcg_at_k(
        self,
        retrieved: List[str],
        relevant_products: Dict[str, float],
        k: int = 10
    ) -> Dict[str, Any]:
        """Test NDCG@K.
        
        Args:
            retrieved: List of retrieved product IDs
            relevant_products: Dict mapping product IDs to relevance scores (0-4)
            k: Number of top results to consider
        
        Returns:
            Dictionary with test results
        """
        ndcg = self.ir_metrics.ndcg_at_k(retrieved, relevant_products, k)
        passed = ndcg >= 0.75  # Target threshold
        
        return {
            "test": f"ndcg@{k}",
            "score": ndcg,
            "passed": passed,
            "threshold": 0.75,
            "retrieved_count": len(retrieved),
            "relevant_count": len(relevant_products)
        }
    
    def evaluate_query(
        self,
        query: str,
        retrieved_products: List[Dict[str, Any]],
        relevant_products: Dict[str, float],
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict[str, Any]:
        """Evaluate retrieval quality for a query.
        
        Args:
            query: User query
            retrieved_products: List of retrieved products
            relevant_products: Dict mapping product IDs to relevance scores
            k_values: List of K values to test
        
        Returns:
            Dictionary with evaluation results
        """
        retrieved_ids = [p.get("id", str(i)) for i, p in enumerate(retrieved_products)]
        relevant_set = set(relevant_products.keys())
        
        results = {}
        all_passed = True
        
        for k in k_values:
            # Precision@K
            precision_result = self.test_precision_at_k(retrieved_ids, relevant_set, k)
            results[f"precision@{k}"] = precision_result
            if not precision_result["passed"]:
                all_passed = False
            
            # Recall@K
            recall_result = self.test_recall_at_k(retrieved_ids, relevant_set, k)
            results[f"recall@{k}"] = recall_result
            if not recall_result["passed"]:
                all_passed = False
        
        # NDCG@10
        ndcg_result = self.test_ndcg_at_k(retrieved_ids, relevant_products, 10)
        results["ndcg@10"] = ndcg_result
        if not ndcg_result["passed"]:
            all_passed = False
        
        # MRR and MAP
        mrr = self.ir_metrics.reciprocal_rank(retrieved_ids, relevant_set)
        map_score = self.ir_metrics.average_precision(retrieved_ids, relevant_set)
        
        results["mrr"] = {
            "test": "mrr",
            "score": mrr,
            "passed": mrr >= 0.70,
            "threshold": 0.70
        }
        results["map"] = {
            "test": "map",
            "score": map_score,
            "passed": map_score >= 0.70,
            "threshold": 0.70
        }
        
        return {
            "query": query,
            "all_passed": all_passed,
            "metrics": results,
            "retrieved_count": len(retrieved_ids),
            "relevant_count": len(relevant_set)
        }
    
    def evaluate_batch(
        self,
        test_cases: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Evaluate a batch of test cases.
        
        Args:
            test_cases: List of dicts with keys: query, retrieved_products, relevant_products
        
        Returns:
            Dictionary with aggregate results
        """
        results = []
        for test_case in test_cases:
            result = self.evaluate_query(
                query=test_case["query"],
                retrieved_products=test_case["retrieved_products"],
                relevant_products=test_case["relevant_products"]
            )
            results.append(result)
        
        # Calculate aggregates
        precision_scores = []
        recall_scores = []
        ndcg_scores = []
        
        for r in results:
            if "precision@5" in r["metrics"]:
                precision_scores.append(r["metrics"]["precision@5"]["score"])
            if "recall@10" in r["metrics"]:
                recall_scores.append(r["metrics"]["recall@10"]["score"])
            if "ndcg@10" in r["metrics"]:
                ndcg_scores.append(r["metrics"]["ndcg@10"]["score"])
        
        return {
            "total_cases": len(test_cases),
            "average_precision@5": sum(precision_scores) / len(precision_scores) if precision_scores else 0,
            "average_recall@10": sum(recall_scores) / len(recall_scores) if recall_scores else 0,
            "average_ndcg@10": sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0,
            "pass_rate": sum(1 for r in results if r["all_passed"]) / len(results) if results else 0,
            "results": results
        }

