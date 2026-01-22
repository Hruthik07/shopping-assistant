"""Information Retrieval metrics for product search evaluation."""
import math
from typing import List, Dict, Any, Set
import numpy as np


class IRMetrics:
    """Calculate IR metrics for retrieval evaluation."""
    
    @staticmethod
    def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
        """
        Calculate Precision@K.
        
        Args:
            retrieved: List of retrieved product IDs (in order)
            relevant: Set of relevant product IDs
            k: Number of top results to consider
            
        Returns:
            Precision@K score (0-1)
        """
        if k == 0:
            return 0.0
        
        top_k = retrieved[:k]
        relevant_in_top_k = sum(1 for item in top_k if item in relevant)
        return relevant_in_top_k / k
    
    @staticmethod
    def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
        """
        Calculate Recall@K.
        
        Args:
            retrieved: List of retrieved product IDs (in order)
            relevant: Set of relevant product IDs
            k: Number of top results to consider
            
        Returns:
            Recall@K score (0-1)
        """
        if len(relevant) == 0:
            return 0.0
        
        top_k = retrieved[:k]
        relevant_in_top_k = sum(1 for item in top_k if item in relevant)
        return relevant_in_top_k / len(relevant)
    
    @staticmethod
    def dcg_at_k(relevance_scores: List[float], k: int) -> float:
        """
        Calculate Discounted Cumulative Gain@K.
        
        Args:
            relevance_scores: List of relevance scores for retrieved items
            k: Number of top results to consider
            
        Returns:
            DCG@K score
        """
        scores = relevance_scores[:k]
        dcg = 0.0
        for i, score in enumerate(scores, start=1):
            if score > 0:  # Only count non-zero relevance
                dcg += score / math.log2(i + 1)
        return dcg
    
    @staticmethod
    def ndcg_at_k(
        retrieved: List[str],
        relevance_dict: Dict[str, float],
        k: int
    ) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain@K.
        
        Args:
            retrieved: List of retrieved product IDs (in order)
            relevance_dict: Dict mapping product_id to relevance score
            k: Number of top results to consider
            
        Returns:
            NDCG@K score (0-1)
        """
        # Get relevance scores for retrieved items
        relevance_scores = [
            relevance_dict.get(item, 0.0) for item in retrieved[:k]
        ]
        
        # Calculate DCG
        dcg = IRMetrics.dcg_at_k(relevance_scores, k)
        
        # Calculate Ideal DCG (sorted by relevance)
        ideal_scores = sorted(relevance_dict.values(), reverse=True)[:k]
        idcg = IRMetrics.dcg_at_k(ideal_scores, k)
        
        # Avoid division by zero
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    @staticmethod
    def reciprocal_rank(retrieved: List[str], relevant: Set[str]) -> float:
        """
        Calculate Reciprocal Rank.
        
        Args:
            retrieved: List of retrieved product IDs (in order)
            relevant: Set of relevant product IDs
            
        Returns:
            Reciprocal rank (0 if no relevant item found)
        """
        for i, item in enumerate(retrieved, start=1):
            if item in relevant:
                return 1.0 / i
        return 0.0
    
    @staticmethod
    def mean_reciprocal_rank(
        query_results: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate Mean Reciprocal Rank across queries.
        
        Args:
            query_results: List of dicts with 'retrieved' and 'relevant' keys
            
        Returns:
            MRR score (0-1)
        """
        rr_scores = []
        for result in query_results:
            rr = IRMetrics.reciprocal_rank(
                result['retrieved'],
                result['relevant']
            )
            rr_scores.append(rr)
        
        return np.mean(rr_scores) if rr_scores else 0.0
    
    @staticmethod
    def average_precision(
        retrieved: List[str],
        relevant: Set[str]
    ) -> float:
        """
        Calculate Average Precision.
        
        Args:
            retrieved: List of retrieved product IDs (in order)
            relevant: Set of relevant product IDs
            
        Returns:
            Average Precision score (0-1)
        """
        if len(relevant) == 0:
            return 0.0
        
        relevant_count = 0
        precision_sum = 0.0
        
        for i, item in enumerate(retrieved, start=1):
            if item in relevant:
                relevant_count += 1
                precision_at_i = relevant_count / i
                precision_sum += precision_at_i
        
        return precision_sum / len(relevant)
    
    @staticmethod
    def mean_average_precision(
        query_results: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate Mean Average Precision across queries.
        
        Args:
            query_results: List of dicts with 'retrieved' and 'relevant' keys
            
        Returns:
            MAP score (0-1)
        """
        ap_scores = []
        for result in query_results:
            ap = IRMetrics.average_precision(
                result['retrieved'],
                result['relevant']
            )
            ap_scores.append(ap)
        
        return np.mean(ap_scores) if ap_scores else 0.0
    
    @staticmethod
    def evaluate_retrieval(
        query: str,
        retrieved_products: List[Dict[str, Any]],
        relevant_products: Dict[str, float],
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation of retrieval results.
        
        Args:
            query: Search query
            retrieved_products: List of retrieved products with 'id' field
            relevant_products: Dict mapping product_id to relevance score
            k_values: List of K values to evaluate
            
        Returns:
            Dictionary with all metrics
        """
        # Input validation
        if not query or not isinstance(query, str):
            raise ValueError("query must be a non-empty string")
        
        if not isinstance(retrieved_products, list):
            raise ValueError("retrieved_products must be a list")
        
        if not isinstance(relevant_products, dict):
            raise ValueError("relevant_products must be a dictionary")
        
        if not isinstance(k_values, list) or not k_values:
            raise ValueError("k_values must be a non-empty list of integers")
        
        if not all(isinstance(k, int) and k > 0 for k in k_values):
            raise ValueError("All k_values must be positive integers")
        
        # Extract product IDs from retrieved results
        retrieved_ids = []
        for p in retrieved_products:
            if not isinstance(p, dict):
                continue
            product_id = p.get('id') or p.get('metadata', {}).get('id', '')
            if product_id:
                retrieved_ids.append(str(product_id))  # Ensure string type
        
        # Create set of relevant product IDs
        relevant_set = set(relevant_products.keys())
        
        # Calculate metrics for each K
        metrics = {
            'query': query,
            'num_retrieved': len(retrieved_ids),
            'num_relevant': len(relevant_set)
        }
        
        for k in k_values:
            metrics[f'precision@{k}'] = IRMetrics.precision_at_k(
                retrieved_ids, relevant_set, k
            )
            metrics[f'recall@{k}'] = IRMetrics.recall_at_k(
                retrieved_ids, relevant_set, k
            )
            metrics[f'ndcg@{k}'] = IRMetrics.ndcg_at_k(
                retrieved_ids, relevant_products, k
            )
        
        # Calculate MRR and MAP
        metrics['mrr'] = IRMetrics.reciprocal_rank(retrieved_ids, relevant_set)
        metrics['map'] = IRMetrics.average_precision(retrieved_ids, relevant_set)
        
        # Calculate Context Precision (precision of items in LLM context, typically top 5)
        metrics['context_precision'] = IRMetrics.context_precision(
            retrieved_ids, relevant_set, context_size=5
        )
        
        return metrics
    
    @staticmethod
    def context_precision(
        retrieved: List[str],
        relevant: Set[str],
        context_size: int
    ) -> float:
        """
        Calculate Context Precision - precision of items in context passed to LLM.
        
        Args:
            retrieved: List of retrieved product IDs (in order)
            relevant: Set of relevant product IDs
            context_size: Number of items passed to LLM context
            
        Returns:
            Context precision (0-1)
        """
        if context_size == 0:
            return 0.0
        
        context_items = retrieved[:context_size]
        if len(context_items) == 0:
            return 0.0
        
        relevant_in_context = sum(1 for item in context_items if item in relevant)
        return relevant_in_context / len(context_items)

