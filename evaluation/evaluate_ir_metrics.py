"""Evaluate IR metrics (NDCG, Recall@K, Precision@K, Context Precision) on retrieval results."""
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from evaluation.ir_metrics import IRMetrics

# Optional import - RAG retriever may not be available if vector_store is missing
try:
    from src.rag.retriever import retriever
    RAG_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    RAG_AVAILABLE = False
    retriever = None


def evaluate_retrieval_quality(
    query: str,
    retrieved_products: List[Dict[str, Any]],
    relevant_products: Dict[str, float],
    k_values: List[int] = [1, 3, 5, 10],
    context_size: int = 5
) -> Dict[str, Any]:
    """
    Evaluate retrieval quality with IR metrics.
    
    Args:
        query: Search query
        retrieved_products: List of retrieved products from RAG
        relevant_products: Dict mapping product_id to relevance score (0-4)
        k_values: List of K values to evaluate
        context_size: Number of products passed to LLM context
        
    Returns:
        Dictionary with all metrics
    """
    # Extract product IDs from retrieved results
    retrieved_ids = []
    for p in retrieved_products:
        product_id = p.get('id') or p.get('metadata', {}).get('id', '')
        if product_id:
            retrieved_ids.append(product_id)
    
    # Create set of relevant product IDs
    relevant_set = set(relevant_products.keys())
    
    # Calculate all metrics
    metrics = {
        'query': query,
        'num_retrieved': len(retrieved_ids),
        'num_relevant': len(relevant_set),
        'retrieved_ids': retrieved_ids[:10]  # Store top 10 for reference
    }
    
    # Calculate metrics for each K
    for k in k_values:
        # Precision@K
        metrics[f'precision@{k}'] = IRMetrics.precision_at_k(
            retrieved_ids, relevant_set, k
        )
        
        # Recall@K
        metrics[f'recall@{k}'] = IRMetrics.recall_at_k(
            retrieved_ids, relevant_set, k
        )
        
        # NDCG@K
        metrics[f'ndcg@{k}'] = IRMetrics.ndcg_at_k(
            retrieved_ids, relevant_products, k
        )
    
    # Calculate Context Precision (precision of items in LLM context)
    metrics['context_precision'] = IRMetrics.context_precision(
        retrieved_ids, relevant_set, context_size
    )
    
    # Calculate MRR and MAP
    metrics['mrr'] = IRMetrics.reciprocal_rank(retrieved_ids, relevant_set)
    metrics['map'] = IRMetrics.average_precision(retrieved_ids, relevant_set)
    
    return metrics


def run_evaluation(dataset_path: str, context_size: int = 5):
    """Run evaluation on entire dataset."""
    # Load relevance dataset
    if not Path(dataset_path).exists():
        print(f"Error: Dataset file not found: {dataset_path}")
        print("Please create the relevance dataset first.")
        return None
    
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    all_metrics = []
    
    print("="*70)
    print("IR METRICS EVALUATION")
    print("="*70)
    print()
    
    for query_data in dataset['queries']:
        query = query_data['query']
        relevant_products = query_data.get('relevant_products', {})
        
        if not relevant_products:
            print(f"Skipping '{query}' - no relevance labels")
            continue
        
        # Retrieve products using RAG (if available)
        if not RAG_AVAILABLE:
            print(f"Warning: RAG retriever not available. Skipping query: {query}")
            continue
            
        print(f"Query: {query}")
        print(f"Retrieving products...")
        retrieved = retriever.retrieve_products(query, n_results=10)
        
        if not retrieved:
            print(f"  Warning: No products retrieved")
            continue
        
        # Evaluate
        metrics = evaluate_retrieval_quality(
            query=query,
            retrieved_products=retrieved,
            relevant_products=relevant_products,
            k_values=[1, 3, 5, 10],
            context_size=context_size
        )
        
        all_metrics.append(metrics)
        
        # Print per-query results
        print(f"  Retrieved: {metrics['num_retrieved']} products")
        print(f"  Relevant: {metrics['num_relevant']} products")
        print(f"  Precision@1: {metrics['precision@1']:.3f}")
        print(f"  Precision@5: {metrics['precision@5']:.3f}")
        print(f"  Recall@5: {metrics['recall@5']:.3f}")
        print(f"  Recall@10: {metrics['recall@10']:.3f}")
        print(f"  NDCG@5: {metrics['ndcg@5']:.3f}")
        print(f"  NDCG@10: {metrics['ndcg@10']:.3f}")
        print(f"  Context Precision (top {context_size}): {metrics['context_precision']:.3f}")
        print(f"  MRR: {metrics['mrr']:.3f}")
        print(f"  MAP: {metrics['map']:.3f}")
        print()
    
    if not all_metrics:
        print("No metrics calculated. Please check your relevance dataset.")
        return None
    
    # Calculate averages
    print("="*70)
    print("AGGREGATE METRICS (Across All Queries)")
    print("="*70)
    print()
    
    num_queries = len(all_metrics)
    
    # Precision metrics
    avg_precision_1 = sum(m['precision@1'] for m in all_metrics) / num_queries
    avg_precision_3 = sum(m['precision@3'] for m in all_metrics) / num_queries
    avg_precision_5 = sum(m['precision@5'] for m in all_metrics) / num_queries
    avg_precision_10 = sum(m['precision@10'] for m in all_metrics) / num_queries
    
    # Recall metrics
    avg_recall_1 = sum(m['recall@1'] for m in all_metrics) / num_queries
    avg_recall_3 = sum(m['recall@3'] for m in all_metrics) / num_queries
    avg_recall_5 = sum(m['recall@5'] for m in all_metrics) / num_queries
    avg_recall_10 = sum(m['recall@10'] for m in all_metrics) / num_queries
    
    # NDCG metrics
    avg_ndcg_1 = sum(m['ndcg@1'] for m in all_metrics) / num_queries
    avg_ndcg_3 = sum(m['ndcg@3'] for m in all_metrics) / num_queries
    avg_ndcg_5 = sum(m['ndcg@5'] for m in all_metrics) / num_queries
    avg_ndcg_10 = sum(m['ndcg@10'] for m in all_metrics) / num_queries
    
    # Other metrics
    avg_context_precision = sum(m['context_precision'] for m in all_metrics) / num_queries
    avg_mrr = sum(m['mrr'] for m in all_metrics) / num_queries
    avg_map = sum(m['map'] for m in all_metrics) / num_queries
    
    print(f"Number of queries evaluated: {num_queries}")
    print()
    print("Precision Metrics:")
    print(f"  Precision@1:  {avg_precision_1:.3f}")
    print(f"  Precision@3:  {avg_precision_3:.3f}")
    print(f"  Precision@5:  {avg_precision_5:.3f}")
    print(f"  Precision@10: {avg_precision_10:.3f}")
    print()
    print("Recall Metrics:")
    print(f"  Recall@1:  {avg_recall_1:.3f}")
    print(f"  Recall@3:  {avg_recall_3:.3f}")
    print(f"  Recall@5:  {avg_recall_5:.3f}")
    print(f"  Recall@10: {avg_recall_10:.3f}")
    print()
    print("NDCG Metrics:")
    print(f"  NDCG@1:  {avg_ndcg_1:.3f}")
    print(f"  NDCG@3:  {avg_ndcg_3:.3f}")
    print(f"  NDCG@5:  {avg_ndcg_5:.3f}")
    print(f"  NDCG@10: {avg_ndcg_10:.3f}")
    print()
    print("Other Metrics:")
    print(f"  Context Precision (top {context_size}): {avg_context_precision:.3f}")
    print(f"  MRR (Mean Reciprocal Rank): {avg_mrr:.3f}")
    print(f"  MAP (Mean Average Precision): {avg_map:.3f}")
    print()
    
    # Performance assessment
    print("="*70)
    print("PERFORMANCE ASSESSMENT")
    print("="*70)
    print()
    
    assessments = []
    
    if avg_precision_5 >= 0.65:
        assessments.append("✓ Precision@5 is GOOD (>= 0.65)")
    else:
        assessments.append(f"✗ Precision@5 needs improvement (current: {avg_precision_5:.3f}, target: >= 0.65)")
    
    if avg_recall_10 >= 0.60:
        assessments.append("✓ Recall@10 is GOOD (>= 0.60)")
    else:
        assessments.append(f"✗ Recall@10 needs improvement (current: {avg_recall_10:.3f}, target: >= 0.60)")
    
    if avg_ndcg_10 >= 0.75:
        assessments.append("✓ NDCG@10 is GOOD (>= 0.75)")
    else:
        assessments.append(f"✗ NDCG@10 needs improvement (current: {avg_ndcg_10:.3f}, target: >= 0.75)")
    
    if avg_mrr >= 0.70:
        assessments.append("✓ MRR is GOOD (>= 0.70)")
    else:
        assessments.append(f"✗ MRR needs improvement (current: {avg_mrr:.3f}, target: >= 0.70)")
    
    if avg_context_precision >= 0.60:
        assessments.append(f"✓ Context Precision is GOOD (>= 0.60)")
    else:
        assessments.append(f"✗ Context Precision needs improvement (current: {avg_context_precision:.3f}, target: >= 0.60)")
    
    for assessment in assessments:
        print(f"  {assessment}")
    
    print()
    
    # Save detailed results
    results_file = f"evaluation/ir_metrics_results_{Path(dataset_path).stem}.json"
    with open(results_file, 'w') as f:
        json.dump({
            'summary': {
                'num_queries': num_queries,
                'avg_precision@5': avg_precision_5,
                'avg_recall@10': avg_recall_10,
                'avg_ndcg@10': avg_ndcg_10,
                'avg_mrr': avg_mrr,
                'avg_map': avg_map,
                'avg_context_precision': avg_context_precision
            },
            'per_query_metrics': all_metrics
        }, f, indent=2)
    
    print(f"Detailed results saved to: {results_file}")
    print()
    
    return all_metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate IR metrics on retrieval results')
    parser.add_argument('--dataset', type=str, default='evaluation/relevance_dataset.json',
                        help='Path to relevance dataset JSON file')
    parser.add_argument('--context-size', type=int, default=5,
                        help='Number of products passed to LLM context')
    
    args = parser.parse_args()
    
    run_evaluation(args.dataset, args.context_size)


