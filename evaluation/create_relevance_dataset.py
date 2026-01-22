"""Helper script to create relevance dataset from recent queries."""
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rag.retriever import retriever


def extract_products_for_labeling(query: str, n_results: int = 10) -> List[Dict[str, Any]]:
    """Extract products for a query to help with labeling."""
    retrieved = retriever.retrieve_products(query, n_results=n_results)
    
    products = []
    for i, product in enumerate(retrieved, 1):
        product_id = product.get('id') or product.get('metadata', {}).get('id', '')
        product_name = product.get('metadata', {}).get('name', '') or product.get('document', '')[:50]
        
        products.append({
            'rank': i,
            'product_id': product_id,
            'name': product_name,
            'description': product.get('document', '')[:200],
            'category': product.get('metadata', {}).get('category', ''),
            'price': product.get('metadata', {}).get('price', ''),
        })
    
    return products


def create_template_dataset():
    """Create a template relevance dataset based on recent queries."""
    
    # Queries from recent evaluation
    queries = [
        "Find me wireless headphones",
        "What's the price of laptops with 16GB RAM?",
        "Show me top 5 AI Engineering books",
        "Find face cream for dark spots under $25",
        "Search for gaming laptops with NVIDIA GPU",
        "Tell me about the best selling products"
    ]
    
    dataset = {
        "description": "Relevance dataset for IR metrics evaluation",
        "relevance_scale": {
            "0": "Not relevant",
            "1": "Somewhat relevant",
            "2": "Relevant",
            "3": "Highly relevant",
            "4": "Perfect match"
        },
        "instructions": [
            "1. For each query, retrieve products using the system",
            "2. Manually label each product with relevance score (0-4)",
            "3. Update the 'relevant_products' dict with product_id: relevance_score",
            "4. Save the file and run evaluation/evaluate_ir_metrics.py"
        ],
        "queries": []
    }
    
    print("="*70)
    print("CREATING RELEVANCE DATASET TEMPLATE")
    print("="*70)
    print()
    print("This script will:")
    print("1. Retrieve products for each query")
    print("2. Show you the products to help with labeling")
    print("3. Create a template dataset file")
    print()
    
    for i, query in enumerate(queries, 1):
        print(f"[{i}/{len(queries)}] Processing: {query}")
        
        # Extract products
        products = extract_products_for_labeling(query, n_results=10)
        
        if not products:
            print(f"  Warning: No products found for this query")
            continue
        
        # Create query entry
        query_entry = {
            "query_id": f"Q{i:03d}",
            "query": query,
            "retrieved_products": products,
            "relevant_products": {}  # To be filled manually
        }
        
        dataset["queries"].append(query_entry)
        
        # Show products for reference
        print(f"  Found {len(products)} products:")
        for product in products[:5]:  # Show top 5
            print(f"    {product['rank']}. {product['name'][:50]}")
            print(f"       ID: {product['product_id']}")
        if len(products) > 5:
            print(f"    ... and {len(products) - 5} more")
        print()
    
    # Save template
    output_file = "evaluation/relevance_dataset.json"
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print("="*70)
    print(f"Template dataset created: {output_file}")
    print("="*70)
    print()
    print("NEXT STEPS:")
    print("1. Open the file and review the retrieved products")
    print("2. For each query, label products with relevance scores:")
    print("   - 0 = Not relevant")
    print("   - 1 = Somewhat relevant")
    print("   - 2 = Relevant")
    print("   - 3 = Highly relevant")
    print("   - 4 = Perfect match")
    print("3. Update 'relevant_products' dict: {\"product_id\": relevance_score}")
    print("4. Run: python evaluation/evaluate_ir_metrics.py")
    print()
    
    return dataset


if __name__ == "__main__":
    create_template_dataset()


