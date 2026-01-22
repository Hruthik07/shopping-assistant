"""Hybrid retrieval system for RAG."""
from typing import List, Dict, Any, Optional
from src.rag.vector_store import vector_store
from src.rag.embeddings import embedding_generator
from src.rag.document_loader import document_loader
from src.utils.helpers import calculate_similarity_score
from src.analytics.logger import logger
from src.analytics.tracker import tracker
from src.analytics.latency_tracker import latency_tracker


class HybridRetriever:
    """Hybrid retrieval combining semantic and keyword search."""
    
    def __init__(self):
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        self.document_loader = document_loader
    
    def retrieve_products(
        self,
        query: str,
        n_results: int = 5,
        category_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve products using hybrid search with latency tracking."""
        # Generate query embedding
        with latency_tracker.track_component("embedding_generation"):
            query_embedding = self.embedding_generator.embed_text(query)
        
        # Semantic search
        with latency_tracker.track_component("vector_search"):
            semantic_results = self.vector_store.search_products(
                query_embedding,
                n_results=n_results * 2,  # Get more for re-ranking
                category_filter=category_filter
            )
        
        # Keyword-based re-ranking
        with latency_tracker.track_component("reranking"):
            scored_results = []
            for result in semantic_results:
                # Combine semantic distance with keyword similarity
                keyword_score = calculate_similarity_score(
                    query,
                    result["document"]
                )
                semantic_score = 1 - (result["distance"] or 0)  # Convert distance to similarity
                
                # Weighted combination (70% semantic, 30% keyword)
                combined_score = 0.7 * semantic_score + 0.3 * keyword_score
                
                scored_results.append({
                    **result,
                    "score": combined_score
                })
            
            # Sort by combined score and return top N
            scored_results.sort(key=lambda x: x["score"], reverse=True)
            final_results = scored_results[:n_results]
        
        tracker.track_rag_retrieval(query, len(final_results))
        logger.debug(f"Retrieved {len(final_results)} products for query: {query[:50]}")
        
        return final_results
    
    def retrieve_reviews(
        self,
        query: str,
        product_id: Optional[str] = None,
        n_results: int = 3
    ) -> List[Dict[str, Any]]:
        """Retrieve reviews using semantic search."""
        query_embedding = self.embedding_generator.embed_text(query)
        
        results = self.vector_store.search_reviews(
            query_embedding,
            product_id=product_id,
            n_results=n_results
        )
        
        return results
    
    def retrieve_faqs(
        self,
        query: str,
        product_id: Optional[str] = None,
        n_results: int = 3
    ) -> List[Dict[str, Any]]:
        """Retrieve FAQs using semantic search."""
        query_embedding = self.embedding_generator.embed_text(query)
        
        results = self.vector_store.search_faqs(
            query_embedding,
            product_id=product_id,
            n_results=n_results
        )
        
        return results
    
    async def get_product_context(
        self,
        product_id: str,
        include_reviews: bool = True,
        include_faqs: bool = True
    ) -> Dict[str, Any]:
        """Get comprehensive context for a product."""
        product = await self.document_loader.get_product_by_id(product_id)
        
        # If product not found locally, try to fetch from API
        if not product and product_id.startswith("google_"):
            # This is a product from API, we'd need to fetch it
            pass
        
        context = {
            "product": product,
            "reviews": [],
            "faqs": []
        }
        
        if include_reviews and product:
            reviews = self.retrieve_reviews(
                query=product.get("name", ""),
                product_id=product_id,
                n_results=3
            )
            context["reviews"] = reviews
        
        if include_faqs and product:
            faqs = self.retrieve_faqs(
                query=product.get("name", ""),
                product_id=product_id,
                n_results=3
            )
            context["faqs"] = faqs
        
        return context


# Global retriever instance
retriever = HybridRetriever()

