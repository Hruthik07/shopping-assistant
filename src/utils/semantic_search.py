"""Semantic search utilities for product retrieval."""

from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from src.analytics.logger import logger
from src.utils.cache import cache_service
from src.utils.config import settings
import numpy as np


class SemanticSearcher:
    """Semantic search for products using embeddings."""

    def __init__(self):
        """Initialize semantic searcher with embedding model."""
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load embedding model."""
        try:
            logger.info(f"Loading semantic search model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Semantic search model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading semantic search model: {e}")
            raise

    async def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a query."""
        try:
            # Check cache first
            cache_key = f"semantic:embedding:{query}"
            cached_embedding = await cache_service.get(cache_key)
            if cached_embedding:
                return cached_embedding

            # Generate embedding
            embedding = self.model.encode(query, convert_to_numpy=True).tolist()

            # Cache the embedding (24 hours)
            await cache_service.set(cache_key, embedding, ttl=86400)

            return embedding
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            raise

    def embed_product(self, product: Dict[str, Any]) -> List[float]:
        """Generate embedding for a product."""
        # Create product text representation
        product_text = self._product_to_text(product)

        # Generate embedding
        embedding = self.model.encode(product_text, convert_to_numpy=True).tolist()
        return embedding

    def _product_to_text(self, product: Dict[str, Any]) -> str:
        """Convert product dict to searchable text."""
        parts = []

        # Product name (most important)
        if product.get("name"):
            parts.append(product["name"])

        # Description
        if product.get("description"):
            parts.append(product["description"])

        # Category
        if product.get("category"):
            parts.append(f"Category: {product['category']}")

        # Brand (if extractable from name)
        name = product.get("name", "")
        if name:
            # Try to extract brand (first word or two, if capitalized)
            words = name.split()
            if len(words) > 1 and words[0][0].isupper():
                parts.append(f"Brand: {words[0]}")

        return " ".join(parts)

    def calculate_similarity(
        self, query_embedding: List[float], product_embedding: List[float]
    ) -> float:
        """Calculate cosine similarity between query and product embeddings."""
        try:
            query_vec = np.array(query_embedding)
            product_vec = np.array(product_embedding)

            # Cosine similarity
            dot_product = np.dot(query_vec, product_vec)
            norm_query = np.linalg.norm(query_vec)
            norm_product = np.linalg.norm(product_vec)

            if norm_query == 0 or norm_product == 0:
                return 0.0

            similarity = dot_product / (norm_query * norm_product)
            return float(similarity)
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0

    async def rerank_products(
        self, query: str, products: List[Dict[str, Any]], top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Re-rank products using semantic similarity to query."""
        if not products:
            return []

        try:
            # Generate query embedding
            query_embedding = await self.embed_query(query)

            # Calculate similarity scores for all products
            scored_products = []
            for product in products:
                # Generate product embedding
                product_embedding = self.embed_product(product)

                # Calculate similarity
                similarity_score = self.calculate_similarity(query_embedding, product_embedding)

                scored_products.append({**product, "_semantic_score": similarity_score})

            # Sort by similarity score (descending)
            scored_products.sort(key=lambda x: x.get("_semantic_score", 0), reverse=True)

            # Keep semantic score when semantic-only retrieval is enabled (useful for grounding/explanations).
            # Otherwise remove the internal score field before returning to keep payloads clean.
            if not getattr(settings, "semantic_only_retrieval", False):
                for product in scored_products:
                    product.pop("_semantic_score", None)

            # Return top K if specified
            if top_k:
                return scored_products[:top_k]

            return scored_products

        except Exception as e:
            logger.error(f"Error in semantic re-ranking: {e}")
            # Return original products if semantic search fails
            return products

    async def search_products(
        self, query: str, products: List[Dict[str, Any]], top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Semantic search and re-rank products."""
        return await self.rerank_products(query, products, top_k=top_k)


# Global semantic searcher instance (lazy initialization)
_semantic_searcher_instance = None


def get_semantic_searcher() -> SemanticSearcher:
    """Get or create semantic searcher instance."""
    global _semantic_searcher_instance
    if _semantic_searcher_instance is None:
        try:
            _semantic_searcher_instance = SemanticSearcher()
        except Exception as e:
            logger.error(f"Failed to initialize semantic searcher: {e}")
            raise
    return _semantic_searcher_instance
