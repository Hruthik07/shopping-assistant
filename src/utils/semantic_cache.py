"""Semantic caching for LLM responses using embeddings."""
import json
import hashlib
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from src.utils.cache import cache_service
from src.analytics.logger import logger


class SemanticCache:
    """Semantic cache that finds similar queries using embeddings."""
    
    def __init__(self, similarity_threshold: float = 0.85):
        """Initialize semantic cache.
        
        Args:
            similarity_threshold: Minimum cosine similarity to consider queries similar (0.0-1.0)
        """
        self.similarity_threshold = similarity_threshold
        self.embedding_generator = None
        self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """Initialize embedding generator if available."""
        try:
            from src.rag.embeddings import embedding_generator
            self.embedding_generator = embedding_generator
            logger.info(f"Semantic cache initialized with threshold: {self.similarity_threshold}")
        except Exception as e:
            logger.warning(f"Could not initialize embeddings for semantic cache: {e}")
            self.embedding_generator = None
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            v1 = np.array(vec1)
            v2 = np.array(vec2)
            dot_product = np.dot(v1, v2)
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return float(dot_product / (norm1 * norm2))
        except Exception as e:
            logger.debug(f"Error calculating cosine similarity: {e}")
            return 0.0
    
    async def _get_query_embedding(self, query: str) -> Optional[List[float]]:
        """Get embedding for a query."""
        if not self.embedding_generator:
            return None
        
        try:
            if hasattr(self.embedding_generator, 'embed_text'):
                embedding = await self.embedding_generator.embed_text(query)
            elif hasattr(self.embedding_generator, 'model'):
                # SentenceTransformer
                embedding = self.embedding_generator.model.encode(query).tolist()
            else:
                return None
            return embedding
        except Exception as e:
            logger.debug(f"Error generating query embedding: {e}")
            return None
    
    async def _find_similar_cached_queries(
        self,
        query_embedding: List[float],
        max_results: int = 5
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Find cached queries with similar embeddings.
        
        Args:
            query_embedding: Embedding vector of the query
            max_results: Maximum number of similar queries to return
            
        Returns:
            List of tuples: (cache_key, similarity_score, cached_data)
        """
        if not cache_service.enabled or not cache_service.redis_client:
            return []
        
        try:
            # Get all LLM response cache keys
            pattern = "llm:response:*"
            similar_queries = []
            
            async for key in cache_service.redis_client.scan_iter(match=pattern):
                try:
                    # Get cached value
                    cached_value = await cache_service.get(key)
                    if not cached_value:
                        continue
                    
                    # Check if it has embedding stored
                    cached_embedding = cached_value.get("_embedding")
                    if not cached_embedding:
                        continue
                    
                    # Calculate similarity
                    similarity = self._cosine_similarity(query_embedding, cached_embedding)
                    
                    if similarity >= self.similarity_threshold:
                        similar_queries.append((key, similarity, cached_value))
                except Exception as e:
                    logger.debug(f"Error checking cache key {key}: {e}")
                    continue
            
            # Sort by similarity (highest first) and return top results
            similar_queries.sort(key=lambda x: x[1], reverse=True)
            return similar_queries[:max_results]
            
        except Exception as e:
            logger.warning(f"Error finding similar cached queries: {e}")
            return []
    
    async def get_semantic(
        self,
        query: str,
        context_hash: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get cached response using semantic similarity.
        
        Args:
            query: User query
            context_hash: Optional context hash for exact matching first
            
        Returns:
            Cached response if found, None otherwise
        """
        # First try exact match (faster)
        if context_hash:
            exact_key = f"llm:response:{context_hash}"
            exact_match = await cache_service.get(exact_key)
            if exact_match:
                logger.debug("Exact cache hit (semantic cache)")
                return exact_match
        
        # If no exact match, try semantic similarity
        if not self.embedding_generator:
            return None
        
        try:
            # Get query embedding
            query_embedding = await self._get_query_embedding(query)
            if not query_embedding:
                return None
            
            # Find similar cached queries
            similar_queries = await self._find_similar_cached_queries(query_embedding, max_results=1)
            
            if similar_queries:
                cache_key, similarity, cached_data = similar_queries[0]
                logger.info(
                    f"Semantic cache hit: similarity={similarity:.3f} "
                    f"(threshold={self.similarity_threshold})"
                )
                # Return cached response (without embedding metadata)
                result = {k: v for k, v in cached_data.items() if k != "_embedding"}
                result["_semantic_similarity"] = similarity
                result["_cache_key"] = cache_key
                return result
            
            return None
            
        except Exception as e:
            logger.warning(f"Error in semantic cache get: {e}")
            return None
    
    async def set_semantic(
        self,
        query: str,
        response_data: Dict[str, Any],
        context_hash: Optional[str] = None,
        ttl: int = 3600
    ) -> bool:
        """Store response in cache with semantic indexing.
        
        Args:
            query: User query
            response_data: Response data to cache
            context_hash: Optional context hash for exact matching
            ttl: Time to live in seconds
            
        Returns:
            True if successful, False otherwise
        """
        # Store with exact match key if context_hash provided
        if context_hash:
            exact_key = f"llm:response:{context_hash}"
            await cache_service.set(exact_key, response_data, ttl=ttl)
        
        # Also store with semantic indexing
        if not self.embedding_generator:
            return True
        
        try:
            # Get query embedding
            query_embedding = await self._get_query_embedding(query)
            if not query_embedding:
                return True  # Still succeed even if embedding fails
            
            # Add embedding to cached data
            response_data_with_embedding = response_data.copy()
            response_data_with_embedding["_embedding"] = query_embedding
            response_data_with_embedding["_query"] = query[:200]  # Store query for debugging
            
            # Generate semantic cache key (hash of embedding)
            embedding_hash = hashlib.sha256(
                json.dumps(query_embedding, sort_keys=True).encode()
            ).hexdigest()[:16]
            
            semantic_key = f"llm:response:semantic:{embedding_hash}"
            await cache_service.set(semantic_key, response_data_with_embedding, ttl=ttl)
            
            logger.debug(f"Stored in semantic cache: {semantic_key}")
            return True
            
        except Exception as e:
            logger.warning(f"Error in semantic cache set: {e}")
            return True  # Don't fail the request if semantic caching fails


# Global semantic cache instance
semantic_cache = SemanticCache(similarity_threshold=0.85)
