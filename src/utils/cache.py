"""Redis-based caching service for the shopping assistant."""
import json
import hashlib
import time
import json as json_module
from typing import Any, Optional, Dict, List
from datetime import timedelta
import redis.asyncio as aioredis
from src.utils.config import settings
from src.analytics.logger import logger

class CacheService:
    """Redis-based caching service with multiple cache layers."""
    
    def __init__(self):
        self.redis_client: Optional[aioredis.Redis] = None
        self.enabled = True
        self._connection_pool: Optional[aioredis.ConnectionPool] = None
        # Cache statistics for monitoring
        self._stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "errors": 0
        }
    
    async def connect(self):
        """Initialize Redis connection."""
        try:
            redis_url = getattr(settings, 'redis_url', 'redis://localhost:6379/0')
            self._connection_pool = aioredis.ConnectionPool.from_url(
                redis_url,
                max_connections=50,
                decode_responses=True
            )
            self.redis_client = aioredis.Redis(connection_pool=self._connection_pool)
            
            # Test connection
            await self.redis_client.ping()
            logger.info("Redis cache connected successfully")
            self.enabled = True
        except Exception as e:
            logger.warning(f"Redis not available, caching disabled: {e}")
            self.enabled = False
            self.redis_client = None
    
    async def disconnect(self):
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()
            if self._connection_pool:
                await self._connection_pool.disconnect()
            logger.info("Redis cache disconnected")
    
    def _generate_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate a cache key from prefix and arguments."""
        # Create a hash of all arguments
        key_parts = [prefix]
        if args:
            key_parts.extend(str(arg) for arg in args)
        if kwargs:
            # Sort kwargs for consistent hashing
            sorted_kwargs = sorted(kwargs.items())
            key_parts.extend(f"{k}:{v}" for k, v in sorted_kwargs)
        
        key_string = ":".join(key_parts)
        # If key is too long, hash it
        if len(key_string) > 250:
            key_hash = hashlib.sha256(key_string.encode()).hexdigest()
            final_key = f"{prefix}:{key_hash}"
            return final_key
        return key_string
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if not self.enabled or not self.redis_client:
            self._stats["misses"] += 1
            return None
        
        try:
            value = await self.redis_client.get(key)
            if value:
                try:
                    parsed = json.loads(value)
                    self._stats["hits"] += 1
                    return parsed
                except json.JSONDecodeError as je:
                    logger.warning(f"Cache get JSON decode error for key {key}: {je}")
                    self._stats["misses"] += 1
                    self._stats["errors"] += 1
                    return None
            self._stats["misses"] += 1
            return None
        except Exception as e:
            logger.warning(f"Cache get error for key {key}: {e}")
            self._stats["misses"] += 1
            self._stats["errors"] += 1
            return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """Set value in cache with optional TTL (in seconds)."""
        if not self.enabled or not self.redis_client:
            return False
        
        try:
            serialized = json.dumps(value, default=str)
            if ttl:
                await self.redis_client.setex(key, ttl, serialized)
            else:
                await self.redis_client.set(key, serialized)
            self._stats["sets"] += 1
            return True
        except Exception as e:
            logger.warning(f"Cache set error for key {key}: {e}")
            self._stats["errors"] += 1
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete a key from cache."""
        if not self.enabled or not self.redis_client:
            return False
        
        try:
            await self.redis_client.delete(key)
            return True
        except Exception as e:
            logger.warning(f"Cache delete error for key {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        if not self.enabled or not self.redis_client:
            return False
        
        try:
            return await self.redis_client.exists(key) > 0
        except Exception as e:
            logger.warning(f"Cache exists error for key {key}: {e}")
            return False
    
    # Specialized cache methods for different use cases
    
    async def get_llm_response(
        self,
        query: str,
        context_hash: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get cached LLM response."""
        key = self._generate_key("llm:response", query, context=context_hash)
        return await self.get(key)
    
    async def set_llm_response(
        self,
        query: str,
        response: Dict[str, Any],
        context_hash: Optional[str] = None,
        ttl: int = 3600  # 1 hour default
    ) -> bool:
        """Cache LLM response."""
        key = self._generate_key("llm:response", query, context=context_hash)
        return await self.set(key, response, ttl=ttl)
    
    async def get_product_search(
        self,
        query: str,
        category: Optional[str] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        max_results: int = 10
    ) -> Optional[List[Dict[str, Any]]]:
        """Get cached product search results."""
        key = self._generate_key(
            "product:search",
            query,
            category=category,
            min_price=min_price,
            max_price=max_price,
            max_results=max_results
        )
        return await self.get(key)
    
    async def set_product_search(
        self,
        query: str,
        products: List[Dict[str, Any]],
        category: Optional[str] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        max_results: int = 10,
        ttl: int = 1800  # 30 minutes default (products change frequently)
    ) -> bool:
        """Cache product search results."""
        key = self._generate_key(
            "product:search",
            query,
            category=category,
            min_price=min_price,
            max_price=max_price,
            max_results=max_results
        )
        return await self.set(key, products, ttl=ttl)

    async def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get cached embedding vector for text."""
        key = self._generate_key("semantic:embedding", text)
        cached = await self.get(key)
        if cached is None:
            return None
        # Embeddings are expected to be a list of floats
        if isinstance(cached, list):
            return cached
        return None

    async def set_embedding(
        self,
        text: str,
        embedding: List[float],
        ttl: int = 86400,  # 24h default
    ) -> bool:
        """Cache embedding vector for text."""
        key = self._generate_key("semantic:embedding", text)
        return await self.set(key, embedding, ttl=ttl)
    
    async def get_session_data(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get cached session data."""
        key = f"session:{session_id}"
        return await self.get(key)
    
    async def set_session_data(
        self,
        session_id: str,
        data: Dict[str, Any],
        ttl: int = 3600  # 1 hour default
    ) -> bool:
        """Cache session data."""
        key = f"session:{session_id}"
        return await self.set(key, data, ttl=ttl)
    
    async def get_conversation_history(
        self,
        session_id: str,
        limit: int = 10
    ) -> Optional[List[Dict[str, Any]]]:
        """Get cached conversation history."""
        key = f"conversation:{session_id}:{limit}"
        return await self.get(key)
    
    async def set_conversation_history(
        self,
        session_id: str,
        history: List[Dict[str, Any]],
        limit: int = 10,
        ttl: int = 3600  # 1 hour default
    ) -> bool:
        """Cache conversation history."""
        key = f"conversation:{session_id}:{limit}"
        return await self.set(key, history, ttl=ttl)
    
    async def invalidate_product_search(self, query: Optional[str] = None):
        """Invalidate product search cache (all or for specific query)."""
        if not self.enabled or not self.redis_client:
            return
        
        try:
            if query:
                # Invalidate specific query pattern
                pattern = f"product:search:{query}*"
            else:
                # Invalidate all product searches
                pattern = "product:search:*"
            
            keys = []
            async for key in self.redis_client.scan_iter(match=pattern):
                keys.append(key)
            
            if keys:
                await self.redis_client.delete(*keys)
                logger.info(f"Invalidated {len(keys)} product search cache entries")
        except Exception as e:
            logger.warning(f"Error invalidating product search cache: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get internal cache statistics (hits, misses, hit rate)."""
        total_requests = self._stats["hits"] + self._stats["misses"]
        hit_rate = (self._stats["hits"] / total_requests * 100) if total_requests > 0 else 0.0
        
        stats = {
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "sets": self._stats["sets"],
            "errors": self._stats["errors"],
            "total_requests": total_requests,
            "hit_rate": round(hit_rate, 2)
        }
        
        # Export to CloudWatch (async, non-blocking)
        try:
            from src.analytics.cloudwatch_exporter import cloudwatch_exporter
            import asyncio
            # Schedule async export (fire and forget) - safer event loop handling
            try:
                loop = asyncio.get_running_loop()
                asyncio.create_task(cloudwatch_exporter.export_cache_metrics(stats))
            except RuntimeError:
                # No running event loop, try to get existing loop
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.create_task(cloudwatch_exporter.export_cache_metrics(stats))
                    else:
                        asyncio.run(cloudwatch_exporter.export_cache_metrics(stats))
                except Exception:
                    pass  # Skip export if event loop handling fails
        except Exception:
            pass  # Graceful degradation if CloudWatch unavailable
        
        return stats
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics (Redis + internal)."""
        internal_stats = self.get_cache_stats()
        
        if not self.enabled or not self.redis_client:
            return {"enabled": False, **internal_stats}
        
        try:
            info = await self.redis_client.info("stats")
            redis_hits = info.get("keyspace_hits", 0)
            redis_misses = info.get("keyspace_misses", 0)
            redis_total = redis_hits + redis_misses
            redis_hit_rate = (redis_hits / redis_total * 100) if redis_total > 0 else 0.0
            
            return {
                "enabled": True,
                "redis": {
                    "keyspace_hits": redis_hits,
                    "keyspace_misses": redis_misses,
                    "total_keys": await self.redis_client.dbsize(),
                    "hit_rate": round(redis_hit_rate, 2)
                },
                "internal": internal_stats
            }
        except Exception as e:
            logger.warning(f"Error getting cache stats: {e}")
            return {"enabled": True, "error": str(e), **internal_stats}


# Global cache service instance
cache_service = CacheService()

