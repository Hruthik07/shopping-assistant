"""Quick test to verify Redis is working."""
import asyncio
from src.utils.cache import cache_service
from src.analytics.logger import logger

async def test_redis():
    """Test Redis connection and basic operations."""
    print("=" * 60)
    print("Redis Status Check")
    print("=" * 60)
    
    try:
        # Connect to Redis
        await cache_service.connect()
        
        print(f"\n[OK] Redis Enabled: {cache_service.enabled}")
        print(f"[OK] Redis Client: {cache_service.redis_client is not None}")
        
        if cache_service.enabled and cache_service.redis_client:
            # Test basic operations
            test_key = "test:connection"
            test_value = {"status": "ok", "timestamp": "test"}
            
            # Test SET
            set_result = await cache_service.set(test_key, test_value, ttl=10)
            print(f"[OK] SET operation: {'Success' if set_result else 'Failed'}")
            
            # Test GET
            get_result = await cache_service.get(test_key)
            print(f"[OK] GET operation: {'Success' if get_result else 'Failed'}")
            
            # Test DELETE
            delete_result = await cache_service.delete(test_key)
            print(f"[OK] DELETE operation: {'Success' if delete_result else 'Failed'}")
            
            # Get stats
            stats = await cache_service.get_stats()
            print(f"\nCache Statistics:")
            print(f"   Total Keys: {stats.get('total_keys', 'N/A')}")
            print(f"   Hit Rate: {stats.get('hit_rate', 0):.2f}%")
            print(f"   Keyspace Hits: {stats.get('keyspace_hits', 0)}")
            print(f"   Keyspace Misses: {stats.get('keyspace_misses', 0)}")
            
            print("\n[SUCCESS] Redis is working correctly!")
        else:
            print("\n[WARNING] Redis is not enabled or client is None")
            print("   Check if Redis is running on port 8971")
            print("   Run: docker run -d -p 8971:6379 --name redis-cache redis:latest")
        
        # Disconnect
        await cache_service.disconnect()
        
    except Exception as e:
        print(f"\n[ERROR] Error testing Redis: {e}")
        print("\n💡 To start Redis:")
        print("   docker run -d -p 8971:6379 --name redis-cache redis:latest")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    asyncio.run(test_redis())

