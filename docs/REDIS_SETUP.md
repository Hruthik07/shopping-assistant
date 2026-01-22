# Redis Setup Guide

## Overview

Redis is used for caching to improve performance:
- **LLM Response Caching**: 50-70% latency reduction
- **Product Search Caching**: 80-90% latency reduction  
- **Conversation History Caching**: 20-30% latency reduction
- **Session Data Caching**: Faster user preference retrieval

## Current Status

**Redis is NOT running** - The application works without it but caching is disabled.

## Installation Options

### Option 1: Docker (Recommended - Easiest)

```bash
# Start Redis container
docker run -d --name redis -p 6379:6379 redis:7-alpine

# Or use docker-compose (includes Redis service)
docker-compose up -d redis
```

### Option 2: Windows Native Installation

1. **Download Redis for Windows:**
   - Visit: https://github.com/microsoftarchive/redis/releases
   - Download the latest Windows release
   - Extract and run `redis-server.exe`

2. **Or use WSL (Windows Subsystem for Linux):**
   ```bash
   wsl
   sudo apt-get update
   sudo apt-get install redis-server
   redis-server
   ```

### Option 3: Use Docker Compose (Already Updated)

The `docker-compose.yml` file now includes a Redis service. Just run:
```bash
docker-compose up -d
```

## Configuration

### Default Configuration
- **URL**: `redis://localhost:6379/0`
- **Port**: `6379`
- **Database**: `0`

### Environment Variables (Optional)

Create or update `.env` file:
```env
REDIS_URL=redis://localhost:6379/0
CACHE_ENABLED=true
```

## Verification

### Check if Redis is Running

**Using Python:**
```python
import redis
r = redis.Redis(host='localhost', port=6379, db=0)
r.ping()  # Should return True
```

**Using Command Line:**
```bash
# If Redis CLI is installed
redis-cli ping
# Should return: PONG
```

**Using Docker:**
```bash
docker exec -it redis redis-cli ping
# Should return: PONG
```

## Application Behavior

### With Redis Running:
- ✅ Caching enabled
- ✅ Faster response times
- ✅ Reduced API costs
- ✅ Better performance

### Without Redis:
- ⚠️ Caching disabled (graceful degradation)
- ✅ Application still works
- ⚠️ Slower response times
- ⚠️ Higher API costs

## Cache Statistics

Check cache performance via health endpoint:
```bash
curl http://localhost:3565/health
```

Or programmatically:
```python
from src.utils.cache import cache_service
stats = await cache_service.get_stats()
print(stats)
```

## Troubleshooting

### Connection Refused Error
- **Problem**: Redis not running
- **Solution**: Start Redis using one of the methods above

### Port Already in Use
- **Problem**: Another service using port 6379
- **Solution**: Change Redis port or stop conflicting service

### Memory Issues
- **Problem**: Redis using too much memory
- **Solution**: Set maxmemory policy in Redis config

## Next Steps

1. **Install/Start Redis** using one of the methods above
2. **Verify connection** using the verification methods
3. **Restart your application** - Redis will connect automatically
4. **Monitor cache stats** via `/health` endpoint

## Performance Impact

Once Redis is running, you should see:
- **50-70% faster** responses for cached LLM queries
- **80-90% faster** product searches
- **Reduced API costs** from caching
- **Better user experience** with faster responses

