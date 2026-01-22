# Deployment Guide

## Production Deployment

This guide covers deploying the Global Customer-First Product Finder to production.

## Prerequisites

- Python 3.11+
- PostgreSQL (or SQLite for small deployments)
- Redis (for caching)
- API keys for data sources (Serper, Amazon, eBay, etc.)

## Step 1: Environment Setup

1. **Copy production environment file:**
   ```bash
   cp .env.production.example .env.production
   ```

2. **Update `.env.production` with your production values:**
   - API keys for all data sources
   - Database connection string
   - Redis URL
   - Secret keys
   - CORS origins

## Step 2: Database Initialization

1. **Initialize database:**
   ```bash
   python scripts/init_db.py
   ```

2. **Verify tables created:**
   - `users`
   - `sessions`
   - `conversations`
   - `user_preferences`
   - `cart_items`
   - `price_history` (new)

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 4: Run Background Jobs

Set up scheduled tasks for:
- Price history cleanup (daily)
- Price tracking updates (hourly)

**Using cron (Linux/Mac):**
```bash
# Daily cleanup at 2 AM
0 2 * * * cd /path/to/project && python scripts/background_jobs.py

# Hourly price tracking (if implemented)
0 * * * * cd /path/to/project && python scripts/background_jobs.py --track-prices
```

**Using Windows Task Scheduler:**
- Create task to run `python scripts/background_jobs.py` daily

## Step 5: Start Application

### Development Mode
```bash
python start_server.py
```

### Production Mode (using gunicorn/uvicorn)
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 3565 --workers 4
```

### Using Docker
```bash
docker-compose -f docker-compose.prod.yml up -d
```

## Step 6: Verify Deployment

1. **Health Check:**
   ```bash
   curl http://localhost:3565/api/health/liveness
   ```

2. **Test Product Search:**
   ```bash
   curl -X POST http://localhost:3565/api/chat \
     -H "Content-Type: application/json" \
     -d '{"message": "Find me wireless headphones under $100"}'
   ```

3. **Check Metrics:**
   ```bash
   curl http://localhost:3565/api/metrics/deals
   ```

## Monitoring

### Metrics Endpoints

- `/api/metrics` - General metrics
- `/api/metrics/deals` - Deal detection metrics
- `/api/metrics/cache` - Cache statistics
- `/api/metrics/latency` - Performance metrics

### Logging

- Application logs: `logs/production.log`
- Error tracking: Langfuse dashboard
- Performance: CloudWatch (if enabled)

## Scaling

### Horizontal Scaling

1. **Load Balancer:** Use nginx or AWS ALB
2. **Multiple Workers:** Increase uvicorn workers
3. **Database Connection Pooling:** Configure in database URL
4. **Redis Cluster:** For distributed caching

### Vertical Scaling

- Increase server resources (CPU, RAM)
- Optimize database indexes
- Enable query caching

## Security Checklist

- [ ] Change `SECRET_KEY` in production
- [ ] Use HTTPS (TLS/SSL)
- [ ] Configure CORS properly
- [ ] Set up rate limiting
- [ ] Enable API key rotation
- [ ] Use environment variables for secrets
- [ ] Enable database backups
- [ ] Set up monitoring alerts

## Troubleshooting

### Database Issues

- Check database connection string
- Verify tables exist: `python scripts/init_db.py`
- Check database logs

### API Issues

- Verify API keys are set
- Check rate limits
- Review error logs

### Performance Issues

- Check Redis cache status
- Review database query performance
- Monitor API response times

## Backup and Recovery

### Database Backup

```bash
# PostgreSQL
pg_dump shopping_assistant_prod > backup.sql

# SQLite
cp shopping_assistant.db backup.db
```

### Configuration Backup

- Backup `.env.production`
- Backup API keys securely
- Document all configuration changes

## Updates and Maintenance

1. **Pull latest code**
2. **Run database migrations** (if any)
3. **Restart application**
4. **Verify health checks**
5. **Monitor for errors**

## Support

For issues or questions:
- Check logs: `logs/production.log`
- Review metrics: `/api/metrics`
- Check Langfuse dashboard for traces
