# Production Readiness Checklist

Use this checklist before deploying to production.

## Pre-Deployment

### Configuration
- [ ] All environment variables set in `.env`
- [ ] `SECRET_KEY` changed from default
- [ ] `ENVIRONMENT=production` set
- [ ] `CORS_ORIGINS` restricted to your domain(s)
- [ ] `LOG_LEVEL` set to `WARNING` or `ERROR`
- [ ] API keys configured and tested
- [ ] Database URL configured (PostgreSQL recommended for production)

### Dependencies
- [ ] All services running (Redis, Database)
- [ ] Redis accessible and responding
- [ ] Database accessible and initialized
- [ ] All Python dependencies installed (`pip install -r requirements.txt`)

### Security
- [ ] API keys stored securely (not in code)
- [ ] Secret key is strong and unique
- [ ] CORS origins restricted
- [ ] Rate limiting enabled
- [ ] HTTPS configured (if applicable)
- [ ] Database credentials secured
- [ ] Redis authentication enabled (if applicable)

## Health Checks

### Service Health
- [ ] `/api/health` returns `"status": "healthy"`
- [ ] `/api/health/liveness` returns `"status": "alive"`
- [ ] `/api/health/readiness` returns `"status": "ready"`
- [ ] Database check passes
- [ ] Cache check passes (or shows degraded if optional)
- [ ] LLM provider check passes

### Functional Tests
- [ ] Can process a test query: `POST /api/chat/`
- [ ] Response includes products
- [ ] Cache is working (check `/api/chat/cache/stats`)
- [ ] Latency is acceptable (<20s for first request)
- [ ] Error handling works (test invalid input)

## Monitoring Setup

### Metrics
- [ ] `/api/metrics` endpoint accessible
- [ ] Cache metrics available (`/api/metrics/cache`)
- [ ] Error tracking functional
- [ ] Latency tracking working

### Observability
- [ ] Langfuse configured (if using)
- [ ] Logs being written to `logs/app.log`
- [ ] Error alerts configured (if applicable)
- [ ] Monitoring dashboard set up (if applicable)

## Performance

### Baseline Metrics
- [ ] Average response time <20s
- [ ] Cache hit rate >30% (after warm-up)
- [ ] Error rate <1%
- [ ] LLM processing time <15s average

### Load Testing
- [ ] Tested with expected concurrent users
- [ ] No memory leaks observed
- [ ] Database connections properly managed
- [ ] Redis connections stable

## Documentation

### Deployment Docs
- [ ] `DEPLOYMENT.md` reviewed and accurate
- [ ] `.env.example` includes all required variables
- [ ] README updated with production notes
- [ ] Troubleshooting guide available

### Runbooks
- [ ] Deployment procedure documented
- [ ] Rollback procedure documented
- [ ] Emergency procedures documented
- [ ] Contact information available

## Backup & Recovery

### Backup Strategy
- [ ] Database backup procedure tested
- [ ] Redis backup procedure tested (if applicable)
- [ ] Backup restoration tested
- [ ] Backup schedule configured

### Disaster Recovery
- [ ] Recovery procedure documented
- [ ] Recovery time objective (RTO) defined
- [ ] Recovery point objective (RPO) defined
- [ ] Failover procedure tested

## Post-Deployment

### Verification
- [ ] Health checks passing
- [ ] Test queries working
- [ ] Metrics being collected
- [ ] Logs being written
- [ ] No critical errors in logs

### Monitoring
- [ ] Set up alerts for:
  - Health check failures
  - High error rates
  - High latency
  - Cache failures
  - Database connection issues

## Sign-Off

- [ ] All critical items checked
- [ ] Performance benchmarks met
- [ ] Security review completed
- [ ] Monitoring configured
- [ ] Team trained on procedures

**Ready for Production**: ☐ Yes ☐ No

**Deployed By**: _________________  
**Date**: _________________  
**Notes**: _________________
