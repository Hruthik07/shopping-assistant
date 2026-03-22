"""Health check and monitoring endpoints."""

from fastapi import APIRouter
from typing import Dict, Any
import time
import asyncio

from src.analytics.logger import logger
from src.utils.cache import cache_service
from src.utils.config import settings
from src.database.db import SessionLocal
from sqlalchemy import text

router = APIRouter(prefix="", tags=["health"])

# ---------------------------------------------------------------------------
# Readiness result cache – prevents the ALB health-check probe (fires every
# 10-30 s) from opening a new DB connection on every single request.
# ---------------------------------------------------------------------------
_READINESS_TTL_SECONDS = 10
_readiness_cache: Dict[str, Any] = {"result": None, "ts": 0.0}


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Comprehensive health check endpoint."""
    health_status = {"status": "healthy", "timestamp": time.time(), "checks": {}}

    overall_healthy = True

    # Check database connectivity
    try:
        db = SessionLocal()
        try:
            db.execute(text("SELECT 1"))
            db.commit()
            health_status["checks"]["database"] = {
                "status": "healthy",
                "message": "Database connection successful",
            }
        finally:
            db.close()
    except Exception as e:
        overall_healthy = False
        health_status["checks"]["database"] = {
            "status": "unhealthy",
            "message": f"Database connection failed: {str(e)}",
        }

    # Check Redis cache
    try:
        if cache_service.enabled and cache_service.redis_client:
            await cache_service.redis_client.ping()
            cache_stats = cache_service.get_cache_stats()
            health_status["checks"]["cache"] = {
                "status": "healthy",
                "message": "Redis cache connected",
                "hit_rate": cache_stats.get("hit_rate", 0),
                "enabled": True,
            }
        else:
            health_status["checks"]["cache"] = {
                "status": "degraded",
                "message": "Cache disabled or not connected",
                "enabled": False,
            }
    except Exception as e:
        overall_healthy = False
        health_status["checks"]["cache"] = {
            "status": "unhealthy",
            "message": f"Cache connection failed: {str(e)}",
            "enabled": False,
        }

    # Check LLM provider configuration
    try:
        provider = settings.llm_provider.lower()
        if provider == "anthropic":
            has_key = bool(settings.anthropic_api_key)
            health_status["checks"]["llm_provider"] = {
                "status": "healthy" if has_key else "degraded",
                "provider": "anthropic",
                "model": settings.llm_model,
                "configured": has_key,
                "message": "Anthropic configured" if has_key else "Anthropic API key missing",
            }
        elif provider == "openai":
            has_key = bool(settings.openai_api_key)
            health_status["checks"]["llm_provider"] = {
                "status": "healthy" if has_key else "degraded",
                "provider": "openai",
                "model": settings.llm_model,
                "configured": has_key,
                "message": "OpenAI configured" if has_key else "OpenAI API key missing",
            }
        else:
            health_status["checks"]["llm_provider"] = {
                "status": "degraded",
                "provider": provider,
                "message": f"Unknown provider: {provider}",
            }
    except Exception as e:
        overall_healthy = False
        health_status["checks"]["llm_provider"] = {
            "status": "unhealthy",
            "message": f"LLM provider check failed: {str(e)}",
        }

    # Set overall status
    if not overall_healthy:
        health_status["status"] = "unhealthy"
        return health_status

    # Check for degraded services
    degraded_checks = [
        check for check in health_status["checks"].values() if check.get("status") == "degraded"
    ]
    if degraded_checks:
        health_status["status"] = "degraded"

    return health_status


@router.get("/health/liveness")
async def liveness() -> Dict[str, str]:
    """Instant liveness probe – used by ALB / ECS target health checks.

    Must return in <1 s with no external calls.  The ALB uses this path to
    decide whether to route traffic to the container; a slow or failing
    response triggers rapid task replacement.
    """
    return {"status": "alive"}


@router.get("/health/readiness")
async def readiness() -> Dict[str, Any]:
    """Readiness probe – checks whether the service can accept traffic.

    Result is cached for ``_READINESS_TTL_SECONDS`` (10 s) so that a
    high-frequency ALB probe does not open a new DB connection on every
    request.  A stale positive result is safe: a truly dead DB will be
    caught within one TTL window.
    """
    now = time.time()
    if _readiness_cache["result"] is not None and (now - _readiness_cache["ts"]) < _READINESS_TTL_SECONDS:
        return _readiness_cache["result"]

    checks: Dict[str, str] = {}
    ready = True

    # Database connectivity
    try:
        db = SessionLocal()
        try:
            db.execute(text("SELECT 1"))
            db.commit()
            checks["database"] = "ready"
        finally:
            db.close()
    except Exception as e:
        ready = False
        checks["database"] = f"not_ready: {str(e)}"

    # Redis (optional – degraded, not fatal)
    try:
        if cache_service.enabled and cache_service.redis_client:
            await cache_service.redis_client.ping()
            checks["cache"] = "ready"
        else:
            checks["cache"] = "not_configured"
    except Exception as e:
        checks["cache"] = f"not_ready: {str(e)}"

    result: Dict[str, Any] = {
        "status": "ready" if ready else "not_ready",
        "checks": checks,
    }

    # Store in cache
    _readiness_cache["result"] = result
    _readiness_cache["ts"] = now

    return result
