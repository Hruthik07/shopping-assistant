"""Error tracking and statistics endpoints."""
from fastapi import APIRouter
from typing import Dict, Any

from src.analytics.error_tracker import error_tracker

router = APIRouter(prefix="/api/errors", tags=["errors"])


@router.get("/stats")
async def get_error_stats(window_seconds: int = 300) -> Dict[str, Any]:
    """Get error statistics for monitoring."""
    try:
        stats = error_tracker.get_error_stats(window_seconds=window_seconds)
        return stats
    except Exception as e:
        return {
            "error": str(e),
            "window_seconds": window_seconds
        }


@router.get("/recent")
async def get_recent_errors(limit: int = 10) -> Dict[str, Any]:
    """Get recent errors for debugging."""
    try:
        recent = error_tracker.get_recent_errors(limit=limit)
        return {
            "count": len(recent),
            "errors": recent
        }
    except Exception as e:
        return {
            "error": str(e),
            "count": 0,
            "errors": []
        }
