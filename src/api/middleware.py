"""API middleware for rate limiting and logging."""

from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from collections import defaultdict
from datetime import datetime, timedelta
from src.analytics.logger import logger
from src.utils.config import settings


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware."""

    def __init__(self, app, calls: int = 60, period: int = 60):
        super().__init__(app)
        self.calls = calls
        self.period = period
        self.clients = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host

        # Clean old entries
        now = datetime.now()
        self.clients[client_ip] = [
            timestamp
            for timestamp in self.clients[client_ip]
            if (now - timestamp).seconds < self.period
        ]

        # Check rate limit
        if len(self.clients[client_ip]) >= self.calls:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded: {self.calls} requests per {self.period} seconds",
            )

        # Add current request
        self.clients[client_ip].append(now)

        response = await call_next(request)
        return response


class LoggingMiddleware(BaseHTTPMiddleware):
    """Request logging middleware."""

    async def dispatch(self, request: Request, call_next):
        start_time = datetime.now()

        # Log request
        logger.info(f"{request.method} {request.url.path} - {request.client.host}")

        response = await call_next(request)

        # Log response
        process_time = (datetime.now() - start_time).total_seconds()
        logger.info(
            f"{request.method} {request.url.path} - "
            f"Status: {response.status_code} - "
            f"Time: {process_time:.3f}s"
        )

        return response
