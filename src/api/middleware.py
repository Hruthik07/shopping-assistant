"""API middleware for rate limiting and logging."""

import asyncio
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from collections import defaultdict
from datetime import datetime
from src.analytics.logger import logger
from src.utils.config import settings


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware.

    Uses a per-instance ``asyncio.Lock`` to serialise mutations of the
    ``clients`` dictionary, eliminating the race condition that could cause
    incorrect request counts under concurrent load (multiple coroutines
    reading/writing the same list without synchronisation).
    """

    def __init__(self, app, calls: int = 60, period: int = 60):
        super().__init__(app)
        self.calls = calls
        self.period = period
        # Dict[ip -> list[datetime]] – access always protected by _lock
        self.clients: dict[str, list[datetime]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host if request.client else "unknown"
        now = datetime.now()

        async with self._lock:
            # Evict timestamps outside the rolling window
            cutoff = now.timestamp() - self.period
            self.clients[client_ip] = [
                ts for ts in self.clients[client_ip]
                if ts.timestamp() > cutoff
            ]

            if len(self.clients[client_ip]) >= self.calls:
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded: {self.calls} requests per {self.period} seconds",
                )

            # Record this request inside the lock to prevent TOCTOU
            self.clients[client_ip].append(now)

        response = await call_next(request)
        return response


class LoggingMiddleware(BaseHTTPMiddleware):
    """Request logging middleware."""

    async def dispatch(self, request: Request, call_next):
        start_time = datetime.now()
        client = request.client.host if request.client else "unknown"

        logger.info(f"{request.method} {request.url.path} - {client}")

        response = await call_next(request)

        process_time = (datetime.now() - start_time).total_seconds()
        logger.info(
            f"{request.method} {request.url.path} - "
            f"Status: {response.status_code} - "
            f"Time: {process_time:.3f}s"
        )

        return response


class XRayTracingMiddleware(BaseHTTPMiddleware):
    """AWS X-Ray distributed tracing middleware.

    Creates one X-Ray segment per HTTP request, recording the HTTP method,
    URL, status code, and latency.  Propagates the ``X-Amzn-Trace-Id`` header
    so downstream AWS services (Lambda, RDS Proxy, etc.) appear as subsegments
    in the same trace.

    Enabled when ``XRAY_ENABLED=true`` is set in the environment.
    If the ``aws-xray-sdk`` package is not installed the middleware is
    a transparent no-op so local development continues to work.
    """

    def __init__(self, app, service_name: str = "shopping-assistant"):
        super().__init__(app)
        self.service_name = service_name
        self._xray_available = False
        self._recorder = None

        try:
            from aws_xray_sdk.core import xray_recorder, patch_all
            from aws_xray_sdk.core.context import Context

            xray_recorder.configure(
                service=service_name,
                context_missing="LOG_ERROR",  # don't raise on missing context
            )
            # Patch supported libraries (botocore, requests, sqlite3, etc.)
            patch_all()
            self._recorder = xray_recorder
            self._xray_available = True
            logger.info("AWS X-Ray tracing enabled for service: %s", service_name)
        except ImportError:
            logger.warning(
                "aws-xray-sdk not installed – X-Ray tracing disabled. "
                "Install it with: pip install aws-xray-sdk"
            )
        except Exception as exc:
            logger.warning("Failed to initialise X-Ray recorder: %s", exc)

    async def dispatch(self, request: Request, call_next):
        if not self._xray_available or self._recorder is None:
            return await call_next(request)

        # Extract upstream trace header if present (ALB / API Gateway injects this)
        trace_header = request.headers.get("X-Amzn-Trace-Id")

        segment_name = f"{request.method} {request.url.path}"
        try:
            self._recorder.begin_segment(
                name=segment_name,
                traceid=trace_header,
            )
            segment = self._recorder.current_segment()
            segment.put_http_meta("request", {
                "method": request.method,
                "url": str(request.url),
                "client_ip": request.client.host if request.client else "unknown",
            })
        except Exception:
            # Never let tracing break real requests
            return await call_next(request)

        try:
            response = await call_next(request)
            try:
                segment.put_http_meta("response", {"status": response.status_code})
                if response.status_code >= 500:
                    segment.add_fault_flag()
                elif response.status_code >= 400:
                    segment.add_error_flag()
            except Exception:
                pass
            return response
        except Exception as exc:
            try:
                segment.add_fault_flag()
                segment.add_exception(exc, remote=False)
            except Exception:
                pass
            raise
        finally:
            try:
                self._recorder.end_segment()
            except Exception:
                pass
