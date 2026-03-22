"""FastAPI main application."""

import asyncio
import signal
import sys
from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
from contextlib import asynccontextmanager
from src.api.routes import chat, products, cart, health, metrics, errors, debug
from src.api.websocket import websocket_endpoint
from src.api.middleware import RateLimitMiddleware, LoggingMiddleware, XRayTracingMiddleware
from src.database.db import init_db
from src.analytics.logger import logger
from src.utils.config import settings
from src.utils.validation import validate_config
from src.utils.cache import cache_service

# ---------------------------------------------------------------------------
# Graceful-shutdown helpers
# ---------------------------------------------------------------------------
# Set by the SIGTERM handler; the lifespan checks it to skip redundant work.
_shutting_down = False


def _handle_sigterm(signum, frame):
    """Convert SIGTERM into a clean asyncio-friendly shutdown signal.

    ECS / ALB sends SIGTERM when draining a task.  Without this handler
    Python ignores SIGTERM (unlike SIGINT/Ctrl-C), so in-flight requests
    would be cut off mid-flight.  By raising KeyboardInterrupt we let
    uvicorn's graceful-shutdown path finish active connections before exit.
    """
    global _shutting_down
    _shutting_down = True
    logger.info("Received SIGTERM – initiating graceful shutdown")
    raise KeyboardInterrupt


# Register on non-Windows platforms (SIGTERM not available on Win32)
if sys.platform != "win32":
    signal.signal(signal.SIGTERM, _handle_sigterm)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting application...")

    # Validate configuration
    config_status = validate_config()
    if not config_status["valid"]:
        logger.error("Configuration validation failed - some features may not work")

    # Initialize database
    try:
        init_db()
        logger.info("Database initialized")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")

    # Initialize Redis cache
    if settings.cache_enabled:
        try:
            await cache_service.connect()
            logger.info("Cache service initialized")
        except Exception as e:
            logger.warning(f"Cache service initialization failed: {e}")

    # Validate API keys based on provider
    provider = settings.llm_provider.lower()
    if provider == "anthropic":
        if not settings.anthropic_api_key:
            logger.warning("ANTHROPIC_API_KEY not set - agent will not work properly")
        else:
            logger.info(f"Using Anthropic provider with model: {settings.llm_model}")
    elif provider == "openai":
        if not settings.openai_api_key:
            logger.warning("OPENAI_API_KEY not set - agent will not work properly")
        else:
            logger.info(f"Using OpenAI provider with model: {settings.llm_model}")
    else:
        logger.warning(f"Unknown LLM provider: {provider}")

    logger.info("Application startup complete")

    yield

    # ------------------------------------------------------------------
    # Shutdown – runs after uvicorn has stopped accepting new requests.
    # At this point all in-flight requests have either completed or been
    # given the graceful-shutdown timeout to complete.
    # ------------------------------------------------------------------
    logger.info("Shutting down application...")

    # Disconnect cache
    if settings.cache_enabled:
        try:
            await cache_service.disconnect()
        except Exception as e:
            logger.warning(f"Error disconnecting cache: {e}")

    logger.info("Shutdown complete.")


# Create FastAPI app
app = FastAPI(
    title="E-Commerce Shopping Assistant API",
    description="AI-powered shopping assistant with MCP tools and real-time product search",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
_raw_cors = settings.cors_origins.strip() if settings.cors_origins else ""
cors_origins = [o.strip() for o in _raw_cors.split(",") if o.strip()] if _raw_cors else ["*"]

if settings.production_mode and "*" in cors_origins:
    raise RuntimeError(
        "CORS_ORIGINS must be set to an explicit list of allowed origins in production. "
        "Setting it to '*' is not permitted. "
        "Set CORS_ORIGINS=https://your-domain.com in your environment."
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key"],
)

# Serve frontend static files (must be before API routes)
# Note: Path, StaticFiles, and FileResponse are already imported at the top

frontend_path = Path(__file__).parent.parent.parent / "frontend"
if frontend_path.exists():
    # Serve CSS and JS files directly
    @app.get("/styles.css")
    async def serve_css():
        css_file = frontend_path / "styles.css"
        if css_file.exists():
            return FileResponse(str(css_file), media_type="text/css")

    @app.get("/app.js")
    async def serve_js():
        js_file = frontend_path / "app.js"
        if js_file.exists():
            return FileResponse(str(js_file), media_type="application/javascript")


# Rate limiting middleware
app.add_middleware(RateLimitMiddleware, calls=settings.rate_limit_per_minute, period=60)

# Logging middleware
app.add_middleware(LoggingMiddleware)

# AWS X-Ray distributed tracing (enabled via XRAY_ENABLED=true env var)
import os as _os
if _os.getenv("XRAY_ENABLED", "false").lower() in ("1", "true", "yes"):
    app.add_middleware(XRayTracingMiddleware, service_name="shopping-assistant")

# ── Versioned router – canonical paths (/api/v1/...) ────────────────────────
_v1 = APIRouter(prefix="/api/v1")
_v1.include_router(chat.router)
_v1.include_router(products.router)
_v1.include_router(cart.router)
_v1.include_router(health.router)
_v1.include_router(metrics.router)
_v1.include_router(errors.router)
_v1.include_router(debug.router)
app.include_router(_v1)

# ── Legacy router – old /api/* paths (30-day deprecation window) ─────────────
# Remove this block after all clients have migrated to /api/v1/*
_legacy = APIRouter(prefix="/api")
_legacy.include_router(chat.router)
_legacy.include_router(products.router)
_legacy.include_router(cart.router)
_legacy.include_router(health.router)
_legacy.include_router(metrics.router)
_legacy.include_router(errors.router)
_legacy.include_router(debug.router)
app.include_router(_legacy)

# WebSocket – canonical path + legacy path
app.websocket("/api/v1/ws")(websocket_endpoint)
app.websocket("/ws")(websocket_endpoint)  # legacy – remove after deprecation window


# Root endpoint - serve frontend if available, otherwise API info
@app.get("/", response_class=FileResponse)
async def root():
    """Root endpoint - serve frontend if available."""
    frontend_path = Path(__file__).parent.parent.parent / "frontend"
    index_file = frontend_path / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    # Fallback to API info if frontend not found
    from fastapi.responses import JSONResponse

    return JSONResponse(
        {"message": "E-Commerce Shopping Assistant API", "version": "1.0.0", "docs": "/docs"}
    )


# Health check endpoints are now in health.py router


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.main:app", host=settings.api_host, port=settings.api_port, reload=True)
