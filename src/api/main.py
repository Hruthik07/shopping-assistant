"""FastAPI main application."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
from contextlib import asynccontextmanager
from src.api.routes import chat, products, cart, health, metrics, errors, debug
from src.api.websocket import websocket_endpoint
from src.api.middleware import RateLimitMiddleware, LoggingMiddleware
from src.database.db import init_db
from src.analytics.logger import logger
from src.utils.config import settings
from src.utils.validation import validate_config
from src.utils.cache import cache_service


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

    # Shutdown
    logger.info("Shutting down application...")

    # Disconnect cache
    if settings.cache_enabled:
        try:
            await cache_service.disconnect()
        except Exception as e:
            logger.warning(f"Error disconnecting cache: {e}")


# Create FastAPI app
app = FastAPI(
    title="E-Commerce Shopping Assistant API",
    description="AI-powered shopping assistant with MCP tools and real-time product search",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
cors_origins = (
    settings.cors_origins.split(",")
    if hasattr(settings, "cors_origins") and settings.cors_origins
    else ["*"]
)
if hasattr(settings, "production_mode") and settings.production_mode and "*" in cors_origins:
    logger.warning("CORS is set to allow all origins in production. Consider restricting this.")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files (must be before API routes)
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

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

# Include routers
app.include_router(chat.router)
app.include_router(products.router)
app.include_router(cart.router)
app.include_router(health.router)
app.include_router(metrics.router)
app.include_router(errors.router)
app.include_router(debug.router)

# WebSocket endpoint
app.websocket("/ws")(websocket_endpoint)


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
