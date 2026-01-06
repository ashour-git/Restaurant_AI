"""
Restaurant SaaS API - Main Application.

This is the main FastAPI application that serves as the backend for the
Restaurant SaaS platform, providing POS, menu management, analytics,
and AI-powered features.
"""

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

# Load environment variables from .env file FIRST
from dotenv import load_dotenv

load_dotenv()

# Add parent directory to path for ML module imports
parent_dir = Path(__file__).resolve().parent.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from app.api.routes import api_router
from app.api.routes.ml import initialize_all_models
from app.api.routes.ml import router as ml_router
from app.core.config import settings
from app.core.middleware import (
    RateLimitMiddleware,
    RequestIdMiddleware,
    SecurityHeadersMiddleware,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class TimingMiddleware(BaseHTTPMiddleware):
    """Add request timing headers for performance monitoring."""

    async def dispatch(self, request: Request, call_next):
        start_time = time.perf_counter()
        response = await call_next(request)
        process_time = (time.perf_counter() - start_time) * 1000
        response.headers["X-Process-Time-Ms"] = f"{process_time:.2f}"
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.

    Handles startup and shutdown events for the application.
    """
    # Startup
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"Debug mode: {settings.DEBUG}")

    # Initialize database tables
    logger.info("Initializing database...")
    try:
        from app.core.database import Base, engine
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created/verified successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")

    # Initialize ML models
    logger.info("Initializing ML models...")
    initialize_all_models(groq_api_key=os.environ.get("GROQ_API_KEY"))

    yield

    # Shutdown
    logger.info("Shutting down application...")


def create_application() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        FastAPI: Configured application instance.
    """
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="Restaurant SaaS API with POS, menu management, analytics, and AI features",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add GZip compression for responses > 500 bytes
    app.add_middleware(GZipMiddleware, minimum_size=500)

    # Add security headers
    app.add_middleware(SecurityHeadersMiddleware)

    # Add request ID for tracing
    app.add_middleware(RequestIdMiddleware)

    # Add rate limiting (100 requests per minute)
    app.add_middleware(RateLimitMiddleware, requests_per_minute=100)

    # Add timing middleware for performance monitoring
    app.add_middleware(TimingMiddleware)

    # Include API router
    app.include_router(api_router, prefix=settings.API_V1_PREFIX)

    # Include ML router
    app.include_router(ml_router, prefix=settings.API_V1_PREFIX)

    # Health check endpoint
    @app.get("/health", tags=["Health"])
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "app": settings.APP_NAME,
            "version": settings.APP_VERSION,
        }

    @app.get("/", tags=["Root"])
    async def root():
        """Root endpoint with API information."""
        return {
            "message": f"Welcome to {settings.APP_NAME}",
            "version": settings.APP_VERSION,
            "docs": "/docs",
            "health": "/health",
        }

    return app


# Create application instance
app = create_application()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
    )
