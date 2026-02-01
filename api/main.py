"""FastAPI application for Qwen3 TTS API."""

import asyncio
import time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from api.src.core.config import settings
from api.src.inference.qwen3_tts_backend import Qwen3TTSBackend
from api.src.routers import openai_compatible
from api.src.services.qwen3_tts_service import Qwen3TTSService


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    This factory function creates a fully configured FastAPI application
    with all routers, middleware, and lifecycle management.

    Returns:
        Configured FastAPI application instance

    Example:
        >>> app = create_app()
        >>> # Use with uvicorn
        >>> # uvicorn api.main:app --host 0.0.0.0 --port 8000
    """
    # Configure logging level from settings
    import sys

    logger.remove()
    logger.add(sys.stderr, level=settings.log_level)
    logger.info(f"Logging configured with level: {settings.log_level}")

    # Create FastAPI app with metadata
    app = FastAPI(
        title=settings.api_title,
        description=settings.api_description,
        version=settings.api_version,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Configure CORS
    configure_cors(app)

    # Include routers
    app.include_router(
        openai_compatible.router,
        prefix="/v1",
        tags=["audio"],
    )

    # Root endpoint
    @app.get("/")
    async def root() -> dict:
        """Root endpoint - health check and API info."""
        return {
            "status": "healthy",
            "service": settings.api_title,
            "version": settings.api_version,
            "docs": "/docs",
            "endpoints": {
                "speech": "/v1/audio/speech",
                "models": "/v1/audio/models",
                "voices": "/v1/audio/voices",
            },
            "config": {
                "default_device": settings.default_device,
                "default_model_size": settings.default_model_size,
                "cleanup_enabled": settings.cleanup_enabled,
                "idle_timeout": settings.idle_timeout,
            },
        }

    # Health check endpoint
    @app.get("/health")
    async def health_check() -> dict:
        """Health check endpoint for monitoring."""
        return {
            "status": "healthy",
            "timestamp": time.time(),
        }

    # Admin endpoint to get current settings
    @app.get("/admin/config")
    async def get_config() -> dict:
        """Get current configuration settings (admin only in production)."""
        return {
            "server": {
                "host": settings.host,
                "port": settings.port,
            },
            "model_manager": {
                "idle_timeout": settings.idle_timeout,
                "default_device": settings.default_device,
                "default_model_size": settings.default_model_size,
            },
            "cleanup": {
                "enabled": settings.cleanup_enabled,
                "interval": settings.cleanup_interval,
            },
        }

    # Admin endpoint to trigger manual cleanup
    @app.post("/admin/cleanup")
    async def manual_cleanup(request) -> dict:
        """Manually trigger cleanup of idle models (admin only in production)."""
        backend = getattr(request.app.state, "tts_backend", None)
        if backend is None:
            return {"error": "Backend not initialized"}

        unloaded = backend.cleanup()
        return {
            "unloaded": unloaded,
            "count": len(unloaded),
        }

    logger.info(f"FastAPI app created: {settings.api_title} v{settings.api_version}")
    return app


def configure_cors(app: FastAPI) -> None:
    """Configure CORS middleware for the application.

    Allows cross-origin requests from any origin. For production,
    consider restricting to specific origins.

    Args:
        app: FastAPI application instance
    """
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allow all origins (configure for production)
        allow_credentials=True,
        allow_methods=["*"],  # Allow all methods
        allow_headers=["*"],  # Allow all headers
    )
    logger.info("CORS middleware configured")


async def cleanup_worker(backend: Qwen3TTSBackend) -> None:
    """Background task to periodically clean up idle models.

    This task runs continuously and checks for idle models every
    cleanup_interval seconds. Models that have been idle longer
    than idle_timeout are unloaded to free VRAM.

    Args:
        backend: The TTS backend instance with the model manager
    """
    if not settings.cleanup_enabled:
        logger.info("Automatic cleanup is disabled. Models will remain loaded.")
        return

    logger.info(
        f"Starting cleanup worker: checking every {settings.cleanup_interval}s, "
        f"unloading after {settings.idle_timeout}s idle"
    )

    while True:
        try:
            # Wait for the configured interval
            await asyncio.sleep(settings.cleanup_interval)

            # Check for and unload idle models
            unloaded = backend.cleanup()

            if unloaded:
                logger.info(
                    f"Cleanup: unloaded {len(unloaded)} idle model(s): {unloaded}"
                )
            else:
                logger.debug("Cleanup: no idle models to unload")

        except asyncio.CancelledError:
            logger.info("Cleanup worker cancelled")
            break
        except Exception as e:
            logger.error(f"Cleanup worker error: {e}")
            # Continue running even after errors


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager.

    Handles startup and shutdown events for the FastAPI application.
    Initializes the TTS backend and service on startup, and cleans up
    resources on shutdown.

    Also starts a background task for automatic cleanup of idle models
    if cleanup_enabled is True.

    Yields:
        None

    Example:
        The lifespan context manager is automatically used by FastAPI
        when the app starts and stops.
    """
    # Startup
    logger.info("Application starting up...")
    cleanup_task = None

    try:
        # Determine device from settings
        device = settings.default_device
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            device = "cpu"

        # Initialize TTS backend with settings
        logger.info(
            f"Initializing backend: device={device}, "
            f"idle_timeout={settings.idle_timeout}s"
        )
        backend = Qwen3TTSBackend(
            device=device,
            idle_timeout=settings.idle_timeout,
        )

        # Initialize tokenizer if path is configured
        tokenizer_path = settings.get_tokenizer_path()
        if tokenizer_path:
            await backend.initialize(tokenizer_path=tokenizer_path)
        else:
            await backend.initialize()

        # Initialize TTS service
        service = Qwen3TTSService(backend)
        await service.initialize()

        # Store in app state for dependency injection
        app.state.tts_backend = backend
        app.state.tts_service = service

        # Register service with router
        openai_compatible.set_tts_service(service)

        # Start background cleanup task if enabled
        if settings.cleanup_enabled:
            cleanup_task = asyncio.create_task(
                cleanup_worker(backend), name="cleanup_worker"
            )
            logger.info(
                f"Cleanup worker started (interval: {settings.cleanup_interval}s)"
            )
        else:
            logger.info("Cleanup worker disabled - models will stay loaded")

        logger.info("Application startup complete")

    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise

    yield

    # Shutdown
    logger.info("Application shutting down...")

    try:
        # Cancel cleanup task if running
        if cleanup_task is not None and not cleanup_task.done():
            cleanup_task.cancel()
            try:
                await cleanup_task
            except asyncio.CancelledError:
                pass
            logger.info("Cleanup worker stopped")

        # Cleanup backend
        if hasattr(app.state, "tts_backend"):
            await app.state.tts_backend.shutdown()

        logger.info("Application shutdown complete")

    except Exception as e:
        logger.error(f"Shutdown error: {e}")


# Create the application instance
app = create_app()
