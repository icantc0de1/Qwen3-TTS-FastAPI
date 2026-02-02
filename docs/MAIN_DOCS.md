# Main Application Documentation

**Module Path**: `api/main.py`

**Last Updated**: 2026-02-01

## Overview

The main module is the entry point for the FastAPI application. It creates and configures the application with all necessary components, including routers, middleware, and lifecycle management.

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                    FastAPI Application                          │
├────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────┐    ┌──────────────────┐                  │
│  │   Lifespan       │    │   CORS Middleware │                  │
│  │   (Startup/      │───▶│   (Cross-Origin)  │                  │
│  │    Shutdown)     │    │                   │                  │
│  └──────────────────┘    └──────────────────┘                  │
│           │                                                      │
│           ▼                                                      │
│  ┌──────────────────┐                                           │
│  │  TTS Backend     │◀─── Initialize on startup                 │
│  │  (ModelManager)  │                                           │
│  └────────┬─────────┘                                           │
│           │                                                      │
│           ▼                                                      │
│  ┌──────────────────┐                                           │
│  │  TTS Service     │◀─── Initialize with backend               │
│  │  (Business Logic)│                                           │
│  └────────┬─────────┘                                           │
│           │                                                      │
│           ▼                                                      │
│  ┌──────────────────┐    ┌──────────────────┐                  │
│  │  Audio Router    │◀───│  Model/Voice     │                  │
│  │  (/v1/audio/*)   │    │  Routers         │                  │
│  └──────────────────┘    └──────────────────┘                  │
│                                                                  │
│  ┌──────────────────┐    ┌──────────────────┐                  │
│  │  Root Endpoint   │    │  Health Check    │                  │
│  │  (/)             │    │  (/health)       │                  │
│  └──────────────────┘    └──────────────────┘                  │
│                                                                  │
└────────────────────────────────────────────────────────────────┘
```

## Key Functions

### `create_app()`

Factory function that creates and configures the FastAPI application.

**Returns**: Configured FastAPI application instance

**Configuration**:
- Title, description, and version from settings
- Automatic API documentation at `/docs` and `/redoc`
- Lifespan context manager for startup/shutdown
- Router registration with `/v1` prefix

**Usage**:
```python
# main.py
app = create_app()

# Run with uvicorn
# uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### `configure_cors(app)`

Configures Cross-Origin Resource Sharing (CORS) middleware.

**Current Configuration** (Development-friendly):
- Allow all origins (`["*"]`)
- Allow credentials
- Allow all methods
- Allow all headers

**Warning**: For production, restrict to specific origins:
```python
allow_origins=["https://yourdomain.com", "https://app.yourdomain.com"]
```

### `lifespan(app)`

Async context manager for application lifecycle management.

**Startup Sequence**:
1. Initialize TTS Backend
   - Create Qwen3TTSBackend instance
   - Detect CUDA/CPU availability
   - Load tokenizer if configured
   
2. Initialize TTS Service
   - Create Qwen3TTSService with backend
   - Load voice/model mappings
   
3. Register service with router
   - Make service available to endpoints

**Shutdown Sequence**:
1. Cleanup backend
   - Unload all models
   - Free GPU memory
   - Close resources

**Error Handling**:
- Startup failures are logged and re-raised
- Shutdown errors are logged but don't prevent cleanup

## Application State

The application uses FastAPI's `app.state` to store shared resources:

```python
app.state.tts_backend = backend    # Qwen3TTSBackend instance
app.state.tts_service = service    # Qwen3TTSService instance
app.state.cleanup_task = task      # Background cleanup task (asyncio.Task)
```

This allows dependency injection in routers:
```python
async def endpoint(request: Request):
    service = request.app.state.tts_service
    # Use service...
```

## Endpoints

### GET /

**Purpose**: Root endpoint providing API information and health status.

**Response**:
```json
{
  "status": "healthy",
  "service": "Qwen3 TTS API",
  "version": "1.0.0",
  "docs": "/docs",
  "endpoints": {
    "speech": "/v1/audio/speech",
    "models": "/v1/audio/models",
    "voices": "/v1/audio/voices"
  }
}
```

**Use Case**: Health checks, service discovery, API exploration.

### GET /health

**Purpose**: Simple health check for monitoring systems.

**Response**:
```json
{
  "status": "healthy",
  "timestamp": 1700000000.123
}
```

**Use Case**: Load balancer health checks, Kubernetes probes, uptime monitoring.

## Admin Endpoints

The admin router provides endpoints for managing and monitoring the TTS service.

### GET /admin/config

**Purpose**: View current application configuration (excluding sensitive values).

**Response**:
```json
{
  "api_title": "Qwen3 TTS API",
  "api_version": "1.0.0",
  "host": "0.0.0.0",
  "port": 8000,
  "idle_timeout": 300,
  "cleanup_interval": 30,
  "cleanup_enabled": true,
  "default_device": "cuda",
  "default_model_size": "1.7B"
}
```

**Use Case**: Debugging, monitoring, verifying configuration.

### POST /admin/cleanup

**Purpose**: Trigger immediate resource cleanup or check cleanup status.

**Query Parameters**:
- `trigger` (bool): If true, immediately trigger cleanup of idle resources

**Response (trigger=false)**:
```json
{
  "cleanup_enabled": true,
  "cleanup_interval": 30,
  "idle_timeout": 300,
  "status": "running",
  "timestamp": 1700000000.123,
  "vram_before_mb": 4096.5,
  "vram_after_mb": 512.0,
  "vram_freed_mb": 3584.5,
  "models_unloaded": 2
}
```

**Response (trigger=true)**:
```json
{
  "cleanup_enabled": true,
  "cleanup_interval": 30,
  "idle_timeout": 300,
  "status": "completed",
  "timestamp": 1700000000.123,
  "message": "Cleanup completed successfully",
  "vram_before_mb": 4096.5,
  "vram_after_mb": 512.0,
  "vram_freed_mb": 3584.5,
  "models_unloaded": 2,
  "models_checked": 3
}
```

**Use Case**: Manual resource cleanup, health monitoring, admin operations, VRAM usage tracking.

## Background Cleanup Worker

The application includes an optional background task that periodically frees GPU memory by unloading idle models.

### How It Works

1. **Periodic Checks**: The cleanup task runs every `cleanup_interval` seconds (default: 30)
2. **Idle Detection**: Models unused for longer than `idle_timeout` seconds are considered idle (default: 300)
3. **Resource Release**: Idle models are moved from GPU to CPU memory, freeing VRAM
4. **Automatic Reload**: Models are automatically reloaded when requested again

### Configuration

Enable/disable and tune the cleanup worker via environment variables:

```bash
# Enable cleanup (default: true)
export TTS_CLEANUP_ENABLED=true

# Check every 30 seconds (default: 30)
export TTS_CLEANUP_INTERVAL=30

# Unload after 5 minutes idle (default: 300)
export TTS_IDLE_TIMEOUT=300
```

### Lifecycle

- **Startup**: Cleanup task is started during application lifespan if `cleanup_enabled=true`
- **Operation**: Runs continuously in background, checking for idle resources
- **Shutdown**: Task is cancelled and awaited during application shutdown

### Benefits

- **Memory Efficiency**: GPU memory is freed when not actively needed
- **Multi-Model Support**: Allows switching between multiple voice models without manual unloading
- **Automatic Management**: No manual intervention required

### Trade-offs

- **First Request Latency**: Slight delay when reloading an unloaded model
- **CPU Memory Usage**: Unloaded models still occupy system RAM

### Implementation Details

The cleanup worker implementation includes several reliability improvements:

1. **Deadlock Prevention**: The `cleanup_idle_models()` function identifies models to unload while holding the lock, then releases the lock before calling `unload_model_internal()`. This prevents deadlocks that could occur when trying to acquire the same lock recursively.

2. **Garbage Collection**: `gc.collect()` is called after model deletion to force Python garbage collection and ensure memory is freed promptly.

3. **CUDA Synchronization**: `torch.cuda.synchronize()` is called before `torch.cuda.empty_cache()` to ensure all pending CUDA operations complete before attempting to free GPU memory.

4. **VRAM Monitoring**: VRAM usage is logged before and after cleanup operations for verification and debugging purposes.

5. **Memory Management**: Local variables in backend methods are explicitly cleared before releasing model references, preventing reference cycles that could delay garbage collection.

## Configuration

The application reads configuration from environment variables and `api.src.core.config.settings`:

### Environment Variables

Configuration is loaded from environment variables with `TTS_` prefix:

```bash
# Core settings
export TTS_IDLE_TIMEOUT=600              # Seconds to unload idle models (default: 600)
export TTS_CLEANUP_INTERVAL=60           # Cleanup check interval in seconds (default: 60)
export TTS_CLEANUP_ENABLED=true          # Enable background cleanup worker (default: true)
export TTS_DEFAULT_DEVICE=cuda           # Default device: cuda or cpu (default: cuda)
export TTS_DEFAULT_MODEL_SIZE=small      # Default model size: small (0.6B) or large (1.7B)
export TTS_ATTENTION_BACKEND=sdpa        # Attention backend: eager, sdpa, or flash_attention_2
```

### Settings Reference

- `api_title`: Service name
- `api_description`: Service description
- `api_version`: API version
- `host`: Server bind address
- `port`: Server port
- `model_paths`: Paths to various models
- `idle_timeout`: Seconds before unloading idle models from GPU memory
- `cleanup_interval`: Seconds between cleanup worker checks
- `cleanup_enabled`: Whether the background cleanup worker runs
- `default_device`: Default compute device ("cuda" or "cpu")
- `default_model_size`: Default model variant ("small"=0.6B or "large"=1.7B)
- `attention_backend`: Attention implementation ("eager", "sdpa", or "flash_attention_2")

See [config documentation](CORE_CONFIG_DOCS.md) for details.

## Running the Application

### Development Mode (with hot reload)

```bash
uv run uvicorn api.main:app --reload
```

### Production Mode

```bash
uv run uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

**Production Considerations**:
- Use multiple workers for concurrency
- Configure CORS for specific origins only
- Set up proper logging
- Use a process manager (systemd, supervisor)
- Consider running behind a reverse proxy (nginx, traefik)

### Docker

```dockerfile
FROM python:3.12

WORKDIR /app
COPY . .
RUN uv pip install -e .

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Testing

### Manual Testing

```bash
# Test root endpoint
curl http://localhost:8000/

# Test health endpoint
curl http://localhost:8000/health

# Test API documentation
curl http://localhost:8000/docs
```

### Automated Testing

```python
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
```

## Deployment Checklist

- [ ] Environment variables configured (TTS_IDLE_TIMEOUT, TTS_CLEANUP_INTERVAL, etc.)
- [ ] Configuration validated via /admin/config endpoint
- [ ] Model files present in `models/` directory
- [ ] Dependencies installed (`uv pip install -e .`)
- [ ] CUDA available (for GPU acceleration)
- [ ] CORS configured for production domains
- [ ] Logging configured appropriately
- [ ] Health check endpoints accessible
- [ ] Admin endpoints secured (authentication, IP whitelist)
- [ ] API documentation accessible (optional)
- [ ] SSL/TLS configured (for HTTPS)
- [ ] Rate limiting configured (optional)
- [ ] Authentication configured (optional)

## Troubleshooting

### Startup Failures

**Symptom**: Application fails to start

**Common Causes**:
1. Missing model files
2. CUDA out of memory
3. Invalid configuration
4. Port already in use

**Solution**:
```bash
# Check model files exist
ls -la models/base models/custom-voice

# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Check port availability
lsof -i :8000
```

### Service Unavailable (503)

**Symptom**: API returns 503 errors

**Cause**: TTS service not initialized

**Solution**: Check startup logs for initialization errors

### Memory Issues

**Symptom**: Out of memory errors

**Solutions**:
- Reduce idle_timeout in backend configuration
- Use smaller models (0.6B instead of 1.7B)
- Reduce number of workers
- Enable model quantization

## Future Enhancements

1. **Configuration Reload**: Hot-reload configuration without restart
2. **Metrics Endpoint**: Prometheus-compatible metrics
3. **Graceful Shutdown**: Wait for requests to complete before shutdown
4. **API Versioning**: Support multiple API versions
5. **WebSocket Support**: Real-time bidirectional communication
6. **Extended Admin Endpoints**: Model preloading, cache statistics, performance metrics
