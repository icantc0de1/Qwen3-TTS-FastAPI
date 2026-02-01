# Model Manager Module Documentation

**Module Path**: `api/src/inference/qwen3_tts_model_manager.py`

**Last Updated**: 2026-02-01

## Overview

The Qwen3ModelManager implements a sophisticated model caching system with lazy loading and automatic memory management. It ensures efficient GPU/CPU memory usage by keeping only actively used models in memory.

## Module Structure

```
api/src/inference/
├── qwen3_tts_model_manager.py    # ModelManager class
└── qwen3_tts_model.py            # Qwen3TTSModel wrapper (existing)
```

## Key Components

### 1. ModelCacheEntry (Dataclass)

**Purpose**: Tracks metadata for each cached model instance.

**Attributes**:
- `model`: The actual Qwen3TTSModel instance
- `model_type`: Classification (base, custom_voice, voice_design)
- `last_accessed`: Unix timestamp for idle detection
- `ref_count`: Active reference count
- `device`: Hardware location (cuda:0, cpu, etc.)

**Why Reference Counting?**
Multiple concurrent requests may reference the same model. Reference counting ensures we don't unload a model while it's still in use. The model is only eligible for auto-unload when `ref_count == 0`.

### 2. Qwen3ModelManager (Singleton)

**Purpose**: Centralized model lifecycle management.

**Singleton Pattern**: Ensures only one manager exists across the application, preventing duplicate model loads and memory waste.

**Key Features**:

#### Lazy Loading
Models are only loaded when first requested:
```python
manager = Qwen3ModelManager()
# No models loaded yet

model = manager.load_model("models/custom-voice", "custom_voice")
# Model loaded on first request
```

#### Reference Counting
Tracks active usage:
```python
model1 = manager.load_model("models/base", "base")  # ref_count = 1
model2 = manager.load_model("models/base", "base")  # ref_count = 2 (same instance)

manager.release_model("models/base")  # ref_count = 1
manager.release_model("models/base")  # ref_count = 0 (eligible for unload)
```

#### Auto-Unload
Models are automatically removed after idle timeout:
```python
# After 5 minutes (default) of inactivity with ref_count == 0
cleanup_idle_models()  # Unloads idle models
```

#### Device Management
Supports CUDA and CPU allocation:
```python
manager = Qwen3ModelManager(default_device="cuda")
model = manager.load_model("path", "type", device="cuda:1")  # Specific GPU
```

## API Reference

### Constructor

```python
Qwen3ModelManager(
    default_device: str = "cuda",
    idle_timeout: int = 300,  # 5 minutes
    dtype: torch.dtype = torch.bfloat16
)
```

**Parameters**:
- `default_device`: Primary compute device
- `idle_timeout`: Seconds before auto-unload (when ref_count == 0)
- `dtype`: Model precision (bfloat16 recommended for performance)

### Methods

#### load_model()
```python
def load_model(
    self,
    model_path: str,
    model_type: str,
    device: Optional[str] = None
) -> Qwen3TTSModel
```

Loads or retrieves a cached model.

**Behavior**:
- Returns cached model if available (updates access time)
- Loads new model if not cached
- Increments reference count
- Tracks failed loads to avoid retrying

**Raises**:
- `RuntimeError`: If model previously failed to load or loading fails
- `ValueError`: For invalid model_type

#### release_model()
```python
def release_model(self, model_path: str) -> None
```

Decrements reference count for a model.

**Important**: Does NOT immediately unload. Model stays cached until timeout.

#### unload_model()
```python
def unload_model(self, model_path: str, force: bool = False) -> bool
```

Explicitly removes a model from cache.

**Parameters**:
- `force`: If True, unload even with active references (dangerous)

**Returns**: True if model was unloaded

#### cleanup_idle_models()
```python
def cleanup_idle_models(self) -> list[str]
```

Removes models idle longer than timeout with zero references.

**Returns**: List of unloaded model paths

**Usage Pattern**:
```python
# Call periodically (e.g., every 60 seconds)
async def cleanup_task():
    while True:
        manager.cleanup_idle_models()
        await asyncio.sleep(60)
```

#### unload_all()
```python
def unload_all(self, force: bool = False) -> int
```

Unloads all cached models.

**Use Case**: Shutdown cleanup or emergency memory recovery.

#### State Queries

```python
def is_model_loaded(self, model_path: str) -> bool
def get_cached_models(self) -> list[dict[str, Any]]
def get_model_info(self, model_path: str) -> Optional[dict[str, Any]]
```

## Design Decisions

### 1. Thread-Safe with Locks

```python
_cache_lock = Lock()

with self._cache_lock:
    # Critical section
    self._cache[model_path] = entry
```

Multiple FastAPI workers may access the manager concurrently. Locks prevent race conditions during load/unload operations.

### 2. Failed Load Tracking

```python
self._failed_loads: set[str] = set()

if model_path in self._failed_loads:
    raise RuntimeError("Model previously failed to load")
```

Prevents repeated expensive failed load attempts.

### 3. CUDA Cache Clearing

```python
if entry.device == "cuda" and torch.cuda.is_available():
    torch.cuda.empty_cache()
```

Explicitly frees GPU memory after model unloading.

### 4. Graceful Reference Management

```python
entry.ref_count = max(0, entry.ref_count - 1)
```

Prevents negative reference counts from bugs.

## Integration Points

### Backend Layer
The Qwen3TTSBackend uses the manager:
```python
manager = Qwen3ModelManager()
model = manager.load_model(path, model_type)
try:
    # Generate audio
finally:
    manager.release_model(path)
```

### FastAPI Lifecycle
Startup event initializes manager, shutdown cleans up:
```python
@app.on_event("startup")
async def startup():
    app.state.model_manager = Qwen3ModelManager()

@app.on_event("shutdown")
async def shutdown():
    app.state.model_manager.unload_all(force=True)
```

### Background Tasks
Periodic cleanup task:
```python
async def cleanup_worker():
    while True:
        get_manager().cleanup_idle_models()
        await asyncio.sleep(300)
```

## Memory Management Strategy

### Optimal Cache Size
Depends on GPU memory:
- **8GB GPU**: 1-2 large (1.7B) models or 3-4 small (0.6B) models
- **16GB GPU**: 3-4 large or 6-8 small models
- **24GB+ GPU**: All model variants can coexist

### Recommended Settings

**High-Traffic Production**:
```python
Qwen3ModelManager(
    idle_timeout=600,  # 10 minutes (keep hot models loaded)
    dtype=torch.bfloat16
)
```

**Low-Memory Development**:
```python
Qwen3ModelManager(
    idle_timeout=60,   # 1 minute (aggressive cleanup)
    dtype=torch.float16  # Lower precision
)
```

### Memory Monitoring

```python
models = manager.get_cached_models()
for model in models:
    print(f"{model['path']}: {model['refs']} refs, idle {model['idle_time']:.0f}s")
```

## Testing Considerations

1. **Concurrency Tests**:
   - Multiple simultaneous load requests for same model
   - Load while unload in progress
   - Reference count accuracy under load

2. **Error Handling**:
   - Failed load retry prevention
   - Invalid model path handling
   - GPU OOM recovery

3. **Cleanup Tests**:
   - Idle timeout accuracy
   - Force unload with active references
   - Memory cleanup verification

## Performance Characteristics

- **Cache Hit**: ~1ms (return existing reference)
- **Cache Miss**: 5-30s (model loading time)
- **Unload**: ~100ms (plus CUDA cache clear)
- **Memory per Model**: ~4GB (0.6B) or ~8GB (1.7B)

## Future Enhancements

1. **LRU Eviction**: Remove least-recently-used when memory constrained
2. **Model Preloading**: Load models before first request based on schedule
3. **Multi-GPU Sharding**: Distribute models across multiple GPUs
4. **Quantization Support**: INT8/INT4 for memory-constrained deployments
5. **Model Warmup**: Pre-run inference to warm up CUDA kernels
