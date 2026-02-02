"""Model manager for Qwen3 TTS models with lazy loading and auto-unload."""

import gc
import time
from dataclasses import dataclass
from threading import Lock
from typing import Any, Optional

import torch
from loguru import logger

from api.src.inference.qwen3_tts_model import Qwen3TTSModel


@dataclass
class ModelCacheEntry:
    """Cache entry for a loaded model.

    Tracks model instance, last access time, and reference count for
    intelligent cache management.

    Attributes:
        model: The loaded Qwen3TTSModel instance
        model_type: Type of model (base, custom_voice, voice_design)
        last_accessed: Unix timestamp of last access
        ref_count: Number of active references to this model
        device: Device the model is loaded on
    """

    model: Qwen3TTSModel
    model_type: str
    last_accessed: float
    ref_count: int
    device: str


class Qwen3ModelManager:
    """Manages Qwen3 TTS models with lazy loading and auto-unload capabilities.

    This singleton manager provides:
    - Lazy loading: Models are loaded on first request
    - Auto-unload: Models are unloaded after idle timeout
    - Reference counting: Tracks active usage to prevent premature unloading
    - Device management: Handles CUDA/CPU allocation
    - Memory optimization: Keeps only frequently used models in memory

    The manager caches models by their path, allowing multiple model types
    to coexist. Each model type (base, custom_voice, voice_design) can be
    loaded independently.

    Example:
        >>> manager = Qwen3ModelManager()
        >>> model = manager.load_model("models/custom-voice", "custom_voice")
        >>> # Use model...
        >>> manager.release_model("models/custom-voice")
        >>> # Model will be unloaded after idle timeout if no other references
    """

    _instance: Optional["Qwen3ModelManager"] = None
    _lock: Lock = Lock()
    _init_params: Optional[dict] = None

    def __new__(
        cls,
        default_device: str = "cuda",
        idle_timeout: int = 300,
        dtype: torch.dtype = torch.bfloat16,
    ) -> "Qwen3ModelManager":
        """Singleton pattern to ensure only one manager exists.

        Args:
            default_device: Default device for model loading ("cuda" or "cpu")
            idle_timeout: Seconds before unloading idle models (default: 5 minutes)
            dtype: Data type for model weights (default: bfloat16)
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
                    cls._init_params = {
                        "default_device": default_device,
                        "idle_timeout": idle_timeout,
                        "dtype": dtype,
                    }
        return cls._instance

    def __init__(
        self,
        default_device: str = "cuda",
        idle_timeout: int = 300,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        """Initialize the model manager.

        Args:
            default_device: Default device for model loading ("cuda" or "cpu")
            idle_timeout: Seconds before unloading idle models (default: 5 minutes)
            dtype: Data type for model weights (default: bfloat16)
        """
        if self._initialized:
            return

        self._initialized = True
        # Use stored params from __new__ if available
        params = self._init_params or {}
        self._default_device = params.get("default_device", default_device)
        self._idle_timeout = params.get("idle_timeout", idle_timeout)
        self._dtype = params.get("dtype", dtype)

        # Cache storage: path -> ModelCacheEntry
        self._cache: dict[str, ModelCacheEntry] = {}
        self._cache_lock = Lock()

        # Track loading errors to avoid retrying failed loads
        self._failed_loads: set[str] = set()

        logger.info(
            f"Qwen3ModelManager initialized: device={default_device}, "
            f"timeout={idle_timeout}s, dtype={dtype}"
        )

    def load_model(
        self,
        model_path: str,
        model_type: str,
        device: Optional[str] = None,
    ) -> Qwen3TTSModel:
        """Load a model from path, using cache if available.

        This method implements lazy loading - models are only loaded when
        first requested. If the model is already in cache, it returns the
        cached instance and updates the access time.

        Args:
            model_path: Path to the model directory or HuggingFace repo ID
            model_type: Type of model ("base", "custom_voice", "voice_design")
            device: Device to load model on (defaults to manager's default)

        Returns:
            Loaded Qwen3TTSModel instance

        Raises:
            RuntimeError: If model loading fails
            ValueError: If model_type is invalid
        """
        if model_path in self._failed_loads:
            raise RuntimeError(f"Model at {model_path} previously failed to load")

        target_device = device or self._default_device

        with self._cache_lock:
            # Check if model is already cached
            if model_path in self._cache:
                entry = self._cache[model_path]
                entry.last_accessed = time.time()
                entry.ref_count += 1
                logger.debug(
                    f"Using cached model: {model_path} (refs={entry.ref_count})"
                )
                return entry.model

            # Load new model
            logger.info(f"Loading model from {model_path} on {target_device}")
            try:
                from api.src.core.config import settings

                model = Qwen3TTSModel.from_pretrained(
                    model_path,
                    device_map=target_device,
                    dtype=self._dtype,
                    attn_implementation=settings.attention_backend,
                )

                # Create cache entry
                entry = ModelCacheEntry(
                    model=model,
                    model_type=model_type,
                    last_accessed=time.time(),
                    ref_count=1,
                    device=target_device,
                )
                self._cache[model_path] = entry

                logger.info(f"Model loaded successfully: {model_path}")
                return model

            except Exception as e:
                self._failed_loads.add(model_path)
                logger.error(f"Failed to load model {model_path}: {e}")
                raise RuntimeError(f"Model loading failed: {e}") from e

    def release_model(self, model_path: str) -> None:
        """Release a reference to a cached model.

        Decrements the reference count. When the count reaches zero,
        the model becomes eligible for auto-unload after the idle timeout.

        Args:
            model_path: Path to the model to release

        Note:
            This does NOT immediately unload the model. The model will
            remain cached until the idle timeout expires or explicit
            unload is called.
        """
        with self._cache_lock:
            if model_path not in self._cache:
                logger.warning(f"Attempted to release uncached model: {model_path}")
                return

            entry = self._cache[model_path]
            entry.ref_count = max(0, entry.ref_count - 1)
            entry.last_accessed = time.time()

            logger.debug(f"Released model: {model_path} (refs={entry.ref_count})")

    def unload_model(self, model_path: str, force: bool = False) -> bool:
        """Explicitly unload a model from cache.

        Args:
            model_path: Path to the model to unload
            force: If True, unload even if there are active references

        Returns:
            True if model was unloaded, False if not in cache

        Note:
            Use with caution when force=True - this may cause errors if
            the model is still in use.
        """
        with self._cache_lock:
            return self._unload_model_internal(model_path, force)

    def _unload_model_internal(self, model_path: str, force: bool = False) -> bool:
        """Internal method to unload a model - must be called with lock held.

        Args:
            model_path: Path to the model to unload
            force: If True, unload even if there are active references

        Returns:
            True if model was unloaded, False if not in cache or has refs
        """
        if model_path not in self._cache:
            return False

        entry = self._cache[model_path]

        if entry.ref_count > 0 and not force:
            logger.warning(
                f"Cannot unload {model_path}: {entry.ref_count} active references"
            )
            return False

        # Unload model
        logger.info(f"Unloading model: {model_path}")

        # Log VRAM before deletion
        if entry.device == "cuda" and torch.cuda.is_available():
            vram_before = torch.cuda.memory_allocated() / 1024**3
            logger.debug(f"VRAM before unload: {vram_before:.2f} GB")

        # Delete model reference
        del entry.model
        del self._cache[model_path]

        # Force garbage collection to actually free the memory
        gc.collect()

        # Log VRAM after garbage collection
        if entry.device == "cuda" and torch.cuda.is_available():
            vram_after_gc = torch.cuda.memory_allocated() / 1024**3
            logger.debug(f"VRAM after gc.collect(): {vram_after_gc:.2f} GB")
            logger.debug(f"VRAM freed by gc: {vram_before - vram_after_gc:.2f} GB")

        return True

    def cleanup_idle_models(self) -> list[str]:
        """Remove models that have been idle longer than timeout.

        This should be called periodically (e.g., by a background task)
        to free memory from unused models.

        Returns:
            List of model paths that were unloaded
        """
        current_time = time.time()
        models_to_unload: list[str] = []

        # First pass: identify models to unload (with lock)
        with self._cache_lock:
            for model_path in list(self._cache.keys()):
                entry = self._cache[model_path]
                idle_time = current_time - entry.last_accessed

                if entry.ref_count == 0 and idle_time > self._idle_timeout:
                    logger.info(
                        f"Auto-unloading idle model: {model_path} "
                        f"(idle for {idle_time:.1f}s)"
                    )
                    models_to_unload.append(model_path)

        # Second pass: unload models (without holding lock to avoid deadlock)
        unloaded: list[str] = []
        for model_path in models_to_unload:
            with self._cache_lock:
                if self._unload_model_internal(model_path, force=False):
                    unloaded.append(model_path)

        # Clear CUDA cache once after all models are unloaded
        if unloaded and torch.cuda.is_available():
            # Synchronize to ensure all CUDA operations complete
            torch.cuda.synchronize()

            vram_before = torch.cuda.memory_allocated() / 1024**3
            torch.cuda.empty_cache()
            vram_after = torch.cuda.memory_allocated() / 1024**3

            logger.info(
                f"CUDA cache cleared: freed {vram_before - vram_after:.2f} GB, "
                f"current VRAM: {vram_after:.2f} GB"
            )

        return unloaded

    def get_cached_models(self) -> list[dict[str, Any]]:
        """Get information about currently cached models.

        Returns:
            List of dictionaries containing model info:
            - path: Model path
            - type: Model type
            - device: Device location
            - refs: Reference count
            - idle_time: Seconds since last access
        """
        current_time = time.time()
        models = []

        with self._cache_lock:
            for path, entry in self._cache.items():
                models.append(
                    {
                        "path": path,
                        "type": entry.model_type,
                        "device": entry.device,
                        "refs": entry.ref_count,
                        "idle_time": current_time - entry.last_accessed,
                    }
                )

        return models

    def unload_all(self, force: bool = False) -> int:
        """Unload all cached models.

        Args:
            force: If True, unload even with active references

        Returns:
            Number of models unloaded
        """
        # Get list of models to unload (with lock)
        with self._cache_lock:
            paths = list(self._cache.keys())

        # Unload each model (acquiring lock per model to avoid deadlock)
        count = 0
        for path in paths:
            with self._cache_lock:
                if self._unload_model_internal(path, force=force):
                    count += 1

        # Clear CUDA cache once after all models are unloaded
        if count > 0 and torch.cuda.is_available():
            torch.cuda.synchronize()
            vram_before = torch.cuda.memory_allocated() / 1024**3
            torch.cuda.empty_cache()
            vram_after = torch.cuda.memory_allocated() / 1024**3
            logger.info(
                f"unload_all: Cleared CUDA cache, freed {vram_before - vram_after:.2f} GB, "
                f"current VRAM: {vram_after:.2f} GB"
            )

        logger.info(f"Unloaded {count} models")
        return count

    def is_model_loaded(self, model_path: str) -> bool:
        """Check if a model is currently cached.

        Args:
            model_path: Path to check

        Returns:
            True if model is in cache
        """
        with self._cache_lock:
            return model_path in self._cache

    def get_model_info(self, model_path: str) -> Optional[dict[str, Any]]:
        """Get information about a specific cached model.

        Args:
            model_path: Path to the model

        Returns:
            Model info dict or None if not cached
        """
        current_time = time.time()

        with self._cache_lock:
            if model_path not in self._cache:
                return None

            entry = self._cache[model_path]
            return {
                "path": model_path,
                "type": entry.model_type,
                "device": entry.device,
                "refs": entry.ref_count,
                "idle_time": current_time - entry.last_accessed,
                "model_size": (
                    entry.model.tts_model_size
                    if hasattr(entry.model, "tts_model_size")
                    else "unknown"
                ),
            }
