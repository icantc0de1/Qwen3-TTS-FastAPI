"""Configuration settings for Qwen3 TTS API.

This module uses Pydantic Settings to load configuration from environment
variables and .env files. Environment variables take precedence over .env
file values, which take precedence over defaults.

Example .env file:
    IDLE_TIMEOUT=600
    CLEANUP_INTERVAL=60
    DEFAULT_DEVICE=cuda
    DEFAULT_MODEL_SIZE=small
    CLEANUP_ENABLED=true
"""

from pathlib import Path
from typing import Literal

import importlib.util
import torch
from loguru import logger
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent.parent


def detect_fastest_attention_backend() -> str:
    """Detect and return the fastest available attention backend.

    Priority order:
    1. flash_attention_2 - Requires flash-attn package (2-5x speedup)
    2. sdpa - PyTorch native efficient attention (good performance)
    3. eager - Fallback (slowest)

    Returns:
        String identifier for the fastest available backend
    """
    if torch.cuda.is_available():
        if importlib.util.find_spec("flash_attn") is not None:
            logger.info("Flash Attention 2 detected - using flash_attention_2 backend")
            return "flash_attention_2"

        if hasattr(torch.nn.attention, "SDPBackend"):
            logger.info("PyTorch SDPA available - using sdpa backend")
            return "sdpa"

    logger.warning("CUDA not available - using eager backend")
    return "eager"


class Settings(BaseSettings):
    """Application settings with environment variable support.

    All settings can be overridden via environment variables or .env file.
    Priority: Environment Variables > .env file > Default values
    """

    # API Settings
    api_title: str = "Qwen3 TTS API"
    api_description: str = "API for text-to-speech generation using Qwen3 models"
    api_version: str = "1.0.0"
    host: str = "127.0.0.1"
    port: int = 8880
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )

    # Model Manager Settings
    idle_timeout: int = Field(
        default=600,
        description="Seconds before unloading idle models (default: 10 minutes)",
    )
    default_device: str = Field(
        default="cuda",
        description="Default device for model loading ('cuda' or 'cpu')",
    )
    default_model_size: Literal["small", "large"] = Field(
        default="small",
        description="Default model size ('small'=0.6B, 'large'=1.7B)",
    )
    attention_backend: Literal["auto", "eager", "sdpa", "flash_attention_2"] = Field(
        default="auto",
        description="Attention implementation backend ('auto' for auto-detection, eager, sdpa, flash_attention_2). Auto-detects fastest available backend at startup.",
    )

    # Cleanup Settings
    cleanup_enabled: bool = Field(
        default=True,
        description="Enable automatic cleanup of idle models",
    )
    cleanup_interval: int = Field(
        default=60,
        description="Seconds between cleanup checks (default: 60s)",
    )

    # Model paths (relative to project root or absolute)
    # Small models (0.6B) - faster, ~4GB VRAM
    base_model_path: str = "models/base"
    custom_voice_model_path: str = "models/custom-voice"
    voice_design_model_path: str = "models/voice-design"

    # Large models (1.7B) - better quality, ~8GB VRAM
    base_large_model_path: str = "models/base-large"
    custom_voice_large_model_path: str = "models/custom-voice-large"

    # Tokenizer path
    tokenizer_path: str = "models/tokenizer"

    def get_model_path(self, model_name: str) -> str | None:
        """Get the model path based on model name.

        Maps model identifiers to their corresponding paths.
        Uses default_model_size to determine which variant to use.
        Returns absolute paths (resolves relative paths to project root).
        """
        # Determine which size variant to use based on settings
        size_suffix = "-large" if self.default_model_size == "large" else ""

        model_paths = {
            # Small models (0.6B) - default, faster
            "qwen3-tts-12hz-0.6b-base": self.base_model_path,
            "qwen3-tts-12hz-0.6b-custom-voice": self.custom_voice_model_path,
            # Large models (1.7B) - better quality
            "qwen3-tts-12hz-1.7b-base": self.base_large_model_path,
            "qwen3-tts-12hz-1.7b-custom-voice": self.custom_voice_large_model_path,
            "qwen3-tts-12hz-1.7b-voice-design": self.voice_design_model_path,
            # OpenAI-compatible aliases (use default_model_size)
            "tts-1": (
                self.custom_voice_model_path
                if self.default_model_size == "small"
                else self.custom_voice_large_model_path
            ),
            "tts-1-hd": self.voice_design_model_path,
            # Generic aliases that respect default_model_size
            "base": getattr(self, f"base{size_suffix}_model_path"),
            "custom-voice": getattr(self, f"custom_voice{size_suffix}_model_path"),
            "voice-design": self.voice_design_model_path,
        }
        path = model_paths.get(model_name)
        if path:
            return self.resolve_path(path)
        return None

    def get_tokenizer_path(self) -> str:
        """Get the tokenizer path (always returns absolute path)."""
        return self.resolve_path(self.tokenizer_path)

    def resolve_path(self, path: str) -> str:
        """Resolve a path to absolute path.

        If path is relative, resolves it relative to project root.
        If path is already absolute, returns as-is.

        Args:
            path: Relative or absolute path

        Returns:
            Absolute path as string
        """
        path_obj = Path(path)
        if path_obj.is_absolute():
            return str(path_obj)
        # Relative path - resolve from project root
        return str(get_project_root() / path_obj)

    @field_validator("attention_backend")
    @classmethod
    def validate_attention_backend(cls, v: str) -> str:
        """Validate and resolve attention_backend setting.

        If 'auto' is specified, it will be resolved at startup time.

        Args:
            v: The attention backend value

        Returns:
            The validated attention backend value
        """
        if v == "auto":
            return v
        return v

    def resolve_attention_backend(self) -> str:
        """Resolve 'auto' backend to the fastest available backend.

        This method is called at application startup to determine
        which backend to use. It performs auto-detection and
        logs the selected backend.

        Returns:
            The resolved attention backend string
        """
        if self.attention_backend == "auto":
            backend = detect_fastest_attention_backend()
            logger.info(f"Auto-detected backend: {backend}")
            return backend
        return self.attention_backend

    class Config:
        """Pydantic configuration."""

        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()
