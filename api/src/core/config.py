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

import os
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent.parent


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
            "tts-1": self.custom_voice_model_path
            if self.default_model_size == "small"
            else self.custom_voice_large_model_path,
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

    class Config:
        """Pydantic configuration."""

        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()
