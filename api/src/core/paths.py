"""Path utilities for Qwen3 TTS API.

This module provides centralized path management for the application,
ensuring consistent path resolution across different environments.
"""

import os
from pathlib import Path
from typing import Optional


def get_project_root() -> Path:
    """Get the project root directory.

    Returns:
        Path to the project root directory
    """
    return Path(__file__).parent.parent.parent.parent


def resolve_path(path: str | Path, base_dir: Optional[Path] = None) -> Path:
    """Resolve a path to an absolute path.

    If the path is already absolute, returns it as-is.
    If the path is relative, resolves it relative to base_dir (or project root if not provided).

    Args:
        path: The path to resolve (relative or absolute)
        base_dir: Base directory for relative path resolution (defaults to project root)

    Returns:
        Absolute Path object

    Example:
        >>> resolve_path("models/base")
        Path('/home/user/Apps/Qwen3-TTS-FastAPI/models/base')

        >>> resolve_path("/absolute/path")
        Path('/absolute/path')
    """
    path_obj = Path(path)

    if path_obj.is_absolute():
        return path_obj

    base = base_dir or get_project_root()
    return base / path_obj


def get_model_path(model_name: str) -> Optional[Path]:
    """Get the path to a specific model.

    Args:
        model_name: Name of the model (e.g., "base", "custom-voice", "voice-design")

    Returns:
        Path to the model directory, or None if model not found

    Example:
        >>> get_model_path("base")
        Path('/home/user/Apps/Qwen3-TTS-FastAPI/models/base')
    """
    project_root = get_project_root()

    model_paths = {
        "base": project_root / "models" / "base",
        "custom-voice": project_root / "models" / "custom-voice",
        "voice-design": project_root / "models" / "voice-design",
        "base-large": project_root / "models" / "base-large",
        "custom-voice-large": project_root / "models" / "custom-voice-large",
        "tokenizer": project_root / "models" / "tokenizer",
    }

    path = model_paths.get(model_name)
    if path and path.exists():
        return path

    return None


def ensure_dir(path: str | Path) -> Path:
    """Ensure a directory exists, creating it if necessary.

    Args:
        path: Path to the directory

    Returns:
        Path object for the directory
    """
    path_obj = resolve_path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def get_relative_path(path: str | Path, base_dir: Optional[Path] = None) -> Path:
    """Get the relative path from a base directory.

    Args:
        path: The path to make relative
        base_dir: Base directory (defaults to project root)

    Returns:
        Relative Path object
    """
    path_obj = Path(path)
    base = base_dir or get_project_root()

    if path_obj.is_absolute():
        return path_obj.relative_to(base)

    return path_obj
