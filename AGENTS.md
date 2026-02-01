# AGENTS.md - Guidelines for Agentic Coding

> This file contains essential information for AI coding agents working in this repository.

## Project Overview

This is a FastAPI-based TTS (Text-to-Speech) API using Qwen3 models. It provides OpenAI-compatible endpoints for speech generation with advanced features like voice cloning, custom voices, and voice design.

- **Language**: Python 3.12
- **Framework**: FastAPI + Pydantic v2
- **Package Manager**: `uv`
- **ML Stack**: PyTorch, Transformers, CUDA 12.6/12.8/13.0
- **License**: Apache 2.0

## Build/Test/Lint Commands

All commands use `uv` as the package manager:

```bash
# Run the development server with hot reload
uv run uvicorn api.main:app --reload

# Run the production server
uv run uvicorn api.main:app --host 0.0.0.0 --port 8000

# Install dependencies
uv pip install -e .

# Install dev dependencies
uv pip install -e ".[dev]"

# Install with specific CUDA version (see scripts/)
./scripts/install-cu130.sh  # or install-cu126.sh, install-cu128.sh, install-cpu.sh
```

### Code Quality

```bash
# Format code with Black (line-length: 88)
uv run black api/

# Lint with Ruff
uv run ruff check api/
uv run ruff check --fix api/    # Auto-fix issues

# Type checking with mypy
uv run mypy api/

# Run all checks (recommended before committing)
uv run black --check api/ && uv run ruff check api/ && uv run mypy api/
```

**Note on Type Checking:** Mypy will report errors in vendored Qwen3-TTS model files (`api/src/core/models/*`, `api/src/core/tokenizer_*`, `api/src/services/text_processing/normalizer.py`). These are third-party files that we do not modify. The configuration in `pyproject.toml` suppresses import errors for libraries without type stubs.

### Testing

```bash
# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_specific.py

# Run a specific test function
uv run pytest tests/test_specific.py::test_function_name

# Run with verbose output
uv run pytest -v

# Run with coverage
uv run pytest --cov=api --cov-report=html
```

## Project Structure

```
.
├── api/                          # FastAPI application
│   ├── main.py                   # Application entry point
│   └── src/                      # Source code
│       ├── core/                 # Core modules
│       │   ├── config.py         # Configuration settings (Pydantic)
│       │   ├── paths.py          # Path utilities
│       │   └── models/           # Model implementations
│       ├── inference/            # Model management and inference
│       │   ├── qwen3_tts_backend.py         # TTS backend operations
│       │   ├── qwen3_tts_model.py           # Model wrapper
│       │   └── qwen3_tts_model_manager.py   # Model lifecycle management
│       ├── routers/              # API routers
│       │   └── openai_compatible.py         # OpenAI-compatible endpoints
│       ├── services/             # Business logic
│       │   ├── qwen3_tts_service.py         # TTS service layer
│       │   └── text_processing/             # Text normalization
│       │       ├── __init__.py
│       │       └── normalizer.py
│       └── structures/           # Data schemas
│           └── schemas.py
│
├── docs/                         # Module documentation
│   ├── MAIN_DOCS.md              # FastAPI application docs
│   ├── INFERENCE_BACKEND_DOCS.md # Backend documentation
│   ├── INFERENCE_MODEL_MANAGER_DOCS.md # Model manager docs
│   ├── SERVICES_DOCS.md          # Service layer docs
│   ├── ROUTERS_DOCS.md           # API routers docs
│   ├── STRUCTURES_DOCS.md        # Data schemas docs
│   ├── PATHS_DOCS.md             # Path utilities docs
│   └── TEXT_PROCESSING_DOCS.md   # Text normalization docs
│
├── models/                       # Model weights (downloaded, gitignored)
│   ├── base/                     # 0.6B base model
│   ├── base-large/               # 1.7B base model
│   ├── custom-voice/             # 0.6B custom voice model
│   ├── custom-voice-large/       # 1.7B custom voice model
│   ├── voice-design/             # 1.7B voice design model
│   ├── tokenizer/                # Speech tokenizer
│   └── download_models.py        # Model download script
│
├── scripts/                      # Installation scripts
│   ├── install-cu126.sh          # CUDA 12.6 installer
│   ├── install-cu128.sh          # CUDA 12.8 installer
│   ├── install-cu130.sh          # CUDA 13.0 installer
│   └── install-cpu.sh            # CPU-only installer
│
├── .env.example                  # Example configuration
├── pyproject.toml                # Project dependencies and config
├── LICENSE                       # Apache 2.0 license
└── NOTICE                        # Attribution notices
```

## Key Features

### 1. VRAM Management & Cleanup
- **Lazy Loading**: Models loaded on first request
- **Auto-Unload**: Models unloaded after idle timeout (configurable via `IDLE_TIMEOUT`)
- **Background Cleanup Worker**: Periodic cleanup every `CLEANUP_INTERVAL` seconds
- **Reference Counting**: Prevents premature unloading during active use

Configuration via `.env`:
```
IDLE_TIMEOUT=600              # Seconds before unloading
CLEANUP_INTERVAL=60           # Seconds between checks
CLEANUP_ENABLED=true          # Enable/disable auto-cleanup
```

### 2. Text Normalization
- **Attribution**: Derived from remsky/kokoro-fastapi (Apache 2.0)
- **Features**: URL, email, phone number, unit, money, number normalization
- **API Integration**: Controlled via `normalization_options` in requests
- **Location**: `api/src/services/text_processing/normalizer.py`

### 3. Multi-CUDA Support
Install scripts support CUDA 12.6, 12.8, and 13.0:
- `install-cu126.sh`
- `install-cu128.sh`
- `install-cu130.sh`
- `install-cpu.sh`

### 4. Model Management
- **Download Script**: `models/download_models.py`
- **Multiple Model Types**: Base, custom-voice, voice-design (0.6B and 1.7B variants)
- **Automatic Downloads**: Via `qwen-tts` package or manual via script

### 5. Configuration System
Environment-based configuration via `.env`:
```
# Server
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO

# Model Manager
IDLE_TIMEOUT=600
DEFAULT_DEVICE=cuda
DEFAULT_MODEL_SIZE=small  # or 'large'

# Cleanup
CLEANUP_ENABLED=true
CLEANUP_INTERVAL=60
```

### 6. Admin Endpoints
- `GET /admin/config` - View current configuration
- `POST /admin/cleanup` - Trigger manual VRAM cleanup

## Code Style Guidelines

### General Style

- **Line length**: 88 characters (Black default)
- **Python version**: 3.12+ (use modern syntax)
- **Quote style**: Double quotes for strings

### Imports

Order imports in three groups separated by blank lines:

```python
# 1. Standard library imports
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

# 2. Third-party imports
import torch
from fastapi import FastAPI
from loguru import logger
from pydantic import BaseModel
from transformers import AutoModel

# 3. Local imports (use absolute imports from `api.`)
from api.src.core.config import settings
from api.src.structures.schemas import OpenAISpeechRequest
```

### Type Hints

- Use type hints for all function parameters and return types
- Use `from __future__ import annotations` for forward references if needed
- Use `|` for union types (Python 3.12+): `str | None`
- Use `list[Type]` instead of `List[Type]`

Example:

```python
def process_audio(audio_path: str, sample_rate: int = 16000) -> bytes | None:
    """Process audio file and return bytes.
    
    Args:
        audio_path: Path to the audio file
        sample_rate: Target sample rate
        
    Returns:
        Processed audio bytes or None if failed
    """
    ...
```

### Naming Conventions

- **Classes**: `PascalCase` (e.g., `Qwen3TTSConfig`, `OpenAISpeechRequest`)
- **Functions/Variables**: `snake_case` (e.g., `generate_speech`, `model_path`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_SAMPLE_RATE`)
- **Private members**: Prefix with `_` (e.g., `_internal_method()`)
- **Modules**: Short, lowercase (e.g., `config.py`, `openai_compatible.py`)

### Docstrings

Use Google-style docstrings:

```python
def load_model(model_path: str, device: str = "cuda") -> nn.Module:
    """Load a TTS model from the given path.
    
    Args:
        model_path: Path to the model directory or HuggingFace repo ID
        device: Device to load the model on ("cuda" or "cpu")
        
    Returns:
        The loaded PyTorch model
        
    Raises:
        FileNotFoundError: If model files are not found
        RuntimeError: If model loading fails
    """
    ...
```

### Error Handling

- Use specific exceptions over generic `Exception`
- Use `loguru` for logging errors with context
- Handle expected errors gracefully with appropriate HTTP status codes

```python
from loguru import logger
from fastapi import HTTPException

try:
    model = load_model(path)
except FileNotFoundError as e:
    logger.error(f"Model not found at {path}: {e}")
    raise HTTPException(status_code=404, detail=f"Model not found: {path}")
except RuntimeError as e:
    logger.exception("Failed to load model")
    raise HTTPException(status_code=500, detail="Model loading failed")
```

### Pydantic Models

- Use Pydantic v2 syntax (this project uses `pydantic>=2.0.0`)
- Add `Config` class or use `model_config` for configuration
- Use field validators when needed

```python
from pydantic import BaseModel, Field, field_validator

class SpeechRequest(BaseModel):
    model: str = Field(default="tts-1", description="Model ID")
    input: str = Field(..., min_length=1, max_length=4096, description="Text to synthesize")
    voice: str = Field(default="default", description="Voice ID")
    normalization_options: Optional[NormalizationOptions] = Field(
        default=None, description="Text normalization settings"
    )
    
    @field_validator("input")
    @classmethod
    def validate_input(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Input text cannot be empty")
        return v.strip()
```

### Async Patterns

- Use `async/await` for I/O-bound operations
- Model inference should use async patterns properly

```python
from fastapi import APIRouter

router = APIRouter()

@router.post("/audio/speech")
async def generate_speech(request: SpeechRequest) -> Response:
    # Async service call
    audio = await tts_service.generate(request)
    return Response(content=audio, media_type="audio/wav")
```

## Important Notes

- **License**: Apache 2.0 (see LICENSE and NOTICE files)
- **Attribution**: Text normalization derived from remsky/kokoro-fastapi
- **Model weights**: Gitignored (use `models/download_models.py` to download)
- **Logging**: Use `loguru` instead of standard logging
- **CUDA**: Supports 12.6, 12.8, 13.0 (use appropriate install script)
- **Python**: 3.12 is strictly required
- **Code checks**: Run `uv run black --check api/ && uv run ruff check api/ && uv run mypy api/` before committing

## Configuration

Configuration is managed via `api/src/core/config.py` using Pydantic Settings:
- Environment variables are loaded from `.env` files
- All settings have sensible defaults
- Model paths are resolved relative to project root
- Copy `.env.example` to `.env` and customize as needed

## Dependencies

Key dependencies (see `pyproject.toml` for full list):
- `qwen-tts>=0.0.5` - Official Qwen3 TTS package (brings in PyTorch, Transformers)
- `fastapi>=0.100.0` - Web framework
- `pydantic>=2.0.0` - Data validation
- `inflect>=7.0.0` - Text normalization (number-to-words)
- `loguru>=0.7.0` - Logging

## Working with This Codebase

### Adding New Features

1. **Follow existing patterns** - Check similar features in the codebase
2. **Update documentation** - Add docs to `/docs/` directory
3. **Update AGENTS.md** - If adding significant new capabilities
4. **Run all checks** - Black, Ruff, and Mypy before committing

### Modifying Vendored Code

**DO NOT MODIFY** files in:
- `api/src/core/models/modeling_qwen3_tts.py`
- `api/src/core/models/configuration_qwen3_tts.py`
- `api/src/core/models/processing_qwen3_tts.py`
- `api/src/core/tokenizer_12hz/*`
- `api/src/core/tokenizer_25hz/*`
- `api/src/services/text_processing/normalizer.py`

These are third-party files. If changes are needed, wrap them in your own code or submit issues upstream.

### Adding Tests

Currently no test suite exists. When adding tests:
- Create `tests/` directory at project root
- Use pytest
- Follow arrange-act-assert pattern
- Mock external dependencies (HuggingFace, CUDA)
