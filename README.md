> [!IMPORTANT]
> **Disclaimer:** This entire project has been vibe-coded. Use at your own risk.

# Qwen3-TTS-FastAPI

An OpenAI-compatible Text-to-Speech (TTS) API server for the [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) model series, heavily inspired by [kokoro-fastapi](https://github.com/remsky/kokoro-fastapi). This project provides a high-performance interface with advanced VRAM management and text normalization.

## Key Features

- **OpenAI Compatibility**: Drop-in replacement for the `/v1/audio/speech` endpoint.
- **Smart VRAM Management**:
    - **Lazy Loading**: Models are only loaded when requested.
    - **Auto-Unload**: Frees up GPU resources after a configurable idle timeout (default: 10 mins).
    - **Reference Counting**: Prevents unloading while requests are active.
- **Advanced Text Normalization**: Integrated normalization pipeline (derived from `kokoro-fastapi`) to handle numbers, currency, dates, and URLs correctly.
- **Multi-Model Support**: Supports Base, Custom Voice, and Voice Design variants (0.6B and 1.7B).
- **Flexible Deployment**: Scripts for various CUDA versions (12.6, 12.8, 13.0) and CPU fallback.

## Security Warning

**The admin endpoints are currently unauthenticated.**

The following endpoints are exposed for management purposes:
- `GET /admin/config`: Displays current server configuration (including paths).
- `POST /admin/cleanup`: Manually triggers VRAM cleanup.

## Installation

### Prerequisites
- Linux (Windows/WSL2 should work but is untested)
- Python 3.12 (Strict requirement)
- [uv](https://github.com/astral-sh/uv) package manager
- NVIDIA GPU with CUDA drivers (optional, but recommended)

### Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/icantc0de1/Qwen3-TTS-FastAPI.git
    cd Qwen3-TTS-FastAPI
    ```

2.  **Install dependencies:**
    Choose the installation script matching your CUDA version:

    ```bash
    # For CUDA 13.0
    ./scripts/install-cu130.sh

    # For CUDA 12.8
    ./scripts/install-cu128.sh

    # For CUDA 12.6
    ./scripts/install-cu126.sh

    # For CPU Only
    ./scripts/install-cpu.sh
    ```

3.  **Download Models:**
    Download the model weights to the `models/` directory. You can use the provided script or download manually.
    ```bash
    # Example using huggingface-cli (ensure it's installed)
    huggingface-cli download Qwen/Qwen2.5-Speech-12Hz-0.6B-Base --local-dir models/base
    ```
    *Note: The project supports custom model paths via configuration.*

## Configuration

Copy the example environment file and configure your settings:

```bash
cp .env.example .env
```

**Key Settings:**

| Variable | Default | Description |
|----------|---------|-------------|
| `IDLE_TIMEOUT` | `600` | Seconds before unloading an idle model (0 to disable) |
| `CLEANUP_INTERVAL` | `60` | How often to check for idle models (seconds) |
| `DEFAULT_DEVICE` | `cuda` | `cuda` or `cpu` |
| `DEFAULT_MODEL_SIZE`| `small` | `small` (0.6B) or `large` (1.7B) |

## Usage

### Development (Hot Reload)
```bash
uv run uvicorn api.main:app --reload
```

### Production
```bash
uv run uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## API Reference

### Speech Generation (OpenAI Compatible)
**POST** `/v1/audio/speech`

```json
{
  "model": "tts-1",
  "input": "Hello, this is a test of the Qwen3 TTS system.",
  "voice": "alloy"
}
```

### Management
- `GET /health`: Health check endpoint.
- `GET /admin/config`: View active configuration.
- `POST /admin/cleanup`: Force unload idle models.

## License & Acknowledgements

This project is licensed under the **Apache 2.0 License**.

**Attributions:**
- **Qwen3-TTS**: Core model architecture and weights by the [Alibaba Qwen Team](https://github.com/QwenLM/Qwen3-TTS).
- **Text Normalization**: The text normalization module (`api/src/services/text_processing/`) is derived from [kokoro-fastapi](https://github.com/remsky/kokoro-fastapi) by **remsky**.

See the `NOTICE` file for full attribution details.
