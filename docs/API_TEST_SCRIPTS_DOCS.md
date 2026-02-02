# API Test Scripts Documentation

**Module Path**: `examples/`

**Last Updated**: 2026-02-02

## Overview

The API test scripts provide a standardized way to test the Qwen3 TTS FastAPI server endpoints and compare performance against direct model inference. These scripts mirror the official Qwen3-TTS batch/single inference tests but send HTTP requests to the API instead of using direct model calls.

## Test Scripts

```
examples/
├── test_model_base_small.py      # Base 0.6B model voice cloning tests
├── test_model_base_large.py      # Base 1.7B model voice cloning tests
├── test_model_custom_voice.py    # Custom voice generation tests
├── test_model_voice_design.py    # Voice design tests
└── __init__.py                   # Package documentation
```

## Key Features

### 1. API-Based Testing

**Purpose**: Test FastAPI endpoints instead of direct model inference.

**Benefits**:
- Validates HTTP API functionality
- Measures real-world API performance
- Tests complete request/response cycle
- Identifies HTTP-specific issues (timeouts, streaming, encoding)

**Comparison with Direct Inference**:
| Metric | Direct Model | API |
|--------|-------------|-----|
| Connection | Local function calls | HTTP requests |
| Overhead | Minimal | ~50-70% slower |
| Batch Support | Native batching | Sequential requests |
| Use Case | Maximum performance | Integration/compat |

### 2. Environment-Based Configuration

**Configuration Priority**:
1. `TTS_API_URL` environment variable (highest priority)
2. `HOST` and `PORT` from `.env` file
3. `HOST` and `PORT` from `.env.example` file
4. Default: `http://127.0.0.1:8000` (lowest priority)

**Example Configuration**:
```bash
# Option 1: Environment variable
export TTS_API_URL=http://192.168.1.100:8080
python examples/test_model_custom_voice.py

# Option 2: .env file
# Create .env with:
HOST=192.168.1.100
PORT=8080

# Option 3: Default (localhost:8000)
python examples/test_model_base_small.py
```

### 3. Automatic API URL Detection

All test scripts include `load_env_config()` and `get_api_url()` functions that:
- Parse `.env` and `.env.example` files
- Extract `HOST` and `PORT` settings
- Construct the full API URL
- Allow quick override via environment variable

## Test Scripts Reference

### test_model_base_small.py

**Purpose**: Test base 0.6B model voice cloning via API.

**Model**: `qwen3-tts-12hz-0.6b-base`

**Test Cases**:
1. **Case 1**: Prompt single + synth single (ICL mode)
2. **Case 1b**: Prompt single + synth single (xvec only mode)
3. **Case 2**: Prompt single + synth batch (ICL mode)
4. **Case 2b**: Prompt single + synth batch (xvec only mode)

**Key Functions**:
```python
def send_api_request(
    api_url: str,
    text: str | List[str],
    language: str | List[str],
    ref_audio_b64: str | None = None,
    ref_text: str | None = None,
    model: str = "qwen3-tts-12hz-0.6b-base",
    response_format: str = "wav",
) -> Tuple[List[np.ndarray], int, float]:
    """Send voice cloning request to API."""
```

**Usage**:
```bash
python examples/test_model_base_small.py

# With custom API URL
TTS_API_URL=http://localhost:8080 python examples/test_model_base_small.py
```

**Output**: Saves WAV files to `qwen3_tts_test_api_base_small_output_wav/`

### test_model_base_large.py

**Purpose**: Test base 1.7B model voice cloning via API.

**Model**: `qwen3-tts-12hz-1.7b-base`

**Test Cases**: Same as test_model_base_small.py but with 1.7B model.

**Key Differences**:
- Larger model (1.7B vs 0.6B parameters)
- Higher quality voice cloning
- Slower generation (~2x inference time)
- More VRAM required (~8GB vs ~4GB)

**Usage**:
```bash
python examples/test_model_base_large.py
```

**Output**: Saves WAV files to `qwen3_tts_test_api_base_large_output_wav/`

### test_model_custom_voice.py

**Purpose**: Test custom voice generation with predefined speakers via API.

**Model**: `models/custom-voice` (0.6B)

**Test Cases**:
1. **Single with instruct**: One text with voice style instruction
2. **Batch**: Two texts with different speakers and instructions

**Supported Speakers**:
- `aiden`, `dylan`, `eric`, `ono_anna`
- `ryan`, `serena`, `sohee`, `uncle_fu`, `vivian`

**Key Functions**:
```python
def send_custom_voice_request(
    api_url: str,
    text: str,
    language: str,
    speaker: str,
    instruct: str | None = None,
    model: str = "models/custom-voice",
    response_format: str = "wav",
) -> Tuple[List[np.ndarray], int, float]:
    """Send custom voice request to API."""
```

**Batch Handling**:
Unlike direct model inference, the API does **not** support native batching. The batch test sends separate HTTP requests for each item and combines results:

```python
# Batch test implementation
for i in range(len(texts)):
    wavs, sr, elapsed = send_custom_voice_request(
        text=texts[i],
        language=languages[i],
        speaker=speakers[i],
        instruct=instructs[i],
    )
    all_wavs.extend(wavs)
    total_time += elapsed
```

**Usage**:
```bash
python examples/test_model_custom_voice.py
```

**Output**: Saves WAV files to `qwen3_tts_test_api_custom_voice_output_wav/`

### test_model_voice_design.py

**Purpose**: Test voice design generation via API.

**Model**: `qwen3-tts-12hz-1.7b-voice-design`

**Test Cases**:
1. **Single**: One text with natural language voice description
2. **Batch**: Two texts with different language-specific voice designs

**Key Functions**:
```python
def send_voice_design_request(
    api_url: str,
    text: str,
    language: str,
    instruct: str,
    model: str = "qwen3-tts-12hz-1.7b-voice-design",
    response_format: str = "wav",
) -> Tuple[List[np.ndarray], int, float]:
    """Send voice design request to API."""
```

**Instruction Examples**:
```python
# Chinese - Cute female voice
"体现撒娇稚嫩的萝莉女声，音调偏高且起伏明显"

# English - Incredulous tone
"Speak in an incredulous tone, with a hint of panic"
```

**Usage**:
```bash
python examples/test_model_voice_design.py
```

**Output**: Saves WAV files to `qwen3_tts_test_api_voice_design_output_wav/`

## Common Configuration

### Request Payload Structure

All test scripts use the same OpenAI-compatible request format:

```python
payload = {
    "model": model_id,           # e.g., "models/custom-voice"
    "input": text,               # Text to synthesize
    "voice": voice_id,           # e.g., "vivian"
    "response_format": "wav",    # Audio format
    "language": language,        # e.g., "Chinese"
    "streaming_mode": "full",    # Always "full" for non-streaming
}

# Optional fields
payload["instruct"] = instruction     # Voice style (custom voice/design)
payload["ref_audio"] = base64_audio   # Reference audio (voice clone)
payload["ref_text"] = reference_text  # Reference text (voice clone)
```

### Debug Logging

Tests include optional debug logging that shows:
- Request endpoint and payload
- Response status and headers
- Content-Type and Transfer-Encoding
- Audio data size and parsing details

Enable by checking the test output - debug lines are prefixed with `[DEBUG]`.

## Performance Comparison

### Single Request Overhead

Direct model inference vs API (measured with custom voice):

| Metric | Direct | API | Overhead |
|--------|--------|-----|----------|
| Single request | 3.68s | 5.67s | +54% |
| Batch (2 items) | 3.80s | 6.60s | +73% |

**Overhead Sources**:
1. HTTP round-trip (~100-300ms)
2. JSON serialization/deserialization
3. FastAPI/uvicorn processing
4. Audio encoding (WAV header)
5. Model loading on first request (if not cached)

### Model Caching Impact

With model caching enabled (default):

| Request | Time | Notes |
|---------|------|-------|
| First request | ~5-6s | Model loading + generation |
| Second request | ~4s | Model cached, just generation |
| Third+ requests | ~2-3s | Fully optimized |

**Recommendation**: Make a warmup request before benchmarking:
```bash
# Warmup
python examples/test_model_custom_voice.py

# Now run actual benchmarks
python examples/test_model_custom_voice.py
```

## Error Handling

### Common Errors

**1. Connection Broken / InvalidChunkLength**
```
ERROR: ("Connection broken: InvalidChunkLength...)
```

**Cause**: Client using streaming mode with non-streaming response.

**Solution**: All tests now use `streaming_mode: "full"` which returns complete responses.

**2. Speaker Not Supported**
```
Speaker 'vivian | ryan' not supported
```

**Cause**: Attempting to batch multiple speakers in one request.

**Solution**: Send separate requests for each speaker (implemented in test scripts).

**3. Model Not Found**
```
Failed to load model: models/custom-voice
```

**Cause**: Model files not downloaded or incorrect path.

**Solution**: Download models using `models/download_models.py`.

### Debug Mode

Enable detailed logging to diagnose issues:

```python
# In test scripts, these lines are already present:
print(f"[DEBUG] Sending request to: {endpoint}")
print(f"[DEBUG] Status code: {response.status_code}")
print(f"[DEBUG] Response content length: {len(response.content)} bytes")
```

## Dependencies

Required packages (included in `pyproject.toml` dev dependencies):

```toml
[project.optional-dependencies]
dev = [
    "requests",              # HTTP client
    "soundfile>=0.12.0",     # Audio file I/O
    "numpy>=1.24.0",         # Array operations
    "types-requests>=2.31.0", # Type stubs
]
```

Install with:
```bash
uv pip install -e ".[dev]"
```

## Usage Workflow

### 1. Start the Server

```bash
# Development mode
uv run uvicorn api.main:app --reload

# Production mode
uv run uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### 2. Configure Environment (Optional)

```bash
# Copy example config
cp .env.example .env

# Edit .env
HOST=0.0.0.0
PORT=8000
```

### 3. Run Tests

```bash
# Test base models
python examples/test_model_base_small.py
python examples/test_model_base_large.py

# Test custom voice
python examples/test_model_custom_voice.py

# Test voice design
python examples/test_model_voice_design.py
```

### 4. Check Output

```bash
# List generated audio files
ls qwen3_tts_test_api_*_output_wav/

# Play audio
ffplay qwen3_tts_test_api_custom_voice_output_wav/qwen3_tts_test_api_custom_single.wav
```

## Comparison with Direct Tests

### Direct Model Tests

**Location**: `examples/test_model_12hz_*.py`

**Characteristics**:
- Direct PyTorch model calls
- No HTTP overhead
- Native batching support
- CUDA synchronization for accurate timing

**Use Case**: Maximum performance, benchmarking model inference speed.

### API Tests (This Document)

**Characteristics**:
- HTTP requests to FastAPI server
- Real-world API performance
- OpenAI-compatible endpoints
- Sequential batch processing

**Use Case**: Integration testing, API validation, end-to-end testing.

### Performance Summary

| Test Type | Single (0.6B) | Batch (0.6B) | Single (1.7B) |
|-----------|--------------|--------------|---------------|
| Direct | ~3.7s | ~3.8s | ~7-8s |
| API | ~5.7s | ~6.6s | ~10-12s |
| Overhead | +54% | +73% | +50% |

## Future Enhancements

1. **True Batching**: Add API endpoint that accepts multiple texts in one request
2. **Streaming Support**: Add tests for streaming audio responses
3. **Base64 Endpoint**: Add tests for `/v1/audio/speech/base64` endpoint
4. **Performance Profiling**: Add detailed timing breakdowns (network, serialization, inference)
5. **Concurrent Requests**: Add tests for parallel API calls
6. **WebSocket Tests**: Add real-time streaming tests

## Troubleshooting

### Server Not Responding

```bash
# Check if server is running
curl http://localhost:8000/health

# Should return:
{"status": "healthy", "timestamp": 1234567890.123}
```

### Model Loading Errors

```bash
# Check model paths
ls models/

# Download if missing
python models/download_models.py
```

### Timeout Errors

Increase timeout in test scripts:
```python
response = requests.post(
    endpoint,
    json=payload,
    timeout=600,  # Increase from 300
)
```

### Slow Performance

Check cleanup settings in `.env`:
```bash
# Disable auto-unload for testing
IDLE_TIMEOUT=3600
CLEANUP_ENABLED=false
```

## Code Quality

All test scripts pass:
- **Black** formatting (line length: 88)
- **Ruff** linting
- **MyPy** type checking

Run checks:
```bash
uv run black examples/
uv run ruff check examples/
uv run mypy examples/
```
