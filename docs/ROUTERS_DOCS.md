# Routers Module Documentation

**Module Path**: `api/src/routers/openai_compatible.py`

**Last Updated**: 2026-02-01

## Overview

The routers module provides OpenAI-compatible REST API endpoints for text-to-speech generation. It implements the standard OpenAI TTS API specification while supporting Qwen3-specific features.

## Module Structure

```
api/src/routers/
├── __init__.py                   # Package initialization
└── openai_compatible.py          # OpenAI-compatible endpoints
```

## API Endpoints

### 1. POST /v1/audio/speech

**Purpose**: Main text-to-speech generation endpoint (streaming).

**OpenAI Compatibility**: ✓ Full compatibility

**Request Body** (OpenAISpeechRequest):
```json
{
    "model": "tts-1",
    "input": "Hello, world!",
    "voice": "alloy",
    "response_format": "mp3",
    "speed": 1.0
}
```

**Response**: Streaming audio in requested format

**Headers**:
- `Content-Type`: Audio MIME type (audio/mpeg, audio/wav, etc.)
- `Content-Disposition`: Attachment with filename
- `X-Model`: Model ID used
- `X-Voice`: Voice ID used

**Streaming Implementation**:
```python
async def audio_stream() -> AsyncGenerator[bytes, None]:
    async for chunk in service.generate_speech(request):
        if isinstance(chunk.data, bytes):
            yield chunk.data
        else:
            # Encode numpy array
            encoded = service.backend.encode_audio(...)
            yield encoded
```

**Error Responses**:
- `400 Bad Request`: Invalid parameters (unknown model, unsupported voice)
- `500 Internal Server Error`: Generation failure
- `503 Service Unavailable`: TTS service not initialized

### 2. GET /v1/audio/models

**Purpose**: List available TTS models.

**OpenAI Compatibility**: ✓ Compatible format

**Response** (ModelsListResponse):
```json
{
    "object": "list",
    "data": [
        {
            "id": "tts-1",
            "object": "model",
            "created": 1700000000,
            "owned_by": "qwen3-tts"
        },
        {
            "id": "qwen3-tts-12hz-1.7b-voice-design",
            "object": "model",
            "created": 1700000000,
            "owned_by": "qwen3-tts"
        }
    ]
}
```

**Models Included**:
- OpenAI aliases: tts-1, tts-1-hd
- Native Qwen3 models: All variants (0.6B, 1.7B, base, custom-voice, voice-design)

### 3. GET /v1/audio/voices

**Purpose**: List available voices/speakers.

**OpenAI Compatibility**: ✓ Extended format

**Response** (VoicesListResponse):
```json
{
    "voices": [
        {
            "voice_id": "alloy",
            "name": "Vivian",
            "preview_url": null
        },
        {
            "voice_id": "echo",
            "name": "Dylan",
            "preview_url": null
        }
    ]
}
```

**Voice Mapping**:
OpenAI voice IDs are mapped to Qwen3 speaker names:
- alloy → Vivian
- ash → Serena
- coral → Uncle_Fu
- echo → Dylan
- fable → Eric
- onyx → Ryan
- nova → Aiden
- sage → Ono_Anna
- shimmer → Sohee

### 4. GET /v1/audio/voices/{voice_id}

**Purpose**: Get details about a specific voice.

**Parameters**:
- `voice_id`: Voice identifier (e.g., "alloy", "echo")

**Response** (VoiceInfo):
```json
{
    "voice_id": "alloy",
    "name": "Vivian",
    "preview_url": null
}
```

**Error Response**:
- `404 Not Found`: Voice not found

### 5. POST /v1/audio/speech/base64 (Non-Standard Extension)

**Purpose**: Generate speech and return as base64-encoded JSON.

**Why This Endpoint?**
Not all clients can handle streaming responses. This endpoint provides:
- Complete audio in single response
- Base64 encoding for easy JSON transport
- Metadata (duration, format, sample rate)

**Request**: Same as `/v1/audio/speech`

**Response**:
```json
{
    "audio": "base64encodedstring...",
    "format": "mp3",
    "sample_rate": 24000,
    "duration_ms": 1500
}
```

**Use Cases**:
- Mobile apps with limited streaming support
- Server-to-server API calls
- Debugging and testing
- Clients requiring synchronous responses

## Architecture

### Request Flow

```
Client Request
    ↓
FastAPI Router
    ↓
Pydantic Validation (OpenAISpeechRequest)
    ↓
Dependency Injection (get_tts_service)
    ↓
Qwen3TTSService
    ↓
Qwen3TTSBackend
    ↓
StreamingResponse / JSON Response
    ↓
Client
```

### Dependency Injection

The router uses FastAPI's dependency injection to access the TTS service:

```python
def get_tts_service(request: Request) -> Qwen3TTSService:
    service = getattr(request.app.state, "tts_service", None)
    if service is None:
        raise HTTPException(status_code=503, ...)
    return service

@router.post("/speech")
async def create_speech(
    request: OpenAISpeechRequest,
    service: Qwen3TTSService = Depends(get_tts_service),
):
    # Use injected service
    ...
```

**Benefits**:
- Service lifecycle managed by main app
- Easy testing with mock services
- Clean separation of concerns
- Automatic error handling for missing service

### Service Registration

The service is registered during app startup:

```python
# In main.py
@app.on_event("startup")
async def startup():
    app.state.tts_service = Qwen3TTSService(backend)
    await app.state.tts_service.initialize()
    
    # Register with router
    from api.src.routers import openai_compatible
    openai_compatible.set_tts_service(app.state.tts_service)
```

## Error Handling

### Error Hierarchy

```
HTTPException
├── 400 Bad Request
│   ├── Invalid model ID
│   ├── Unsupported voice
│   └── Validation errors
├── 404 Not Found
│   └── Voice not found
├── 500 Internal Server Error
│   ├── Generation failures
│   └── Unexpected errors
└── 503 Service Unavailable
    └── TTS service not initialized
```

### Error Response Format

```json
{
    "detail": "Error message describing what went wrong"
}
```

### Logging Strategy

All errors are logged with appropriate severity:
- `logger.info()`: Normal operations (requests, successes)
- `logger.warning()`: Validation errors (client errors)
- `logger.error()`: Server errors (generation failures)
- `logger.exception()`: Unexpected exceptions (with stack trace)

## Content Type Mapping

Audio formats are mapped to MIME types:

| Format | MIME Type |
|--------|-----------|
| mp3 | audio/mpeg |
| wav | audio/wav |
| ogg | audio/ogg |
| opus | audio/opus |
| aac | audio/aac |
| flac | audio/flac |
| pcm | audio/pcm |

## Streaming Strategy

### Why Streaming?

1. **Memory Efficiency**: Don't buffer entire audio in memory
2. **Low Latency**: Client receives first bytes while generating
3. **Scalability**: Handle concurrent requests without OOM

### Implementation Details

```python
async def audio_stream() -> AsyncGenerator[bytes, None]:
    chunk_count = 0
    total_bytes = 0
    
    async for chunk in service.generate_speech(request):
        chunk_count += 1
        
        # Convert to bytes if needed
        if isinstance(chunk.data, bytes):
            audio_bytes = chunk.data
        else:
            audio_bytes = service.backend.encode_audio(...)
        
        total_bytes += len(audio_bytes)
        yield audio_bytes
        
        if chunk.is_last:
            logger.info(f"Complete: {chunk_count} chunks, {total_bytes} bytes")

return StreamingResponse(
    audio_stream(),
    media_type=content_type,
    headers={...},
)
```

### Client-Side Handling

**Python (requests)**:
```python
response = requests.post(url, json=payload, stream=True)
with open("output.mp3", "wb") as f:
    for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)
```

**JavaScript (fetch)**:
```javascript
const response = await fetch(url, { method: 'POST', body: JSON.stringify(payload) });
const reader = response.body.getReader();
while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    // Process chunk
}
```

## Testing

### Manual Testing with curl

**Basic Request**:
```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "Hello, world!",
    "voice": "alloy",
    "response_format": "mp3"
  }' \
  --output output.mp3
```

**List Models**:
```bash
curl http://localhost:8000/v1/audio/models
```

**List Voices**:
```bash
curl http://localhost:8000/v1/audio/voices
```

**Base64 Endpoint**:
```bash
curl -X POST http://localhost:8000/v1/audio/speech/base64 \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "Hello!",
    "voice": "alloy"
  }' | jq -r '.audio' | base64 -d > output.mp3
```

## Security Considerations

### Input Validation

- All inputs validated via Pydantic models
- Text length limited to 4096 characters
- Speed constrained to 0.25x - 4.0x
- Format restricted to supported values

### Rate Limiting

Not implemented in this version. Recommended for production:
```python
from slowapi import Limiter

limiter = Limiter(key_func=lambda: request.client.host)

@router.post("/speech")
@limiter.limit("10/minute")
async def create_speech(...):
    ...
```

### Authentication

Not implemented in this version. Recommended for production:
```python
from fastapi.security import HTTPBearer

security = HTTPBearer()

@router.post("/speech")
async def create_speech(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    ...
):
    verify_token(credentials.credentials)
    ...
```

## Performance Optimization

### Connection Handling

- Use `keep-alive` connections for multiple requests
- Streaming prevents connection timeouts for long generations
- Appropriate buffer sizes (8192 bytes) for chunk transmission

### Response Compression

Not currently enabled. For production with non-streaming endpoints:
```python
from fastapi.middleware.gzip import GZipMiddleware

app.add_middleware(GZipMiddleware, minimum_size=1000)
```

## Future Enhancements

1. **WebSocket Support**: Real-time streaming with bidirectional communication
2. **Batch API**: Process multiple texts in single request
3. **Voice Preview Endpoint**: Generate short samples for voice selection
4. **SSML Support**: Speech Synthesis Markup Language for fine control
5. **Progress Callbacks**: SSE (Server-Sent Events) for generation progress
6. **Caching Layer**: Redis caching for identical requests
7. **Metrics Endpoint**: Prometheus metrics for monitoring
