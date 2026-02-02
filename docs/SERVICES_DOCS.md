# Service Layer Documentation

**Module Path**: `api/src/services/qwen3_tts_service.py`

**Last Updated**: 2026-02-01

## Overview

The Qwen3TTSService provides a high-level business logic layer that bridges OpenAI-compatible API requests with the underlying TTS backend. It handles request validation, voice/model mapping, and response formatting.

## Module Structure

```
api/src/services/
├── __init__.py                   # Package initialization
└── qwen3_tts_service.py          # Main service implementation
```

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Qwen3TTSService                            │
├──────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐  ┌──────────────────┐                   │
│  │ Request Handling │  │ Response Handling│                   │
│  │                  │  │                  │                   │
│  │ • Validation     │  │ • Audio encoding │                   │
│  │ • Mode detection │  │ • Base64 encoding│                   │
│  │ • Routing        │  │ • Chunk assembly │                   │
│  └────────┬─────────┘  └────────┬─────────┘                   │
│           │                     │                             │
│  ┌────────▼─────────────────────▼─────────┐                   │
│  │         Mapping & Resolution           │                   │
│  │                                        │                   │
│  │  • OpenAI model → Qwen3 model path     │                   │
│  │  • OpenAI voice → Qwen3 speaker        │                   │
│  └──────────────────┬─────────────────────┘                   │
│                     │                                         │
│         ┌───────────▼────────────┐                            │
│         │   Qwen3TTSBackend      │                            │
│         │                        │                            │
│         │ • voice_clone()        │                            │
│         │ • custom_voice()       │                            │
│         │ • voice_design()       │                            │
│         └────────────────────────┘                            │
└──────────────────────────────────────────────────────────────┘
```

## Key Responsibilities

### 1. Request Translation

The service translates OpenAI API requests into Qwen3-specific operations:

**OpenAI → Qwen3 Mapping**:
```python
# Model IDs
"tts-1" → "models/custom-voice"      # 0.6B custom voice
"tts-1-hd" → "models/voice-design"   # 1.7B voice design

# Voice IDs
"alloy" → "Vivian"
"echo" → "Dylan"
"fable" → "Eric"
```

**Generation Mode Detection**:
```python
# Based on request parameters
if request.ref_audio:
    mode = VoiceModelType.BASE           # Voice cloning
elif request.speaker:
    mode = VoiceModelType.CUSTOM_VOICE   # Predefined speaker
elif request.instruct:
    mode = VoiceModelType.VOICE_DESIGN   # Voice design
```

### 2. Request Validation

Pre-flight checks before generation:

```python
async def validate_request(self, request: OpenAISpeechRequest) -> None:
    # 1. Verify model exists
    model_path = self._resolve_model(request.model)
    if not model_path:
        raise ValueError(f"Unknown model: {request.model}")
    
    # 2. Validate speaker compatibility
    supported = self.backend.get_supported_speakers(model_path)
    if speaker not in supported:
        raise ValueError(f"Speaker '{speaker}' not supported")
```

### 3. Audio Processing Pipeline

```
Request → Resolve Model/Voice → Determine Mode → Backend Generation 
    ↓
Audio Encoding → Chunk Assembly → Yield Response
```

**Format Support**:
- Input: Always raw PCM from backend
- Output: MP3, WAV, OGG, FLAC, AAC, OPUS, PCM
- Automatic format conversion via backend.encode_audio()

## API Reference

### Constructor

```python
Qwen3TTSService(backend: Qwen3TTSBackend)
```

**Dependencies**:
- Requires initialized Qwen3TTSBackend
- Loads mappings from `api/src/core/openai_mappings.json`

### Core Methods

#### generate_speech()

**Purpose**: Main entry point for speech generation.

**Signature**:
```python
async def generate_speech(
    self,
    request: OpenAISpeechRequest,
) -> AsyncGenerator[AudioChunk, None]
```

**Processing Flow**:
1. Resolve model path from request.model
2. Determine generation mode from request parameters
3. Map OpenAI voice to Qwen3 speaker
4. Normalize input text for better pronunciation
5. Apply text segmentation based on streaming mode
6. Call appropriate backend method for each text segment
7. Encode audio chunks to requested format
8. Yield formatted AudioChunk objects

**Streaming Support**:
The service supports three streaming modes controlled by `request.streaming_mode`:

- **FULL**: Generates complete audio for the entire text before yielding any chunks
  - Lowest latency for short texts
  - Best quality across sentence boundaries
  
- **SENTENCE**: Splits text by sentence boundaries and streams each as completed
  - Default mode (StreamingMode.SENTENCE)
  - Natural streaming experience with smart sentence detection
  - Handles abbreviations, decimals, CJK punctuation
  
- **CHUNK**: Splits text by fixed character count with overlap
  - Customizable chunk sizes via `request.chunk_size`
  - Overlap ensures smooth audio transitions
  - Best for very long texts

**Example**:
```python
# Sentence-level streaming (default)
request = OpenAISpeechRequest(
    model="tts-1",
    input="First sentence. Second sentence. Third sentence.",
    voice="alloy",
    response_format="mp3",
    streaming_mode=StreamingMode.SENTENCE
)

async for chunk in service.generate_speech(request):
    # chunk.data contains MP3-encoded bytes
    # chunk.is_last indicates final chunk
    save_to_file(chunk.data)

# Fixed-size chunk streaming
chunk_request = OpenAISpeechRequest(
    model="tts-1",
    input="This is a very long text...",
    voice="alloy", 
    response_format="mp3",
    streaming_mode=StreamingMode.CHUNK,
    chunk_size=200  # 200 characters per chunk
)
```

**Performance Characteristics**:
- **FULL mode**: First chunk latency = total generation time
- **SENTENCE mode**: First chunk latency = first sentence generation time
- **CHUNK mode**: First chunk latency = first chunk generation time
- Subsequent chunks: Streamed as available
- Total time: Depends on text length, model size, and streaming mode

#### get_available_models()

**Purpose**: List available TTS models in OpenAI format.

**Returns**:
```python
[
    {
        "id": "tts-1",
        "object": "model",
        "created": 1700000000,
        "owned_by": "qwen3-tts"
    },
    ...
]
```

#### get_available_voices()

**Purpose**: List available voices with metadata.

**Returns**: List of VoiceInfo objects containing voice_id, name, and optional preview_url.

#### validate_request()

**Purpose**: Pre-flight validation without generation.

**Validations**:
- Model existence
- Voice/speaker compatibility
- Parameter constraints

## Mapping System

### Voice Mappings

Loaded from `openai_mappings.json`:

```json
{
    "voices": {
        "alloy": "Vivian",
        "ash": "Serena",
        "coral": "Uncle_Fu",
        "echo": "Dylan",
        "fable": "Eric",
        "onyx": "Ryan",
        "nova": "Aiden",
        "sage": "Ono_Anna",
        "shimmer": "Sohee"
    }
}
```

**Fallback Behavior**:
If mappings file is missing, uses hardcoded defaults for common voices.

### Model Mappings

```json
{
    "models": {
        "tts-1": "qwen3-tts-12hz-0.6b-custom-voice",
        "tts-1-hd": "qwen3-tts-12hz-1.7b-voice-design"
    }
}
```

**Resolution Priority**:
1. Check if model_id is a direct path (starts with "models/" or contains "/")
2. Check mappings dictionary
3. Resolve via settings.get_model_path()
4. Fallback to default model

## Generation Mode Logic

### Mode Detection Algorithm

```python
def _determine_generation_mode(self, request: OpenAISpeechRequest) -> VoiceModelType:
    # Priority 1: Voice cloning (requires reference audio)
    if request.ref_audio:
        return VoiceModelType.BASE
    
    # Priority 2: Explicit speaker selection
    elif request.speaker:
        return VoiceModelType.CUSTOM_VOICE
    
    # Priority 3: Voice design instruction
    elif request.instruct:
        return VoiceModelType.VOICE_DESIGN
    
    # Default: Custom voice with mapped speaker
    else:
        return VoiceModelType.CUSTOM_VOICE
```

### Mode-Specific Handling

**Voice Clone Mode**:
- Requires `ref_audio` parameter
- Optionally uses `ref_text` for ICL mode
- Uses Base model regardless of requested model

**Custom Voice Mode**:
- Maps `voice` to `speaker`
- Validates speaker against model capabilities
- Supports optional instruction (1.7B models)

**Voice Design Mode**:
- Requires `instruct` parameter
- Creates voice from text description
- Most flexible but slower

## Audio Format Handling

### Format Conversion Pipeline

```
Raw PCM (from backend)
    ↓
encode_audio(format="mp3")  # or wav, ogg, etc.
    ↓
Encoded bytes
    ↓
Yield as AudioChunk
```

### Supported Formats

| Format | Encoding | Notes |
|--------|----------|-------|
| MP3    | Lossy    | Requires pydub, smallest size |
| WAV    | Lossless | Largest size, fastest decode |
| OGG    | Lossy    | Good compression |
| FLAC   | Lossless | Compressed lossless |
| AAC    | Lossy    | Requires pydub |
| OPUS   | Lossy    | Low latency, requires pydub |
| PCM    | Raw      | No header, raw samples |

### Speed Adjustment

Maps OpenAI `speed` parameter to generation temperature:

```python
temperature = max(0.1, min(1.0, 0.9 / request.speed))
```

- speed=1.0 → temperature=0.9 (normal)
- speed=2.0 → temperature=0.45 (faster/colder)
- speed=0.5 → temperature=1.0 (slower/warmer)

## Error Handling

### Error Categories

**Validation Errors** (ValueError):
- Unknown model ID
- Unsupported voice/speaker
- Invalid request parameters

**Generation Errors** (RuntimeError):
- Model loading failures
- Generation failures
- Resource exhaustion

### Error Response Strategy

```python
try:
    async for chunk in self.generate_speech(request):
        yield chunk
except ValueError as e:
    # Return 400 Bad Request
    raise HTTPException(status_code=400, detail=str(e))
except RuntimeError as e:
    # Return 500 Internal Server Error
    raise HTTPException(status_code=500, detail=str(e))
```

## Integration Points

### Router Layer

```python
@router.post("/audio/speech")
async def speech_endpoint(
    request: OpenAISpeechRequest,
    service: Qwen3TTSService = Depends(get_service)
):
    # Service handles all translation
    chunks = []
    async for chunk in service.generate_speech(request):
        chunks.append(chunk)
    return assemble_response(chunks)
```

### Backend Layer

Service delegates all model operations to backend:
```python
# Service determines mode and calls appropriate backend method
if mode == VoiceModelType.BASE:
    chunks = self.backend.voice_clone(...)
elif mode == VoiceModelType.CUSTOM_VOICE:
    chunks = self.backend.custom_voice(...)
```

### Configuration Layer

Uses settings for model path resolution:
```python
from api.src.core.config import settings

model_path = settings.get_model_path(qwen3_model_id)
```

## Testing Considerations

### Unit Tests

1. **Mapping Tests**:
   - OpenAI model ID resolution
   - Voice name translation
   - Fallback behavior

2. **Mode Detection Tests**:
   - Correct mode for each parameter combination
   - Default mode selection

3. **Validation Tests**:
   - Invalid model handling
   - Unsupported speaker detection

### Integration Tests

1. **End-to-End Flow**:
   - Request → Service → Backend → Response
   - Verify audio format correctness

2. **Error Scenarios**:
   - Missing mappings file
   - Backend initialization failure
   - Invalid request parameters

### Performance Tests

1. **Concurrent Requests**:
   - Multiple simultaneous generations
   - Resource contention handling

2. **Memory Management**:
   - Service doesn't leak memory
   - Backend cleanup on errors

## Design Decisions

### 1. Async-First Design

All methods are async to support:
- Non-blocking I/O
- Concurrent request handling
- Streaming responses

### 2. Separation of Concerns

Service layer responsibilities:
- ✅ Request translation
- ✅ Validation
- ✅ Format conversion
- ❌ Model management (delegated to backend)
- ❌ Audio generation (delegated to backend)

### 3. Immutable Requests

Service never modifies the request object:
```python
# Create new variables instead of modifying request
speaker = self._resolve_voice(request.voice)
# NOT: request.voice = self._resolve_voice(request.voice)
```

### 4. Fail-Fast Validation

Validation happens before any expensive operations:
```python
await self.validate_request(request)  # Fast checks first
async for chunk in self.generate_speech(request):  # Expensive operation
    ...
```

## Future Enhancements

1. **Request Caching**: Cache generated audio for identical requests
2. **Batch API**: Support batch generation for multiple texts
3. **Voice Preview**: Generate short voice previews
4. **Streaming Improvements**: True streaming with partial results
5. **Metrics Collection**: Track generation times, success rates
6. **Rate Limiting**: Built-in request throttling
7. **A/B Testing**: Support multiple model versions
