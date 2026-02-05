# TTS Backend Module Documentation

**Module Path**: `api/src/inference/qwen3_tts_backend.py`

**Last Updated**: 2026-02-05

## Overview

The Qwen3TTSBackend provides a unified interface for all three Qwen3 TTS generation modes. It abstracts model management, audio processing, and streaming generation behind a clean async API.

## Seed Parameter Support

All generation methods support the `seed` parameter for reproducible generation:

**Implementation**:
- `_set_seed(seed)`: Internal function that sets seeds for Python's random NumPy, and PyTorch
- Seed validation: 0 to 2^32-1 (32-bit unsigned integer range)
- Stream-level consistency: Seed is applied per generation call to ensure consistent voice characteristics across streaming chunks

**Behavior**:
- When `seed=None`: Uses default behavior (random initialization)
- When `seed` is set: Ensures identical voice characteristics (emotion, tone, speed) for the same input
- Service layer integration: Service auto-generates a seed per request if not provided, ensuring consistency across streaming chunks

**Example**:
```python
# Reproducible voice cloning
async for chunk in backend.voice_clone(
    text="Hello world",
    model_path="models/base",
    ref_audio="ref.wav",
    ref_text="Original text",
    language="en",
    seed=42  # Consistent voice across generations
):
    yield chunk.data

# Reproducible custom voice
async for chunk in backend.custom_voice(
    text="Welcome",
    model_path="models/custom-voice",
    speaker="Vivian",
    language="en",
    seed=12345
):
    yield chunk.data
```

## Module Structure

```
api/src/inference/
├── qwen3_tts_backend.py          # Main backend implementation
├── qwen3_tts_model_manager.py    # Model lifecycle management
├── qwen3_tts_model.py            # Model wrapper (existing)
└── qwen3_tts_tokenizer.py        # Audio tokenizer (existing)
```

## Key Components

### 1. AudioChunk (Dataclass)

**Purpose**: Represents a segment of audio in streaming contexts.

**Attributes**:
- `data`: Audio content (numpy array for PCM, bytes for encoded)
- `sample_rate`: Sample rate in Hz (typically 24000)
- `is_last`: Boolean flag for final chunk
- `format`: Format identifier ("pcm", "wav", "mp3", etc.)
- `timestamp_ms`: Position in audio stream for progress tracking

**Design Rationale**:
The AudioChunk abstraction enables:
1. **Streaming responses**: Audio is yielded as it's generated
2. **Format flexibility**: Same interface for raw and encoded audio
3. **Progress tracking**: timestamp_ms allows clients to track playback position
4. **End detection**: is_last flag signals stream completion

**Example**:
```python
async for chunk in backend.voice_clone(text="Hello", model_path="models/base", ref_audio="ref.wav"):
    if chunk.is_last:
        print("Generation complete!")
    print(f"Generated {len(chunk.data)} samples at {chunk.timestamp_ms}ms")
```

### 2. Qwen3TTSBackend

**Purpose**: Unified interface for all TTS operations.

**Architecture**:
```
┌─────────────────────────────────────────────────────────┐
│                    Qwen3TTSBackend                       │
├─────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ Voice Clone  │  │ Custom Voice │  │ Voice Design │  │
│  │   (Base)     │  │(CustomVoice) │  │(VoiceDesign) │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  │
│         │                 │                 │          │
│         └─────────────────┼─────────────────┘          │
│                           │                            │
│              ┌────────────┴────────────┐               │
│              │   Qwen3ModelManager      │               │
│              │  (Lazy Loading & Cache)  │               │
│              └──────────────────────────┘               │
└─────────────────────────────────────────────────────────┘
```

## Generation Methods

### voice_clone()

**Purpose**: Clone a voice from reference audio using the Base model.

**Modes**:
1. **X-Vector Only** (`x_vector_only_mode=True`):
   - Uses only speaker embedding
   - Faster inference
   - Less accurate voice matching
   - No reference text needed

2. **ICL Mode** (`x_vector_only_mode=False`):
   - Uses In-Context Learning
   - Requires reference text
   - Better voice quality and consistency
   - More accurate prosody matching

**Parameters**:
- `text`: Text to synthesize
- `model_path`: Path to base model
- `ref_audio`: Reference audio (path, URL, or base64)
- `ref_text`: Reference transcript (required for ICL)
- `language`: Target language
- `x_vector_only_mode`: Mode selection
- `seed`: Random seed (0 to 2^32-1) for reproducible generation
- `**generation_kwargs`: Generation hyperparameters

**Usage Example**:
```python
# ICL mode (recommended)
chunks = backend.voice_clone(
    text="Hello, this is my cloned voice!",
    model_path="models/base",
    ref_audio="my_voice_sample.wav",
    ref_text="Original sample text",
    language="en",
    temperature=0.9,
    top_p=1.0,
)

# With reproducible generation
chunks = backend.voice_clone(
    text="Hello, this is my cloned voice!",
    model_path="models/base",
    ref_audio="my_voice_sample.wav",
    ref_text="Original sample text",
    language="en",
    seed=42,  # Consistent voice characteristics across generations
    temperature=0.9,
)

async for chunk in chunks:
    audio_data = chunk.data  # numpy array
```

### custom_voice()

**Purpose**: Generate speech using predefined speaker voices (CustomVoice model).

**Features**:
- Consistent speaker identity across generations
- Optional instruction for style control (1.7B models)
- Validation of speaker names against model capabilities

**Parameters**:
- `text`: Text to synthesize
- `model_path`: Path to custom voice model
- `speaker`: Speaker name (e.g., "Vivian", "Serena")
- `language`: Target language
- `instruct`: Optional style instruction
- `seed`: Random seed (0 to 2^32-1) for reproducible generation
- `**generation_kwargs`: Generation hyperparameters

**Usage Example**:
```python
chunks = backend.custom_voice(
    text="Welcome to the presentation!",
    model_path="models/custom-voice",
    speaker="Vivian",
    language="en",
    instruct="Speak with enthusiasm and energy",
)

# With reproducible generation
chunks = backend.custom_voice(
    text="Welcome to the presentation!",
    model_path="models/custom-voice",
    speaker="Vivian",
    language="en",
    instruct="Speak with enthusiasm and energy",
    seed=42,  # Consistent voice characteristics across generations
)

async for chunk in chunks:
    process_audio(chunk.data)
```

### voice_design()

**Purpose**: Design custom voices from natural language descriptions (VoiceDesign model).

**Use Cases**:
- Creating unique voices without reference audio
- Describing voice characteristics (age, gender, tone, emotion)
- Experimental voice design

**Parameters**:
- `text`: Text to synthesize
- `model_path`: Path to voice design model
- `instruct`: Voice description (e.g., "A warm, friendly female voice in her 30s")
- `language`: Target language
- `seed`: Random seed (0 to 2^32-1) for reproducible generation
- `**generation_kwargs`: Generation hyperparameters

**Usage Example**:
```python
chunks = backend.voice_design(
    text="Hello, how can I help you today?",
    model_path="models/voice-design",
    instruct="A professional male voice with a calm, reassuring tone",
    language="en",
)

# With reproducible generation
chunks = backend.voice_design(
    text="Hello, how can I help you today?",
    model_path="models/voice-design",
    instruct="A professional male voice with a calm, reassuring tone",
    language="en",
    seed=42,  # Consistent voice characteristics across generations
)

async for chunk in chunks:
    play_audio(chunk.data)
```

## Audio Encoding

### encode_audio()

**Purpose**: Convert raw PCM audio to various formats.

**Supported Formats**:
- **PCM**: Raw 16-bit signed integers
- **WAV**: Standard uncompressed audio
- **FLAC**: Lossless compression
- **OGG**: Ogg Vorbis format
- **MP3**: Lossy compression (requires pydub)
- **AAC**: Advanced Audio Coding (requires pydub)
- **OPUS**: Opus codec (requires pydub)

**Implementation Notes**:
- Formats requiring pydub (MP3, AAC, OPUS) fall back to WAV if pydub unavailable
- Uses soundfile library for WAV, FLAC, OGG
- MP3/AAC/OPUS encoding involves WAV intermediate step

**Example**:
```python
# Generate audio
async for chunk in backend.custom_voice(...):
    if chunk.format == "pcm":
        # Encode to MP3
        mp3_bytes = backend.encode_audio(
            chunk.data, 
            chunk.sample_rate, 
            format="mp3"
        )
        save_to_file(mp3_bytes, "output.mp3")
```

## Lifecycle Management

### Initialization

```python
backend = Qwen3TTSBackend(device="cuda", idle_timeout=300)
await backend.initialize(tokenizer_path="models/tokenizer")
```

**Why Separate initialize()?**
- Allows async initialization in FastAPI startup
- Tokenizer loading is expensive (done once)
- Separates configuration from initialization

### Shutdown

```python
await backend.shutdown()
```

**Actions**:
- Unloads all cached models
- Clears tokenizer
- Frees GPU memory

**Integration with FastAPI**:
```python
@app.on_event("startup")
async def startup():
    app.state.tts_backend = Qwen3TTSBackend()
    await app.state.tts_backend.initialize()

@app.on_event("shutdown")
async def shutdown():
    await app.state.tts_backend.shutdown()
```

## Model Type Detection

The backend automatically detects model type from path:

```python
def _get_model_type_from_path(self, model_path: str) -> VoiceModelType:
    path_lower = model_path.lower()
    if "voice-design" in path_lower:
        return VoiceModelType.VOICE_DESIGN
    elif "custom-voice" in path_lower:
        return VoiceModelType.CUSTOM_VOICE
    else:
        return VoiceModelType.BASE
```

**Path Conventions**:
- `*/base*`: Base model (voice clone)
- `*/custom-voice*`: CustomVoice model
- `*/voice-design*`: VoiceDesign model

## Error Handling

### Exception Hierarchy

```
RuntimeError
├── Model loading failures
├── Generation failures
└── Unexpected errors

ValueError
├── Invalid parameters
├── Unsupported speakers
└── Missing required arguments
```

### Error Recovery

All generation methods follow this pattern:
```python
try:
    model = self.model_manager.load_model(...)
    # Generate...
except Exception as e:
    logger.error(f"Generation failed: {e}")
    raise RuntimeError(f"... failed: {e}") from e
finally:
    self.model_manager.release_model(...)  # Always release
```

**Key Features**:
- Always releases model reference (even on failure)
- Logs detailed error information
- Wraps exceptions with context
- Preserves original exception chain

## Performance Considerations

### Memory Management

**Per-Model Memory**:
- 0.6B models: ~4GB GPU memory
- 1.7B models: ~8GB GPU memory

**Optimization Strategies**:
1. Use `idle_timeout` to auto-unload unused models
2. Share models between concurrent requests (reference counting)
3. Use bfloat16 for 50% memory reduction vs float32
4. Cleanup idle models periodically

### Latency Breakdown

**First Request** (cold start):
- Model loading: 5-30 seconds
- Generation: 1-5 seconds (depends on text length)
- Total: 6-35 seconds

**Subsequent Requests** (hot):
- Cache hit: ~1ms
- Generation: 1-5 seconds
- Total: 1-5 seconds

### Throughput

**Single Model**:
- 0.6B: ~10-20 concurrent requests
- 1.7B: ~5-10 concurrent requests

**Recommendation**: Use multiple workers with separate model caches for high throughput.

## Testing Strategy

### Unit Tests

1. **Generation Methods**:
   - Test each mode with minimal inputs
   - Verify AudioChunk structure
   - Check error handling

2. **Audio Encoding**:
   - Round-trip encoding/decoding
   - Format validation
   - Fallback behavior

3. **Model Type Detection**:
   - Path parsing accuracy
   - Edge cases

### Integration Tests

1. **Full Pipeline**:
   - End-to-end generation for each mode
   - Multiple sequential requests
   - Concurrent request handling

2. **Memory Management**:
   - Model loading/unloading
   - Reference counting accuracy
   - Idle timeout behavior

3. **Error Scenarios**:
   - Invalid model paths
   - Missing reference audio
   - Unsupported speakers

## Integration Points

### Service Layer

The backend is used by Qwen3TTSService:
```python
class Qwen3TTSService:
    def __init__(self, backend: Qwen3TTSBackend):
        self.backend = backend
    
    async def generate(self, request: OpenAISpeechRequest):
        # Map request to appropriate backend method
        if request.ref_audio:
            chunks = self.backend.voice_clone(...)
        elif request.speaker:
            chunks = self.backend.custom_voice(...)
        else:
            chunks = self.backend.voice_design(...)
```

### Router Layer

Endpoints use the backend through the service:
```python
@router.post("/audio/speech")
async def generate_speech(request: OpenAISpeechRequest):
    chunks = []
    async for chunk in service.generate(request):
        chunks.append(chunk)
    return encode_response(chunks)
```

## Future Enhancements

1. **True Streaming**: Current implementation buffers all audio before yielding. True streaming would yield partial results during generation.

2. **Voice Mixing**: Blend multiple reference voices for hybrid voices.

3. **Prosody Control**: Fine-grained control over pitch, speed, and intonation.

4. **Batch Processing**: Process multiple texts in parallel for efficiency.

5. **Caching**: Cache generated audio for identical requests.

6. **Format Optimization**: Hardware-accelerated encoding (GPU MP3 encoding).
