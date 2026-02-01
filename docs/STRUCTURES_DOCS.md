# Structures Module Documentation

**Module Path**: `api/src/structures/`

**Last Updated**: 2026-02-01

## Overview

The structures module defines all Pydantic data models and schemas used throughout the Qwen3 TTS API. These schemas ensure type safety, validation, and OpenAI API compatibility.

## Module Structure

```
api/src/structures/
├── __init__.py          # Package exports
└── schemas.py           # All Pydantic models and schemas
```

## Key Components

### 1. VoiceModelType (Enum)

**Purpose**: Defines the three supported Qwen3 TTS model types.

**Values**:
- `BASE` ("base"): Voice cloning from reference audio
  - Requires reference audio input
  - Supports x-vector only mode or ICL (In-Context Learning) mode
  - Best for cloning specific voices from samples
  
- `VOICE_DESIGN` ("voice_design"): Natural language voice design
  - Uses text instructions to design voices
  - No reference audio needed
  - Best for creating custom voices from descriptions
  
- `CUSTOM_VOICE` ("custom_voice"): Predefined speaker voices
  - Uses predefined speaker IDs
  - Optional instruction for style control
  - Best for consistent, named voices

**Usage**:
```python
from api.src.structures.schemas import VoiceModelType

model_type = VoiceModelType.CUSTOM_VOICE
if model_type == VoiceModelType.VOICE_DESIGN:
    # Handle voice design logic
    pass
```

### 2. OpenAISpeechRequest (Pydantic Model)

**Purpose**: Validates and structures OpenAI-compatible TTS requests.

**Key Features**:
- OpenAI API compatibility (model, input, voice, response_format, speed)
- Qwen3-specific extensions (language, speaker, instruct, ref_audio, ref_text)
- Automatic validation of input text (non-empty, max 4096 chars)
- Speed validation (0.25x to 4.0x range)

**Fields**:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| model | str | "tts-1" | Model identifier |
| input | str | Required | Text to synthesize (1-4096 chars) |
| voice | str | "alloy" | Voice/speaker identifier |
| response_format | Literal | "mp3" | Output format (mp3, wav, opus, etc.) |
| speed | float | 1.0 | Speed multiplier (0.25-4.0) |
| language | Optional[str] | None | Language code (en, zh, ja, etc.) |
| speaker | Optional[str] | None | Speaker name (custom_voice models) |
| instruct | Optional[str] | None | Voice design instruction |
| ref_audio | Optional[str] | None | Base64 reference audio |
| ref_text | Optional[str] | None | Reference text for ICL mode |

**Validation Logic**:
- `input`: Strips whitespace, validates non-empty after stripping
- `speed`: Validates 0.25 <= speed <= 4.0

**Usage Example**:
```python
from api.src.structures.schemas import OpenAISpeechRequest

# Valid request
request = OpenAISpeechRequest(
    model="tts-1",
    input="Hello, world!",
    voice="alloy",
    response_format="mp3"
)

# Access validated data
print(request.input)  # "Hello, world!"
print(request.speed)  # 1.0

# Invalid request raises ValidationError
try:
    request = OpenAISpeechRequest(input="   ")  # Whitespace only
except ValueError as e:
    print(e)  # "Input text cannot be empty or whitespace-only"
```

### 3. Supporting Response Models

**SpeechResponse**: Returned after successful speech generation
- `audio`: Base64-encoded audio data
- `format`: Audio format identifier
- `sample_rate`: Sample rate in Hz
- `duration_ms`: Optional duration

**ModelInfo**: Information about available models
- `id`: Model identifier
- `object`: Always "model"
- `created`: Unix timestamp
- `owned_by": "qwen3-tts"

**VoiceInfo**: Information about available voices
- `voice_id`: Voice identifier
- `name`: Human-readable name
- `preview_url`: Optional preview audio URL

**ModelsListResponse**: List of available models
**VoicesListResponse**: List of available voices

## Design Decisions

### 1. OpenAI Compatibility First
The `OpenAISpeechRequest` model prioritizes OpenAI API compatibility while adding Qwen3-specific fields as optional extensions. This allows:
- Drop-in replacement for OpenAI TTS API clients
- Gradual adoption of Qwen3-specific features
- Easy migration path for existing applications

### 2. Strict Validation
All fields include validation:
- Length constraints on text input
- Range constraints on numeric fields
- Literal type constraints for enums
- Type safety through Pydantic v2

### 3. Extensibility
The schema design allows for future extensions:
- Additional response formats can be added to the Literal
- New optional fields can be added without breaking existing clients
- VoiceModelType can support new model types

## Integration Points

### Routers Layer
- `openai_compatible.py` uses `OpenAISpeechRequest` for request validation
- Returns `SpeechResponse`, `ModelsListResponse`, `VoicesListResponse`

### Service Layer
- Receives validated `OpenAISpeechRequest` objects
- Maps request fields to appropriate generation methods

### External API Clients
- Compatible with OpenAI SDK clients
- Supports standard OpenAI TTS parameters
- Extends with Qwen3-specific capabilities

## Testing Considerations

1. **Validation Tests**:
   - Empty/whitespace input rejection
   - Speed bounds checking
   - Maximum text length enforcement

2. **Serialization Tests**:
   - JSON serialization/deserialization
   - OpenAI SDK compatibility

3. **Field Mapping Tests**:
   - Voice name mapping (OpenAI aliases to Qwen3 speakers)
   - Model ID mapping (OpenAI models to Qwen3 models)

## Future Enhancements

1. Additional audio formats (WebM, OGG Vorbis)
2. Voice style parameters (emotion, intensity)
3. Batch processing requests
4. Streaming response schemas
5. Voice preview generation schemas
