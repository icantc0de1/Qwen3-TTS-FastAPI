"""Pydantic schemas for Qwen3 TTS API."""

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class VoiceModelType(str, Enum):
    """Supported model types for Qwen3 TTS.

    Each model type provides different capabilities:
    - BASE: Voice cloning from reference audio (requires reference audio)
    - VOICE_DESIGN: Natural language voice design (uses text instructions)
    - CUSTOM_VOICE: Predefined speaker voices (uses speaker IDs)
    """

    BASE = "base"
    VOICE_DESIGN = "voice_design"
    CUSTOM_VOICE = "custom_voice"


class StreamingMode(str, Enum):
    """Streaming mode for TTS generation.

    Controls how audio is chunked for streaming delivery:
    - FULL: Generate complete audio first, then deliver (no streaming)
    - SENTENCE: Split text by sentences, stream each as completed
    - CHUNK: Split text by fixed character count with overlap
    """

    FULL = "full"
    SENTENCE = "sentence"
    CHUNK = "chunk"


class NormalizationOptions(BaseModel):
    """Options for the normalization system"""

    normalize: bool = Field(
        default=True,
        description="Normalizes input text to make it easier for the model to say",
    )
    unit_normalization: bool = Field(
        default=False, description="Transforms units like 10KB to 10 kilobytes"
    )
    url_normalization: bool = Field(
        default=True,
        description="Changes urls so they can be properly pronounced",
    )
    email_normalization: bool = Field(
        default=True,
        description="Changes emails so they can be properly pronounced",
    )
    optional_pluralization_normalization: bool = Field(
        default=True,
        description="Replaces (s) with s so some words get pronounced correctly",
    )
    phone_normalization: bool = Field(
        default=True,
        description="Changes phone numbers so they can be properly pronounced",
    )
    replace_remaining_symbols: bool = Field(
        default=True,
        description="Replaces the remaining symbols after normalization with their words",
    )


class OpenAISpeechRequest(BaseModel):
    """OpenAI-compatible request schema for text-to-speech generation.

    This model validates and structures incoming requests to the /v1/audio/speech
    endpoint, ensuring compatibility with OpenAI's TTS API format.

    Attributes:
        model: Model identifier (e.g., "tts-1", "tts-1-hd", or Qwen3-specific IDs)
        input: Text to synthesize (1-4096 characters)
        voice: Voice identifier (e.g., "alloy", "echo", or Qwen3 speaker names)
        response_format: Output audio format
        speed: Speech speed multiplier (0.25x to 4.0x)
        language: Optional language code for multilingual synthesis
        speaker: Optional speaker name (for custom_voice models)
        instruct: Optional voice design instruction (for voice_design models)
        ref_audio: Optional base64-encoded reference audio (for voice_clone models)
        ref_text: Optional reference text (for voice_clone ICL mode)
    """

    model: str = Field(
        default="tts-1",
        description="Model ID for TTS generation. Determines generation method: "
        "custom-voice models use 'voice' parameter, "
        "voice-design models use 'instruct' parameter, "
        "base models require 'ref_audio' for voice cloning.",
        examples=["tts-1", "tts-1-hd", "qwen3-tts-12hz-1.7b-custom-voice"],
    )
    input: str = Field(
        ...,
        min_length=1,
        max_length=4096,
        description="Text to synthesize into speech",
    )
    voice: str = Field(
        default="alloy",
        description="Voice/speaker identifier for custom_voice models. "
        "OpenAI aliases (alloy, echo, etc.) mapped to Qwen3 speakers. "
        "Required for custom_voice models, ignored by voice_design.",
    )
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = Field(
        default="mp3",
        description="Output audio format",
    )
    speed: float = Field(
        default=1.0,
        ge=0.25,
        le=4.0,
        description="Speech speed multiplier",
    )
    # Qwen3-specific optional fields
    language: Optional[str] = Field(
        default=None,
        description="Language (e.g., 'English', 'Chinese')",
    )
    speaker: Optional[str] = Field(
        default=None,
        description="Optional: Direct speaker name override for custom_voice models. "
        "If provided, takes precedence over 'voice' field.",
    )
    instruct: Optional[str] = Field(
        default=None,
        description="Voice instruction. Required for voice_design models. "
        "Optional for custom_voice models (modifies tone/emotion). "
        "Example: 'Very happy' or 'A young female voice with warm tone'",
    )
    ref_audio: Optional[str] = Field(
        default=None,
        description="Base64-encoded reference audio for base model voice cloning. "
        "Required when using base models for voice cloning.",
    )
    ref_text: Optional[str] = Field(
        default=None,
        description="Reference text for base model voice cloning ICL mode. "
        "Optional: provide transcript of reference audio for better cloning.",
    )
    normalization_options: Optional[NormalizationOptions] = Field(
        default=None,
        description="Optional text normalization settings. If not provided, text will be normalized with default settings.",
    )
    streaming_mode: Optional[StreamingMode] = Field(
        default=StreamingMode.SENTENCE,
        description="Streaming mode for audio generation. "
        "'full': Generate complete audio first (no streaming). "
        "'sentence': Split by sentences for true streaming (default). "
        "'chunk': Split by fixed character count with overlap.",
    )
    chunk_size: Optional[int] = Field(
        default=None,
        ge=50,
        le=1000,
        description="Maximum characters per chunk for streaming. "
        "For 'sentence' mode: max sentence length before forced split. "
        "For 'chunk' mode: target characters per chunk. Default: 150 (sentence), 200 (chunk)",
    )

    @field_validator("input")
    @classmethod
    def validate_input(cls, v: str) -> str:
        """Validate that input text is not empty or whitespace-only.

        Args:
            v: Input text to validate

        Returns:
            Stripped input text

        Raises:
            ValueError: If input is empty or contains only whitespace
        """
        stripped = v.strip()
        if not stripped:
            raise ValueError("Input text cannot be empty or whitespace-only")
        return stripped

    @field_validator("speed")
    @classmethod
    def validate_speed(cls, v: float) -> float:
        """Validate speed is within reasonable bounds.

        Args:
            v: Speed multiplier value

        Returns:
            Validated speed value

        Raises:
            ValueError: If speed is outside valid range
        """
        if v < 0.25 or v > 4.0:
            raise ValueError("Speed must be between 0.25 and 4.0")
        return v


class SpeechResponse(BaseModel):
    """Response schema for speech generation.

    Attributes:
        audio: Base64-encoded audio data
        format: Audio format (mp3, wav, etc.)
        sample_rate: Audio sample rate in Hz
        duration_ms: Approximate audio duration in milliseconds
    """

    audio: str = Field(..., description="Base64-encoded audio data")
    format: str = Field(..., description="Audio format identifier")
    sample_rate: int = Field(..., description="Sample rate in Hz")
    duration_ms: Optional[int] = Field(None, description="Duration in milliseconds")


class ModelInfo(BaseModel):
    """Information about an available TTS model.

    Attributes:
        id: Model identifier
        object: Type of object (always "model")
        created: Unix timestamp of model creation
        owned_by: Organization that owns the model
    """

    id: str = Field(..., description="Model identifier")
    object: str = Field(default="model", description="Object type")
    created: int = Field(..., description="Unix timestamp")
    owned_by: str = Field(default="qwen3-tts", description="Model owner")


class VoiceInfo(BaseModel):
    """Information about an available voice.

    Attributes:
        voice_id: Voice identifier
        name: Human-readable voice name
        preview_url: Optional URL to voice preview audio
    """

    voice_id: str = Field(..., description="Voice identifier")
    name: str = Field(..., description="Human-readable name")
    preview_url: Optional[str] = Field(None, description="Preview audio URL")


class ModelsListResponse(BaseModel):
    """Response schema for listing available models.

    Attributes:
        object: Type of object (always "list")
        data: List of available models
    """

    object: str = Field(default="list", description="Object type")
    data: list[ModelInfo] = Field(..., description="List of available models")


class VoicesListResponse(BaseModel):
    """Response schema for listing available voices.

    Attributes:
        voices: List of available voices
    """

    voices: list[VoiceInfo] = Field(..., description="List of available voices")
