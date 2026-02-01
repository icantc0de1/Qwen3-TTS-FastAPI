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
        description="Model ID for TTS generation",
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
        description="Voice identifier (OpenAI aliases mapped to Qwen3 speakers)",
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
        description="Language code (e.g., 'en', 'zh', 'ja')",
    )
    speaker: Optional[str] = Field(
        default=None,
        description="Speaker name for custom_voice models",
    )
    instruct: Optional[str] = Field(
        default=None,
        description="Voice design instruction for voice_design models",
    )
    ref_audio: Optional[str] = Field(
        default=None,
        description="Base64-encoded reference audio for voice_clone models",
    )
    ref_text: Optional[str] = Field(
        default=None,
        description="Reference text for voice_clone ICL mode",
    )
    normalization_options: Optional[NormalizationOptions] = Field(
        default=None,
        description="Optional text normalization settings. If not provided, text will be normalized with default settings.",
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
