"""Qwen3 TTS service for OpenAI-compatible API."""

import base64
import time
from typing import AsyncGenerator, Optional

import numpy as np
from loguru import logger

from api.src.core.config import settings
from api.src.inference.qwen3_tts_backend import AudioChunk, Qwen3TTSBackend
from api.src.services.text_processing import normalize_text
from api.src.structures.schemas import (
    NormalizationOptions,
    OpenAISpeechRequest,
    VoiceInfo,
    VoiceModelType,
)


class Qwen3TTSService:
    """High-level service for Qwen3 TTS operations.

    This service layer bridges the OpenAI-compatible API interface with the
    underlying Qwen3TTSBackend. It handles:
    - Request validation and normalization
    - Voice/model name mapping (OpenAI aliases to Qwen3)
    - Audio format encoding
    - Error handling with appropriate HTTP status codes
    - Streaming response generation

    The service is designed to be used by FastAPI routers and provides
    a clean interface that matches OpenAI's TTS API semantics.

    Example:
        >>> service = Qwen3TTSService(backend)
        >>> async for chunk in service.generate_speech(request):
        ...     process_audio_chunk(chunk)
    """

    def __init__(self, backend: Qwen3TTSBackend) -> None:
        """Initialize the TTS service.

        Args:
            backend: Initialized Qwen3TTSBackend instance
        """
        self.backend = backend
        self._voice_mappings: dict[str, str] = {}
        self._model_mappings: dict[str, str] = {}
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the service and load mappings."""
        if self._initialized:
            return

        # Load OpenAI-compatible mappings
        await self._load_mappings()
        self._initialized = True
        logger.info("Qwen3TTSService initialized")

    async def _load_mappings(self) -> None:
        """Load voice and model mappings from configuration."""
        import json
        from pathlib import Path

        mappings_path = Path(__file__).parent.parent / "core" / "openai_mappings.json"

        try:
            with open(mappings_path) as f:
                mappings = json.load(f)

            self._voice_mappings = mappings.get("voices", {})
            self._model_mappings = mappings.get("models", {})

            logger.info(
                f"Loaded mappings: {len(self._voice_mappings)} voices, "
                f"{len(self._model_mappings)} models"
            )
        except Exception as e:
            logger.error(f"Failed to load mappings: {e}")
            # Use default mappings as fallback
            self._voice_mappings = {
                "alloy": "Vivian",
                "echo": "Dylan",
                "fable": "Eric",
                "onyx": "Ryan",
                "nova": "Aiden",
                "shimmer": "Sohee",
            }
            self._model_mappings = {
                "tts-1": "qwen3-tts-12hz-0.6b-custom-voice",
                "tts-1-hd": "qwen3-tts-12hz-1.7b-voice-design",
            }

    def _normalize_input_text(
        self, text: str, normalization_options: Optional[NormalizationOptions] = None
    ) -> str:
        """Normalize input text for better TTS pronunciation.

        Text normalization is derived from the kokoro-fastapi project
        by remsky (https://github.com/remsky/kokoro-fastapi), licensed under Apache 2.0.

        Args:
            text: Raw input text
            normalization_options: Optional normalization settings.
                If None, uses default normalization (all features enabled).

        Returns:
            Normalized text ready for TTS processing
        """
        if normalization_options is None:
            # Use default normalization settings
            normalization_options = NormalizationOptions()

        if not normalization_options.normalize:
            # Normalization disabled, return text as-is
            return text

        try:
            normalized = normalize_text(text, normalization_options)
            if normalized != text:
                logger.debug(
                    f"Text normalized: '{text[:50]}...' -> '{normalized[:50]}...'"
                )
            return normalized
        except Exception as e:
            logger.warning(f"Text normalization failed, using original text: {e}")
            return text

    def _resolve_model(self, model_id: str) -> str:
        """Resolve OpenAI model ID to Qwen3 model path.

        Args:
            model_id: OpenAI model identifier (e.g., "tts-1")

        Returns:
            Path to the Qwen3 model
        """
        # Check if it's a direct model path
        if model_id.startswith("models/") or "/" in model_id:
            return model_id

        # Check mappings
        if model_id in self._model_mappings:
            qwen3_id = self._model_mappings[model_id]
            # Get path from settings
            model_path = settings.get_model_path(qwen3_id)
            if model_path:
                return model_path

        # Default fallback
        return settings.custom_voice_model_path

    def _resolve_voice(self, voice_id: str) -> str:
        """Resolve OpenAI voice ID to Qwen3 speaker name.

        Args:
            voice_id: OpenAI voice identifier (e.g., "alloy")

        Returns:
            Qwen3 speaker name
        """
        return self._voice_mappings.get(voice_id, voice_id)

    def _determine_generation_mode(
        self, request: OpenAISpeechRequest
    ) -> VoiceModelType:
        """Determine which generation mode to use based on request.

        Args:
            request: The speech request

        Returns:
            Appropriate VoiceModelType
        """
        if request.ref_audio:
            return VoiceModelType.BASE
        elif request.speaker:
            return VoiceModelType.CUSTOM_VOICE
        elif request.instruct:
            return VoiceModelType.VOICE_DESIGN
        else:
            # Default to custom voice if voice is specified
            return VoiceModelType.CUSTOM_VOICE

    async def generate_speech(
        self,
        request: OpenAISpeechRequest,
    ) -> AsyncGenerator[AudioChunk, None]:
        """Generate speech from an OpenAI-compatible request.

        This is the main entry point for speech generation. It:
        1. Resolves model and voice mappings
        2. Determines the appropriate generation mode
        3. Routes to the correct backend method
        4. Encodes audio to the requested format

        Args:
            request: Validated OpenAISpeechRequest

        Yields:
            AudioChunk objects with encoded audio

        Raises:
            ValueError: If request parameters are invalid
            RuntimeError: If generation fails
        """
        start_time = time.time()

        # Resolve model path
        model_path = self._resolve_model(request.model)
        logger.info(
            f"Speech generation: model={request.model} -> {model_path}, "
            f"voice={request.voice}, format={request.response_format}"
        )

        # Determine generation mode
        mode = self._determine_generation_mode(request)
        logger.debug(f"Generation mode: {mode.value}")

        # Resolve voice to speaker name
        speaker = self._resolve_voice(request.voice)

        # Normalize input text for better TTS pronunciation
        normalized_text = self._normalize_input_text(
            request.input, request.normalization_options
        )

        # Common generation kwargs
        gen_kwargs = {
            "temperature": max(0.1, min(1.0, 0.9 / request.speed)),
            "top_p": 1.0,
            "top_k": 50,
            "max_new_tokens": 2048,
        }

        try:
            # Route to appropriate backend method
            if mode == VoiceModelType.BASE and request.ref_audio:
                # Voice cloning mode
                chunks = self.backend.voice_clone(
                    text=normalized_text,
                    model_path=model_path,
                    ref_audio=request.ref_audio,
                    ref_text=request.ref_text,
                    language=request.language,
                    **gen_kwargs,
                )

            elif mode == VoiceModelType.VOICE_DESIGN and request.instruct:
                # Voice design mode
                chunks = self.backend.voice_design(
                    text=normalized_text,
                    model_path=model_path,
                    instruct=request.instruct,
                    language=request.language,
                    **gen_kwargs,
                )

            else:
                # Default to custom voice mode
                chunks = self.backend.custom_voice(
                    text=normalized_text,
                    model_path=model_path,
                    speaker=speaker,
                    language=request.language,
                    instruct=request.instruct,
                    **gen_kwargs,
                )

            # Process and encode audio chunks
            total_duration_ms = 0
            chunk_count = 0

            async for chunk in chunks:
                chunk_count += 1

                # Encode to requested format
                if isinstance(chunk.data, np.ndarray):
                    encoded_bytes = self.backend.encode_audio(
                        chunk.data,
                        chunk.sample_rate,
                        format=request.response_format,
                    )
                else:
                    encoded_bytes = chunk.data

                # Calculate duration
                if isinstance(chunk.data, np.ndarray):
                    duration_ms = int(len(chunk.data) / chunk.sample_rate * 1000)
                else:
                    # Estimate based on format (rough approximation)
                    duration_ms = len(encoded_bytes) // 32  # ~32 bytes/ms for mp3

                total_duration_ms += duration_ms

                yield AudioChunk(
                    data=encoded_bytes,
                    sample_rate=chunk.sample_rate,
                    is_last=chunk.is_last,
                    format=request.response_format,
                    timestamp_ms=chunk.timestamp_ms,
                )

            elapsed = time.time() - start_time
            logger.info(
                f"Generation complete: {chunk_count} chunks, "
                f"{total_duration_ms}ms audio, {elapsed:.2f}s elapsed"
            )

        except Exception as e:
            logger.error(f"Speech generation failed: {e}")
            raise

    async def get_available_models(self) -> list[dict]:
        """Get list of available models.

        Returns:
            List of model information dictionaries
        """
        models = []
        current_time = int(time.time())

        # Add mapped OpenAI models
        for openai_id, qwen3_id in self._model_mappings.items():
            models.append(
                {
                    "id": openai_id,
                    "object": "model",
                    "created": current_time,
                    "owned_by": "qwen3-tts",
                }
            )

        # Add direct Qwen3 models
        qwen3_models = [
            "qwen3-tts-12hz-0.6b-base",
            "qwen3-tts-12hz-0.6b-custom-voice",
            "qwen3-tts-12hz-1.7b-base",
            "qwen3-tts-12hz-1.7b-custom-voice",
            "qwen3-tts-12hz-1.7b-voice-design",
        ]

        for model_id in qwen3_models:
            if model_id not in models:
                models.append(
                    {
                        "id": model_id,
                        "object": "model",
                        "created": current_time,
                        "owned_by": "qwen3-tts",
                    }
                )

        return models

    async def get_available_voices(self) -> list[VoiceInfo]:
        """Get list of available voices.

        Returns:
            List of VoiceInfo objects
        """
        voices = []

        for openai_id, qwen3_name in self._voice_mappings.items():
            voices.append(
                VoiceInfo(
                    voice_id=openai_id,
                    name=qwen3_name,
                    preview_url=None,  # Could add preview URLs in future
                )
            )

        return voices

    async def get_voice_info(self, voice_id: str) -> Optional[VoiceInfo]:
        """Get information about a specific voice.

        Args:
            voice_id: Voice identifier

        Returns:
            VoiceInfo or None if not found
        """
        if voice_id in self._voice_mappings:
            return VoiceInfo(
                voice_id=voice_id,
                name=self._voice_mappings[voice_id],
                preview_url=None,
            )
        return None

    def encode_audio_to_base64(self, audio_bytes: bytes) -> str:
        """Encode audio bytes to base64 string.

        Args:
            audio_bytes: Raw audio bytes

        Returns:
            Base64-encoded string
        """
        return base64.b64encode(audio_bytes).decode("utf-8")

    async def validate_request(self, request: OpenAISpeechRequest) -> None:
        """Validate a speech request before processing.

        Args:
            request: The request to validate

        Raises:
            ValueError: If request is invalid
        """
        # Validate model exists
        model_path = self._resolve_model(request.model)
        if not model_path:
            raise ValueError(f"Unknown model: {request.model}")

        # Validate voice/speaker compatibility
        mode = self._determine_generation_mode(request)

        if mode == VoiceModelType.CUSTOM_VOICE:
            # Check if speaker is supported
            speaker = self._resolve_voice(request.voice)
            supported_speakers = self.backend.get_supported_speakers(model_path)
            if supported_speakers and speaker.lower() not in [
                s.lower() for s in supported_speakers
            ]:
                logger.warning(
                    f"Speaker '{speaker}' may not be supported by {request.model}"
                )

        logger.debug(f"Request validation passed for model={request.model}")

    @property
    def is_initialized(self) -> bool:
        """Check if service is initialized."""
        return self._initialized
