"""OpenAI-compatible router for text-to-speech with Qwen3 models."""

import io
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel

from api.src.services.qwen3_tts_service import Qwen3TTSService
from api.src.structures.schemas import (
    ModelsListResponse,
    ModelInfo,
    OpenAISpeechRequest,
    VoicesListResponse,
    VoiceInfo,
)

router = APIRouter(prefix="/audio", tags=["audio"])

# Global service instance (initialized in main.py)
_tts_service: Qwen3TTSService | None = None


def get_tts_service(request: Request) -> Qwen3TTSService:
    """Dependency to get the TTS service from app state.

    Args:
        request: FastAPI request object

    Returns:
        Initialized Qwen3TTSService

    Raises:
        HTTPException: If service is not initialized
    """
    service = getattr(request.app.state, "tts_service", None)
    if service is None:
        logger.error("TTS service not initialized")
        raise HTTPException(
            status_code=503,
            detail="TTS service is not available. Please try again later.",
        )
    return service


def set_tts_service(service: Qwen3TTSService) -> None:
    """Set the global TTS service instance.

    Called during application startup to initialize the service.

    Args:
        service: Initialized Qwen3TTSService instance
    """
    global _tts_service
    _tts_service = service
    logger.info("TTS service registered with router")


class SpeechResponse(BaseModel):
    """Response model for speech generation endpoint."""

    status: str
    message: str


@router.post("/speech")
async def create_speech(
    request: OpenAISpeechRequest,
    service: Qwen3TTSService = Depends(get_tts_service),
) -> StreamingResponse:
    """Generate speech from text (OpenAI-compatible endpoint).

    This endpoint provides OpenAI-compatible text-to-speech generation
    using Qwen3 TTS models. It supports voice cloning, custom voices,
    and voice design through various parameter combinations.

    Args:
        request: OpenAI-compatible speech request with text, voice, model, etc.
        service: TTS service instance (injected)

    Returns:
        StreamingResponse with audio data in the requested format

    Raises:
        HTTPException: 400 for invalid requests, 500 for generation errors

    Example:
        ```python
        import requests

        response = requests.post(
            "http://localhost:8000/v1/audio/speech",
            json={
                "model": "tts-1",
                "input": "Hello, world!",
                "voice": "alloy",
                "response_format": "mp3"
            }
        )

        with open("output.mp3", "wb") as f:
            f.write(response.content)
        ```
    """
    logger.info(
        f"Speech request: model={request.model}, voice={request.voice}, "
        f"format={request.response_format}, text_length={len(request.input)}"
    )

    try:
        # Validate request
        await service.validate_request(request)

        # Generate speech
        async def audio_stream() -> AsyncGenerator[bytes, None]:
            """Stream audio chunks as they're generated."""
            chunk_count = 0
            total_bytes = 0

            try:
                async for chunk in service.generate_speech(request):
                    chunk_count += 1

                    if isinstance(chunk.data, bytes):
                        audio_bytes = chunk.data
                    else:
                        # Encode numpy array to requested format
                        audio_bytes = service.backend.encode_audio(
                            chunk.data,
                            chunk.sample_rate,
                            format=request.response_format,
                        )

                    total_bytes += len(audio_bytes)
                    yield audio_bytes

                    if chunk.is_last:
                        logger.info(
                            f"Streaming complete: {chunk_count} chunks, "
                            f"{total_bytes} bytes total"
                        )
            except Exception as e:
                logger.error(f"Streaming error in chunk {chunk_count}: {e}")
                # Re-raise so the HTTP layer can handle it
                raise

        # Determine content type based on format
        content_type_map = {
            "mp3": "audio/mpeg",
            "wav": "audio/wav",
            "ogg": "audio/ogg",
            "opus": "audio/opus",
            "aac": "audio/aac",
            "flac": "audio/flac",
            "pcm": "audio/pcm",
        }
        content_type = content_type_map.get(
            request.response_format, "application/octet-stream"
        )

        logger.info(f"Returning StreamingResponse with media_type={content_type}")

        return StreamingResponse(
            audio_stream(),
            media_type=content_type,
            headers={
                "Content-Disposition": f'attachment; filename="speech.{request.response_format}"',
                "X-Model": request.model,
                "X-Voice": request.voice,
                "Cache-Control": "no-cache",
            },
        )

    except ValueError as e:
        logger.warning(f"Invalid request: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Speech generation failed: {str(e)}"
        )
    except Exception:
        logger.exception("Unexpected error during speech generation")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred during speech generation",
        )


@router.get("/models")
async def list_models(
    service: Qwen3TTSService = Depends(get_tts_service),
) -> ModelsListResponse:
    """List available TTS models (OpenAI-compatible).

    Returns a list of available models in OpenAI format, including both
    OpenAI-compatible aliases (tts-1, tts-1-hd) and native Qwen3 model IDs.

    Args:
        service: TTS service instance (injected)

    Returns:
        ModelsListResponse containing list of ModelInfo objects

    Example Response:
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
    """
    logger.debug("Listing available models")

    try:
        models = await service.get_available_models()

        # Convert to ModelInfo objects
        model_infos = [
            ModelInfo(
                id=m["id"],
                object=m["object"],
                created=m["created"],
                owned_by=m["owned_by"],
            )
            for m in models
        ]

        return ModelsListResponse(
            object="list",
            data=model_infos,
        )

    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve model list")


@router.get("/voices")
async def list_voices(
    service: Qwen3TTSService = Depends(get_tts_service),
) -> VoicesListResponse:
    """List available voices (OpenAI-compatible).

    Returns a list of available voices with their OpenAI-compatible IDs
    and Qwen3 speaker names.

    Args:
        service: TTS service instance (injected)

    Returns:
        VoicesListResponse containing list of VoiceInfo objects

    Example Response:
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
    """
    logger.debug("Listing available voices")

    try:
        voices = await service.get_available_voices()
        return VoicesListResponse(voices=voices)

    except Exception as e:
        logger.error(f"Failed to list voices: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve voice list")


@router.get("/voices/{voice_id}")
async def get_voice(
    voice_id: str,
    service: Qwen3TTSService = Depends(get_tts_service),
) -> VoiceInfo:
    """Get information about a specific voice.

    Args:
        voice_id: Voice identifier (e.g., "alloy", "echo")
        service: TTS service instance (injected)

    Returns:
        VoiceInfo containing voice details

    Raises:
        HTTPException: 404 if voice not found

    Example Response:
        ```json
        {
            "voice_id": "alloy",
            "name": "Vivian",
            "preview_url": null
        }
        ```
    """
    logger.debug(f"Getting voice info: {voice_id}")

    try:
        voice = await service.get_voice_info(voice_id)

        if voice is None:
            raise HTTPException(status_code=404, detail=f"Voice '{voice_id}' not found")

        return voice

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get voice info: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to retrieve voice information"
        )


@router.post("/speech/base64")
async def create_speech_base64(
    request: OpenAISpeechRequest,
    service: Qwen3TTSService = Depends(get_tts_service),
) -> dict:
    """Generate speech and return as base64-encoded JSON (non-streaming).

    This endpoint provides a non-streaming alternative to /audio/speech,
    returning the complete audio as a base64-encoded string in JSON format.
    Useful for clients that cannot handle streaming responses.

    Args:
        request: OpenAI-compatible speech request
        service: TTS service instance (injected)

    Returns:
        Dictionary with base64-encoded audio and metadata

    Example Response:
        ```json
        {
            "audio": "base64encodedstring...",
            "format": "mp3",
            "sample_rate": 24000,
            "duration_ms": 1500
        }
        ```
    """
    logger.info(f"Base64 speech request: model={request.model}, voice={request.voice}")

    try:
        # Collect all audio chunks
        audio_buffer = io.BytesIO()
        sample_rate = 24000  # Default
        duration_ms = 0

        async for chunk in service.generate_speech(request):
            if isinstance(chunk.data, bytes):
                audio_buffer.write(chunk.data)
            else:
                # Encode and write
                encoded = service.backend.encode_audio(
                    chunk.data,
                    chunk.sample_rate,
                    format=request.response_format,
                )
                audio_buffer.write(encoded)

            sample_rate = chunk.sample_rate
            if isinstance(chunk.data, bytes):
                # Estimate duration (rough approximation)
                duration_ms += len(chunk.data) // 32
            else:
                duration_ms += int(len(chunk.data) / chunk.sample_rate * 1000)

        # Encode to base64
        audio_base64 = service.encode_audio_to_base64(audio_buffer.getvalue())

        return {
            "audio": audio_base64,
            "format": request.response_format,
            "sample_rate": sample_rate,
            "duration_ms": duration_ms,
        }

    except ValueError as e:
        logger.warning(f"Invalid request: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail="Speech generation failed")
