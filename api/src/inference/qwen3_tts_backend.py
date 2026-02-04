"""Qwen3-specific backend implementation."""

import io
import random
from dataclasses import dataclass
from typing import AsyncGenerator, Optional

import numpy as np
import soundfile as sf
import torch
from loguru import logger

from api.src.inference.qwen3_tts_model import Qwen3TTSModel
from api.src.inference.qwen3_tts_model_manager import Qwen3ModelManager
from api.src.inference.qwen3_tts_tokenizer import Qwen3TTSTokenizer
from api.src.structures.schemas import VoiceModelType


def _set_seed(seed: int) -> None:
    """Set random seed for reproducible generation.

    Sets seeds for Python's random, NumPy, and PyTorch to ensure
    consistent voice characteristics across streaming chunks.

    Args:
        seed: Random seed value (0 to 2^32-1)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


@dataclass
class AudioChunk:
    """A chunk of generated audio data.

    Represents a segment of audio in a streaming generation context.
    Can contain either raw PCM data or encoded audio bytes.

    Attributes:
        data: Audio data as bytes (encoded) or numpy array (raw PCM)
        sample_rate: Sample rate in Hz
        is_last: Whether this is the final chunk
        format: Audio format identifier ("pcm", "wav", "mp3", etc.)
        timestamp_ms: Timestamp within the audio stream (for progress tracking)
    """

    data: bytes | np.ndarray
    sample_rate: int
    is_last: bool = False
    format: str = "pcm"
    timestamp_ms: int = 0


class Qwen3TTSBackend:
    """Backend for Qwen3 TTS model operations.

    Provides a unified interface for all three Qwen3 TTS generation modes:
    - Voice Clone (Base model): Clone voice from reference audio
    - Custom Voice (CustomVoice model): Use predefined speaker IDs
    - Voice Design (VoiceDesign model): Design voices from text descriptions

    The backend handles:
    - Model loading via the ModelManager
    - Audio preprocessing and postprocessing
    - Streaming audio generation
    - Device management
    - Error handling and logging

    Example:
        >>> backend = Qwen3TTSBackend()
        >>> await backend.initialize()
        >>>
        >>> # Voice cloning
        >>> chunks = await backend.voice_clone(
        ...     text="Hello world",
        ...     ref_audio="path/to/reference.wav",
        ...     model_path="models/base"
        ... )
        >>>
        >>> # Custom voice
        >>> chunks = await backend.custom_voice(
        ...     text="Hello world",
        ...     speaker="Vivian",
        ...     model_path="models/custom-voice"
        ... )
        >>>
        >>> # Voice design
        >>> chunks = await backend.voice_design(
        ...     text="Hello world",
        ...     instruct="A warm, friendly female voice",
        ...     model_path="models/voice-design"
        ... )
    """

    def __init__(
        self,
        device: str = "cuda",
        idle_timeout: int = 300,
    ) -> None:
        """Initialize the TTS backend.

        Args:
            device: Default compute device ("cuda" or "cpu")
            idle_timeout: Seconds before auto-unloading idle models
        """
        self.device = device
        self.model_manager = Qwen3ModelManager(
            default_device=device,
            idle_timeout=idle_timeout,
        )
        self._initialized = False
        self._tokenizer: Optional[Qwen3TTSTokenizer] = None

        logger.info(f"Qwen3TTSBackend initialized (device={device})")

    async def initialize(self, tokenizer_path: Optional[str] = None) -> None:
        """Initialize the backend and load the tokenizer.

        Args:
            tokenizer_path: Path to the speech tokenizer model

        Note:
            This method is separate from __init__ to allow async initialization
            in FastAPI startup events.
        """
        if self._initialized:
            return

        if tokenizer_path:
            logger.info(f"Loading speech tokenizer from {tokenizer_path}")
            self._tokenizer = Qwen3TTSTokenizer.from_pretrained(
                tokenizer_path,
                device_map=self.device,
                dtype=torch.bfloat16,
            )
            logger.info("Speech tokenizer loaded successfully")

        self._initialized = True
        logger.info("Qwen3TTSBackend fully initialized")

    async def shutdown(self) -> None:
        """Cleanup and unload all models.

        Should be called during application shutdown to free GPU memory.
        """
        logger.info("Shutting down Qwen3TTSBackend")
        self.model_manager.unload_all(force=True)
        self._tokenizer = None
        self._initialized = False
        logger.info("Qwen3TTSBackend shutdown complete")

    def _get_model_type_from_path(self, model_path: str) -> VoiceModelType:
        """Infer model type from path string.

        Args:
            model_path: Path to the model directory

        Returns:
            Detected model type
        """
        path_lower = model_path.lower()
        if "voice-design" in path_lower or "voice_design" in path_lower:
            return VoiceModelType.VOICE_DESIGN
        elif "custom-voice" in path_lower or "custom_voice" in path_lower:
            return VoiceModelType.CUSTOM_VOICE
        else:
            return VoiceModelType.BASE

    async def voice_clone(
        self,
        text: str,
        model_path: str,
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        language: Optional[str] = None,
        x_vector_only_mode: bool = False,
        seed: Optional[int] = None,
        **generation_kwargs,
    ) -> AsyncGenerator[AudioChunk, None]:
        """Generate speech using voice cloning (Base model).

        Clones a voice from reference audio. Supports two modes:
        - x_vector_only_mode=True: Use only speaker embedding
        - x_vector_only_mode=False: Use ICL (In-Context Learning) with reference text

        Args:
            text: Text to synthesize
            model_path: Path to the base model
            ref_audio: Reference audio (file path, URL, or base64)
            ref_text: Reference text (required for ICL mode)
            language: Target language code
            x_vector_only_mode: Use speaker embedding only (no ICL)
            seed: Random seed for reproducible generation
            **generation_kwargs: Additional generation parameters
                (temperature, top_k, top_p, max_new_tokens, etc.)

        Yields:
            AudioChunk objects containing generated audio segments

        Raises:
            RuntimeError: If model loading or generation fails
            ValueError: If ref_audio is None but required
        """
        if not ref_audio:
            raise ValueError("ref_audio is required for voice_clone generation")

        logger.info(f"Voice clone request: model={model_path}, text_length={len(text)}")

        model = None
        try:
            # Set seed for reproducible generation
            if seed is not None:
                _set_seed(seed)
                logger.debug(f"Voice clone using seed={seed}")

            # Load model
            model = self.model_manager.load_model(model_path, "base")

            # Generate audio
            wavs, sample_rate = model.generate_voice_clone(
                text=text,
                language=language or "Auto",
                ref_audio=ref_audio,
                ref_text=ref_text,
                x_vector_only_mode=x_vector_only_mode,
                **generation_kwargs,
            )

            # Yield audio chunks
            duration_ms = 0
            for i, wav in enumerate(wavs):
                chunk_duration = int(len(wav) / sample_rate * 1000)

                yield AudioChunk(
                    data=wav,
                    sample_rate=sample_rate,
                    is_last=(i == len(wavs) - 1),
                    format="pcm",
                    timestamp_ms=duration_ms,
                )
                duration_ms += chunk_duration

            logger.info(
                f"Voice clone complete: {len(wavs)} chunks, {duration_ms}ms total"
            )

        except Exception as e:
            logger.error(f"Voice clone generation failed: {e}")
            raise RuntimeError(f"Voice clone failed: {e}") from e
        finally:
            # Clear local reference before releasing to ensure proper cleanup
            if model is not None:
                model = None
            if model_path:
                self.model_manager.release_model(model_path)

    async def custom_voice(
        self,
        text: str,
        model_path: str,
        speaker: str,
        language: Optional[str] = None,
        instruct: Optional[str] = None,
        seed: Optional[int] = None,
        **generation_kwargs,
    ) -> AsyncGenerator[AudioChunk, None]:
        """Generate speech using a predefined speaker (CustomVoice model).

        Uses a specific speaker ID from the model's supported speakers list.
        Optionally accepts an instruction for style control (1.7B models only).

        Args:
            text: Text to synthesize
            model_path: Path to the custom voice model
            speaker: Speaker name (e.g., "Vivian", "Serena", "Dylan")
            language: Target language code
            instruct: Optional instruction for voice style (1.7B models only)
            seed: Random seed for reproducible generation
            **generation_kwargs: Additional generation parameters

        Yields:
            AudioChunk objects containing generated audio

        Raises:
            RuntimeError: If model loading or generation fails
            ValueError: If speaker is not supported by the model
        """
        logger.info(
            f"Custom voice request: model={model_path}, speaker={speaker}, "
            f"text_length={len(text)}"
        )

        model = None
        try:
            # Set seed for reproducible generation
            if seed is not None:
                _set_seed(seed)
                logger.debug(f"Custom voice using seed={seed}")

            # Load model
            model = self.model_manager.load_model(model_path, "custom_voice")

            # Validate speaker
            supported_speakers = model.get_supported_speakers()
            if supported_speakers and speaker.lower() not in [
                s.lower() for s in supported_speakers
            ]:
                raise ValueError(
                    f"Speaker '{speaker}' not supported. "
                    f"Supported: {supported_speakers}"
                )

            # Generate audio
            wavs, sample_rate = model.generate_custom_voice(
                text=text,
                speaker=speaker,
                language=language or "Auto",
                instruct=instruct,
                **generation_kwargs,
            )

            # Yield audio chunks
            duration_ms = 0
            for i, wav in enumerate(wavs):
                chunk_duration = int(len(wav) / sample_rate * 1000)

                yield AudioChunk(
                    data=wav,
                    sample_rate=sample_rate,
                    is_last=(i == len(wavs) - 1),
                    format="pcm",
                    timestamp_ms=duration_ms,
                )
                duration_ms += chunk_duration

            logger.info(
                f"Custom voice complete: {len(wavs)} chunks, {duration_ms}ms total"
            )

        except Exception as e:
            logger.error(f"Custom voice generation failed: {e}")
            raise RuntimeError(f"Custom voice failed: {e}") from e
        finally:
            # Clear local reference before releasing to ensure proper cleanup
            if model is not None:
                model = None
            if model_path:
                self.model_manager.release_model(model_path)

    async def voice_design(
        self,
        text: str,
        model_path: str,
        instruct: str,
        language: Optional[str] = None,
        seed: Optional[int] = None,
        **generation_kwargs,
    ) -> AsyncGenerator[AudioChunk, None]:
        """Generate speech using voice design (VoiceDesign model).

                Creates a voice based on natural language instructions describing
        the desired voice characteristics (age, gender, tone, emotion, etc.).

                Args:
                    text: Text to synthesize
                    model_path: Path to the voice design model
                    instruct: Voice design instruction (e.g., "A young female voice with a warm tone")
                    language: Target language code
                    seed: Random seed for reproducible generation
                    **generation_kwargs: Additional generation parameters

                Yields:
                    AudioChunk objects containing generated audio

                Raises:
                    RuntimeError: If model loading or generation fails
        """
        logger.info(
            f"Voice design request: model={model_path}, instruct_length={len(instruct)}, "
            f"text_length={len(text)}"
        )

        model = None
        try:
            # Set seed for reproducible generation
            if seed is not None:
                _set_seed(seed)
                logger.debug(f"Voice design using seed={seed}")

            # Load model
            model = self.model_manager.load_model(model_path, "voice_design")

            # Generate audio
            wavs, sample_rate = model.generate_voice_design(
                text=text,
                instruct=instruct,
                language=language or "Auto",
                **generation_kwargs,
            )

            # Yield audio chunks
            duration_ms = 0
            for i, wav in enumerate(wavs):
                chunk_duration = int(len(wav) / sample_rate * 1000)

                yield AudioChunk(
                    data=wav,
                    sample_rate=sample_rate,
                    is_last=(i == len(wavs) - 1),
                    format="pcm",
                    timestamp_ms=duration_ms,
                )
                duration_ms += chunk_duration

            logger.info(
                f"Voice design complete: {len(wavs)} chunks, {duration_ms}ms total"
            )

        except Exception as e:
            logger.error(f"Voice design generation failed: {e}")
            raise RuntimeError(f"Voice design failed: {e}") from e
        finally:
            # Clear local reference before releasing to ensure proper cleanup
            if model is not None:
                model = None
            if model_path:
                self.model_manager.release_model(model_path)

    def encode_audio(
        self,
        wav: np.ndarray,
        sample_rate: int,
        format: str = "mp3",
    ) -> bytes:
        """Encode raw audio to specified format.

        Args:
            wav: Raw audio waveform as numpy array
            sample_rate: Audio sample rate
            format: Target format ("mp3", "wav", "ogg", "flac", "aac", "pcm")

        Returns:
            Encoded audio bytes

        Raises:
            ValueError: If format is not supported
        """
        format = format.lower()

        if format == "pcm":
            # Raw PCM bytes (16-bit signed int)
            wav_int16 = (wav * 32767).astype(np.int16)
            return wav_int16.tobytes()

        elif format in ["wav", "flac", "ogg"]:
            # Use soundfile for these formats
            buffer = io.BytesIO()
            sf.write(buffer, wav, sample_rate, format=format)
            return buffer.getvalue()

        elif format == "mp3":
            # MP3 encoding requires pydub
            try:
                from pydub import AudioSegment

                # First convert to WAV in memory
                wav_buffer = io.BytesIO()
                sf.write(wav_buffer, wav, sample_rate, format="wav")
                wav_buffer.seek(0)

                # Convert to MP3
                audio = AudioSegment.from_wav(wav_buffer)
                mp3_buffer = io.BytesIO()
                audio.export(mp3_buffer, format="mp3")
                return mp3_buffer.getvalue()
            except ImportError:
                logger.warning("pydub not installed, falling back to WAV format")
                return self.encode_audio(wav, sample_rate, "wav")

        elif format == "aac":
            # AAC encoding requires pydub
            try:
                from pydub import AudioSegment

                wav_buffer = io.BytesIO()
                sf.write(wav_buffer, wav, sample_rate, format="wav")
                wav_buffer.seek(0)

                audio = AudioSegment.from_wav(wav_buffer)
                aac_buffer = io.BytesIO()
                audio.export(aac_buffer, format="adts")  # AAC format
                return aac_buffer.getvalue()
            except ImportError:
                logger.warning("pydub not installed, falling back to WAV format")
                return self.encode_audio(wav, sample_rate, "wav")

        elif format == "opus":
            # Opus encoding
            try:
                from pydub import AudioSegment

                wav_buffer = io.BytesIO()
                sf.write(wav_buffer, wav, sample_rate, format="wav")
                wav_buffer.seek(0)

                audio = AudioSegment.from_wav(wav_buffer)
                opus_buffer = io.BytesIO()
                audio.export(opus_buffer, format="opus")
                return opus_buffer.getvalue()
            except ImportError:
                logger.warning("pydub not installed, falling back to OGG format")
                return self.encode_audio(wav, sample_rate, "ogg")

        else:
            raise ValueError(f"Unsupported audio format: {format}")

    def get_supported_speakers(self, model_path: str) -> Optional[list[str]]:
        """Get list of supported speakers for a custom voice model.

        Args:
            model_path: Path to the custom voice model

        Returns:
            List of speaker names or None if model doesn't support speaker selection
        """
        try:
            model = self.model_manager.load_model(model_path, "custom_voice")
            speakers = model.get_supported_speakers()
            self.model_manager.release_model(model_path)
            return speakers
        except Exception as e:
            logger.error(f"Failed to get supported speakers: {e}")
            return None

    def get_supported_languages(self, model_path: str) -> Optional[list[str]]:
        """Get list of supported languages for a model.

        Args:
            model_path: Path to the model

        Returns:
            List of language codes or None if model supports all languages
        """
        try:
            model_type = self._get_model_type_from_path(model_path)
            model = self.model_manager.load_model(model_path, model_type.value)
            languages = model.get_supported_languages()
            self.model_manager.release_model(model_path)
            return languages
        except Exception as e:
            logger.error(f"Failed to get supported languages: {e}")
            return None

    def cleanup(self) -> list[str]:
        """Remove idle models from cache.

        Returns:
            List of unloaded model paths
        """
        return self.model_manager.cleanup_idle_models()

    @property
    def is_initialized(self) -> bool:
        """Check if backend is initialized."""
        return self._initialized

    @property
    def tokenizer(self) -> Optional[Qwen3TTSTokenizer]:
        """Get the loaded tokenizer."""
        return self._tokenizer
