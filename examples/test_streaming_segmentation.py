"""Test streaming segmentation modes for Qwen3-TTS FastAPI.

This script demonstrates the three streaming modes available in the API:
1. FULL - Generate complete audio first (no streaming)
2. SENTENCE - Split text by sentences for true streaming
3. CHUNK - Split text by fixed character count with overlap

The segmentation feature allows control over how text is chunked for
progressive audio delivery, which is useful for:
- Reducing perceived latency in streaming applications
- Managing memory usage for long texts
- Controlling chunk size for network transmission

Usage:
    python test_streaming_segmentation.py

    Set the TTS_API_URL environment variable to override the default
    API URL (http://127.0.0.1:8000).
"""

from __future__ import annotations

import os
import time
import base64
import io
from pathlib import Path

import numpy as np
import requests
import soundfile as sf


def load_env_config() -> dict[str, str]:
    """Load configuration from .env file.

    Tries to load from .env first, then falls back to .env.example.

    Returns:
        Dictionary of environment variables
    """
    env_vars = {}

    for env_file in [".env", ".env.example"]:
        env_path = Path(env_file)
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        env_vars[key] = value
            break

    return env_vars


def get_api_url() -> str:
    """Get API URL from environment or .env file.

    Priority:
    1. TTS_API_URL environment variable
    2. HOST and PORT from .env/.env.example
    3. Default: http://127.0.0.1:8000

    Returns:
        API base URL string
    """
    env_url = os.getenv("TTS_API_URL")
    if env_url:
        return env_url

    env_vars = load_env_config()
    host = env_vars.get("HOST", "127.0.0.1")
    port = env_vars.get("PORT", "8000")

    return f"http://{host}:{port}"


def ensure_dir(d: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(d, exist_ok=True)


def download_audio(url: str) -> bytes:
    """Download audio file from URL."""
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.content


def audio_to_base64(audio_bytes: bytes) -> str:
    """Convert audio bytes to base64 string."""
    return base64.b64encode(audio_bytes).decode("utf-8")


def send_api_request(
    api_url: str,
    text: str,
    voice: str = "alloy",
    model: str = "tts-1",
    response_format: str = "wav",
    streaming_mode: str = "sentence",
    chunk_size: int | None = None,
    seed: int | None = None,
) -> tuple[np.ndarray, int, float, dict]:
    """Send request to TTS API and return audio data with timing.

    Args:
        api_url: Base URL of the FastAPI server
        text: Text to synthesize
        voice: Voice identifier (alloy, echo, fable, onyx, nova, shimmer)
        model: Model identifier (tts-1, tts-1-hd)
        response_format: Audio format (wav, mp3, ogg, opus, aac, flac)
        streaming_mode: Streaming mode (full, sentence, chunk)
        chunk_size: Maximum characters per chunk (optional)
        seed: Random seed for reproducible generation (optional)

    Returns:
        Tuple of (audio array, sample rate, elapsed time in seconds, response info dict)
    """
    endpoint = f"{api_url}/v1/audio/speech"

    payload: dict[str, str | int] = {
        "model": model,
        "input": text,
        "voice": voice,
        "response_format": response_format,
        "streaming_mode": streaming_mode,
    }

    if chunk_size is not None:
        payload["chunk_size"] = chunk_size

    if seed is not None:
        payload["seed"] = seed

    # Time the request
    t0 = time.time()
    response = requests.post(endpoint, json=payload, stream=False, timeout=300)
    t1 = time.time()

    response.raise_for_status()

    # Read audio data
    audio_data = response.content

    # Parse WAV data
    wav_buffer = io.BytesIO(audio_data)
    audio_array, sample_rate = sf.read(wav_buffer, dtype="float32")

    # Handle mono/stereo
    if len(audio_array.shape) > 1:
        audio_array = audio_array[:, 0]

    # Extract info from response headers if available
    info = {
        "elapsed_seconds": t1 - t0,
        "content_type": response.headers.get("content-type", ""),
        "content_length": response.headers.get("content-length", "unknown"),
    }

    return audio_array, sample_rate, t1 - t0, info


from api.src.services.text_processing.segmentation import (
    split_into_sentences,
    split_into_chunks,
)


def print_segmentation_info(text: str, mode: str, chunk_size: int | None) -> None:
    """Print information about how text will be segmented."""

    print(f"\n  [{mode.upper()}] Segmentation Details:")
    print(f"    Text length: {len(text)} characters")

    if mode == "sentence":
        chunks = split_into_sentences(text, max_length=chunk_size or 150)
        print(f"    Method: Sentence boundary detection")
        if chunk_size:
            print(f"    Max sentence length: {chunk_size} chars")
        print(f"    Result: {len(chunks)} sentence(s)")
        for i, s in enumerate(chunks):
            preview = s[:50] + "..." if len(s) > 50 else s
            print(f"      [{i+1}] ({len(s)} chars): {preview}")
    elif mode == "chunk":
        chunks = split_into_chunks(text, max_chars=chunk_size or 200)
        print(f"    Method: Fixed chunk size with overlap")
        if chunk_size:
            print(f"    Target chunk size: {chunk_size} chars")
        print(f"    Result: {len(chunks)} chunk(s)")
        for i, s in enumerate(chunks):
            preview = s[:50] + "..." if len(s) > 50 else s
            print(f"      [{i+1}] ({len(s)} chars): {preview}")


def run_comparison_test(
    api_url: str, out_dir: str, text: str, test_name: str, test_seed: int | None = None
) -> None:
    """Run all three streaming modes and compare results.

    Args:
        api_url: Base URL of the FastAPI server
        out_dir: Directory to save output files
        text: Text to synthesize
        test_name: Name for this test case
        test_seed: Random seed for reproducible generation (optional, uses global default)
    """
    seed = test_seed if test_seed is not None else TEST_SEED

    print(f"\n{'='*70}")
    print(f"TEST: {test_name}")
    print(f"{'='*70}")

    # Test each streaming mode
    modes = [
        ("full", None, "Full audio generation (no streaming)"),
        ("sentence", 150, "Sentence-based streaming (default)"),
        ("chunk", 200, "Fixed chunk size streaming"),
    ]

    results = []

    for mode, chunk_size, description in modes:
        print(f"\n  Testing: {description}")
        print(f"  Mode: {mode}")
        if chunk_size:
            print(f"  Chunk size: {chunk_size}")

        print_segmentation_info(text, mode, chunk_size)

        try:
            audio_array, sample_rate, elapsed, info = send_api_request(
                api_url=api_url,
                text=text,
                voice="alloy",
                model="tts-1",
                response_format="wav",
                streaming_mode=mode,
                chunk_size=chunk_size,
                seed=seed,
            )

            output_path = os.path.join(out_dir, f"{test_name}_{mode}.wav")
            sf.write(output_path, audio_array, sample_rate)

            print(f"  [SUCCESS] Time: {elapsed:.3f}s")
            print(f"  [SUCCESS] Audio length: {len(audio_array)/sample_rate:.2f}s")
            print(f"  [SUCCESS] Saved: {output_path}")
            print(f"  [INFO] Content-Type: {info['content_type']}")
            print(f"  [INFO] Content-Length: {info['content_length']}")

            results.append((mode, elapsed))

        except Exception as e:
            print(f"  [ERROR] {e}")

    # Print timing comparison
    if results:
        print(f"\n  Timing Comparison:")
        for mode, elapsed in results:
            bar_length = int(elapsed * 10)
            bar = "#" * bar_length
            print(f"    {mode:10s}: {elapsed:.2f}s [{bar}]")


# Global seed for reproducibility across all tests
TEST_SEED = 42


def run_edge_cases(
    api_url: str, out_dir: str, test_seed: int | None = None
) -> None:
    """Test edge cases for segmentation.

    Args:
        api_url: Base URL of the FastAPI server
        out_dir: Directory to save output files
        test_seed: Random seed for reproducible generation (optional, uses global default)
    """
    seed = test_seed if test_seed is not None else TEST_SEED

    print(f"\n{'='*70}")
    print("EDGE CASE TESTS")
    print(f"{'='*70}")

    # Edge case 1: Very short text
    short_text = "Hello world!"
    print(f"\n  [{short_text}] Length: {len(short_text)} chars")
    try:
        audio, sr, elapsed, _ = send_api_request(
            api_url=api_url,
            text=short_text,
            streaming_mode="sentence",
            seed=seed,
        )
        output_path = os.path.join(out_dir, "edge_short_sentence.wav")
        sf.write(output_path, audio, sr)
        print(f"  [SUCCESS] Saved: {output_path}")
    except Exception as e:
        print(f"  [ERROR] {e}")

    # Edge case 2: Very long text (multiple sentences)
    long_text = (
        "The quick brown fox jumps over the lazy dog. "
        "This sentence is intentionally long to test the sentence splitting feature. "
        "When text exceeds the maximum chunk size, it should be automatically split. "
        "This helps manage memory usage and reduces latency in streaming applications. "
        "The segmentation algorithm handles various edge cases gracefully. "
        "Punctuation marks like periods, exclamation points, and question marks "
        "are used to determine natural break points in the text. "
        "Connection words such as 'and', 'but', and 'or' are preserved properly. "
        "Decimal numbers like 3.14 are not treated as sentence endings. "
        "Abbreviations like Mr., Dr., and Mrs. are handled correctly."
    )
    print(f"\n  [Long Text] Length: {len(long_text)} chars")
    print_segmentation_info(long_text, "sentence", 100)

    try:
        audio, sr, elapsed, _ = send_api_request(
            api_url=api_url,
            text=long_text,
            streaming_mode="sentence",
            chunk_size=100,
            seed=seed,
        )
        output_path = os.path.join(out_dir, "edge_long_text.wav")
        sf.write(output_path, audio, sr)
        print(f"  [SUCCESS] Time: {elapsed:.3f}s")
        print(f"  [SUCCESS] Saved: {output_path}")
    except Exception as e:
        print(f"  [ERROR] {e}")

    # Edge case 3: Chinese text with CJK punctuation
    chinese_text = "你好！这是一个测试。请听一段中文语音合成。这很重要！"
    print(f"\n  [Chinese Text] Length: {len(chinese_text)} chars")
    print_segmentation_info(chinese_text, "sentence", 50)

    try:
        audio, sr, elapsed, _ = send_api_request(
            api_url=api_url,
            text=chinese_text,
            voice="nova",
            streaming_mode="sentence",
            chunk_size=50,
            seed=seed,
        )
        output_path = os.path.join(out_dir, "edge_chinese_text.wav")
        sf.write(output_path, audio, sr)
        print(f"  [SUCCESS] Time: {elapsed:.3f}s")
        print(f"  [SUCCESS] Saved: {output_path}")
    except Exception as e:
        print(f"  [ERROR] {e}")

    # Edge case 4: Chunk mode with different sizes
    test_text = (
        "This is a test sentence that we will use to demonstrate "
        "different chunk sizes in chunk mode streaming. "
        "By adjusting the chunk size, you can control the granularity "
        "of audio streaming for your application."
    )

    print(f"\n  [Chunk Mode Comparison] Length: {len(test_text)} chars")

    for chunk_size in [50, 100, 200]:
        print(f"\n    Testing chunk_size={chunk_size}:")
        chunks = split_into_chunks(test_text, max_chars=chunk_size)
        print(f"      Result: {len(chunks)} chunks")
        for i, c in enumerate(chunks):
            preview = c[:40] + "..." if len(c) > 40 else c
            print(f"        [{i+1}] ({len(c)}): {preview}")

        try:
            audio, sr, elapsed, _ = send_api_request(
                api_url=api_url,
                text=test_text,
                streaming_mode="chunk",
                chunk_size=chunk_size,
                seed=seed,
            )
            output_path = os.path.join(out_dir, f"edge_chunk_size_{chunk_size}.wav")
            sf.write(output_path, audio, sr)
            print(f"      [SUCCESS] Time: {elapsed:.3f}s")
        except Exception as e:
            print(f"      [ERROR] {e}")


def main() -> None:
    """Main entry point for streaming segmentation tests."""
    API_BASE_URL = get_api_url()
    OUT_DIR = "qwen3_tts_test_segmentation_output"

    ensure_dir(OUT_DIR)

    print(f"Testing against API: {API_BASE_URL}")
    print(f"Output directory: {OUT_DIR}")

    # Test texts with varying characteristics
    test_cases = [
        (
            "Simple text with a few sentences. This tests basic sentence splitting. "
            "The algorithm should handle short sentences appropriately.",
            "Basic sentence splitting",
        ),
        (
            "First, let's consider the main topic. Then we will examine the details. "
            "After that, we can draw some conclusions. Finally, we will summarize.",
            "Connection words and ordering",
        ),
        (
            "Welcome to the demonstration of our text-to-speech system. "
            "This feature allows for streaming audio generation with configurable chunking. "
            "You can choose between sentence-level or fixed-chunk streaming modes. "
            "The chunk size parameter controls how text is divided for processing.",
            "Feature demonstration",
        ),
    ]

    print("\n" + "=" * 70)
    print("Qwen3-TTS Streaming Segmentation Tests")
    print("=" * 70)
    print("\nThis demo tests the three streaming modes:")
    print("  1. FULL - Complete audio generation (no streaming)")
    print("  2. SENTENCE - Split by sentence boundaries")
    print("  3. CHUNK - Split by fixed character count with overlap")
    print("\nStreaming is useful for:")
    print("  - Reducing perceived latency")
    print("  - Managing memory for long texts")
    print("  - Controlling chunk size for network transmission")

    for text, name in test_cases:
        run_comparison_test(API_BASE_URL, OUT_DIR, text, name)

    run_edge_cases(API_BASE_URL, OUT_DIR, TEST_SEED)

    print("\n" + "=" * 70)
    print("All streaming segmentation tests completed")
    print(f"Output files saved to: {OUT_DIR}")
    print("=" * 70)

    print("\nNote: Audio files generated with the same seed should have")
    print("similar voice characteristics, though timing may vary by mode.")


if __name__ == "__main__":
    main()