# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""API-based test for Qwen3-TTS Voice Design model.

This script tests the voice design model via FastAPI server instead of
direct model inference, allowing comparison of timing between API and
direct inference approaches.
"""

import os
import time
from pathlib import Path
from typing import List, Tuple

import requests
import soundfile as sf
import io
import numpy as np


def load_env_config() -> dict[str, str]:
    """Load configuration from .env file.

    Tries to load from .env first, then falls back to .env.example.

    Returns:
        Dictionary of environment variables
    """
    env_vars = {}

    # Try .env first, then .env.example
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
    # Check environment variable first
    env_url = os.getenv("TTS_API_URL")
    if env_url:
        return env_url

    # Load from .env file
    env_vars = load_env_config()
    host = env_vars.get("HOST", "127.0.0.1")
    port = env_vars.get("PORT", "8000")

    return f"http://{host}:{port}"


def ensure_dir(d: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(d, exist_ok=True)


def send_voice_design_request(
    api_url: str,
    text: str | List[str],
    language: str | List[str],
    instruct: str | List[str],
    model: str = "qwen3-tts-12hz-1.7b-voice-design",
    response_format: str = "wav",
) -> Tuple[List[np.ndarray], int, float]:
    """Send request to TTS API for voice design generation.

    Args:
        api_url: Base URL of the FastAPI server
        text: Text to synthesize (single string or list for batch)
        language: Language code(s) for synthesis
        instruct: Voice design instruction(s)
        model: Model identifier
        response_format: Audio format (wav, mp3, etc.)

    Returns:
        Tuple of (list of audio arrays, sample rate, elapsed time in seconds)
    """
    endpoint = f"{api_url}/v1/audio/speech"

    # Handle batch vs single
    if isinstance(text, list):
        input_text = " | ".join(text)
        lang = language if isinstance(language, str) else " | ".join(language)
        inst = instruct if isinstance(instruct, str) else " | ".join(instruct)
    else:
        input_text = text
        lang = language if isinstance(language, str) else language[0]
        inst = instruct if isinstance(instruct, str) else instruct[0]

    payload = {
        "model": model,
        "input": input_text,
        "voice": "alloy",
        "response_format": response_format,
        "language": lang,
        "instruct": inst,
        "streaming_mode": "full",
    }

    # Time the request
    t0 = time.time()
    response = requests.post(endpoint, json=payload, stream=False, timeout=300)
    t1 = time.time()

    response.raise_for_status()

    # Read all audio data
    audio_data = response.content

    # Parse WAV data
    wav_buffer = io.BytesIO(audio_data)
    audio_array, sample_rate = sf.read(wav_buffer, dtype="float32")

    # Handle mono/stereo
    if len(audio_array.shape) > 1:
        audio_array = audio_array[:, 0]

    return [audio_array], sample_rate, t1 - t0


def main() -> None:
    """Main entry point for API-based voice design tests."""
    # Configuration
    API_BASE_URL = get_api_url()
    MODEL_PATH = "qwen3-tts-12hz-1.7b-voice-design"
    OUT_DIR = "qwen3_tts_test_api_voice_design_output_wav"

    ensure_dir(OUT_DIR)

    print(f"Testing against API: {API_BASE_URL}")
    print(f"Model: {MODEL_PATH}")
    print(f"Output directory: {OUT_DIR}")

    print("\n" + "=" * 70)
    print("Starting API-based Voice Design tests")
    print("=" * 70)

    # -------- Single --------
    print("\n[VoiceDesign Single] Running...")
    try:
        wavs, sr, elapsed_time = send_voice_design_request(
            api_url=API_BASE_URL,
            text="哥哥，你回来啦，人家等了你好久好久了，要抱抱！",
            language="Chinese",
            instruct="体现撒娇稚嫩的萝莉女声，音调偏高且起伏明显，营造出黏人、做作又刻意卖萌的听觉效果。",
            model=MODEL_PATH,
        )

        print(
            f"[VoiceDesign Single] time: {elapsed_time:.3f}s, n_wavs={len(wavs)}, sr={sr}"
        )

        output_path = os.path.join(
            OUT_DIR, "qwen3_tts_test_api_voice_design_single.wav"
        )
        sf.write(output_path, wavs[0], sr)
        print(f"[VoiceDesign Single] Saved: {output_path}")

    except Exception as e:
        print(f"[VoiceDesign Single] ERROR: {e}")

    # -------- Batch --------
    print("\n[VoiceDesign Batch] Running...")
    try:
        texts = [
            "哥哥，你回来啦，人家等了你好久好久了，要抱抱！",
            "It's in the top drawer... wait, it's empty? No way, that's impossible! I'm sure I put it there!",
        ]
        languages = ["Chinese", "English"]
        instructs = [
            "体现撒娇稚嫩的萝莉女声，音调偏高且起伏明显，营造出黏人、做作又刻意卖萌的听觉效果。",
            "Speak in an incredulous tone, but with a hint of panic beginning to creep into your voice.",
        ]

        wavs, sr, elapsed_time = send_voice_design_request(
            api_url=API_BASE_URL,
            text=texts,
            language=languages,
            instruct=instructs,
            model=MODEL_PATH,
        )

        print(
            f"[VoiceDesign Batch] time: {elapsed_time:.3f}s, n_wavs={len(wavs)}, sr={sr}"
        )

        for i, w in enumerate(wavs):
            output_path = os.path.join(
                OUT_DIR, f"qwen3_tts_test_api_voice_design_batch_{i}.wav"
            )
            sf.write(output_path, w, sr)
            print(f"[VoiceDesign Batch] Saved: {output_path}")

    except Exception as e:
        print(f"[VoiceDesign Batch] ERROR: {e}")

    print("\n" + "=" * 70)
    print("API-based Voice Design tests completed")
    print(f"Output files saved to: {OUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
