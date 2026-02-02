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
"""API-based test for Qwen3-TTS Custom Voice model.

This script tests the custom voice model via FastAPI server instead of
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


def send_custom_voice_request(
    api_url: str,
    text: str | List[str],
    language: str | List[str],
    speaker: str | List[str],
    instruct: str | List[str] | None = None,
    model: str = "qwen3-tts-12hz-0.6b-custom-voice",
    response_format: str = "wav",
) -> Tuple[List[np.ndarray], int, float]:
    """Send request to TTS API for custom voice generation.

    Args:
        api_url: Base URL of the FastAPI server
        text: Text to synthesize (single string or list for batch)
        language: Language code(s) for synthesis
        speaker: Speaker name(s) to use
        instruct: Voice instruction(s) for tone/emotion
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
        spk = speaker if isinstance(speaker, str) else " | ".join(speaker)
        inst = (
            instruct
            if isinstance(instruct, str)
            else " | ".join(instruct)
            if instruct
            else ""
        )
    else:
        input_text = text
        lang = language if isinstance(language, str) else language[0]
        spk = speaker if isinstance(speaker, str) else speaker[0]
        inst = (
            instruct if isinstance(instruct, str) else (instruct[0] if instruct else "")
        )

    payload = {
        "model": model,
        "input": input_text,
        "voice": spk.lower(),  # OpenAI voice IDs are lowercase
        "speaker": spk,
        "response_format": response_format,
        "language": lang,
        "streaming_mode": "full",
    }

    # Add instruction if provided
    if inst:
        payload["instruct"] = inst

    # Log request details
    print(f"[DEBUG] Sending request to: {endpoint}")
    print(
        f"[DEBUG] Payload: model={payload.get('model')}, voice={payload.get('voice')}, streaming_mode={payload.get('streaming_mode')}"
    )
    print(f"[DEBUG] Text length: {len(payload.get('input', ''))}")

    # Time the request
    t0 = time.time()
    try:
        response = requests.post(
            endpoint,
            json=payload,
            stream=False,
            timeout=300,
            headers={"Accept": "audio/wav, */*", "Connection": "keep-alive"},
        )
        t1 = time.time()
        print(f"[DEBUG] Response received in {t1 - t0:.3f}s")
        print(f"[DEBUG] Status code: {response.status_code}")
        print(f"[DEBUG] Content-Type: {response.headers.get('Content-Type')}")
        print(f"[DEBUG] Content-Length: {response.headers.get('Content-Length')}")
        print(f"[DEBUG] Transfer-Encoding: {response.headers.get('Transfer-Encoding')}")
        print(f"[DEBUG] Response content length: {len(response.content)} bytes")
    except Exception as e:
        print(f"[DEBUG] Request failed: {type(e).__name__}: {e}")
        raise

    response.raise_for_status()

    # Read all audio data
    audio_data = response.content
    print(f"[DEBUG] Audio data size: {len(audio_data)} bytes")

    if len(audio_data) == 0:
        raise ValueError("Empty response received from server")

    # Parse WAV data
    try:
        wav_buffer = io.BytesIO(audio_data)
        audio_array, sample_rate = sf.read(wav_buffer, dtype="float32")
        print(f"[DEBUG] Audio parsed: shape={audio_array.shape}, sr={sample_rate}")
    except Exception as e:
        print(f"[DEBUG] Failed to parse audio: {e}")
        # Save raw data for debugging
        debug_file = "/tmp/debug_audio_data.bin"
        with open(debug_file, "wb") as f:
            f.write(audio_data[:1000])  # Save first 1000 bytes
        print(f"[DEBUG] Saved first 1000 bytes to {debug_file}")
        raise

    # Handle mono/stereo
    if len(audio_array.shape) > 1:
        audio_array = audio_array[:, 0]

    # Split batch responses if needed
    # For simplicity, return as single audio for now
    return [audio_array], sample_rate, t1 - t0


def main() -> None:
    """Main entry point for API-based custom voice tests."""
    # Configuration
    API_BASE_URL = get_api_url()
    MODEL_PATH = "models/custom-voice"
    OUT_DIR = "qwen3_tts_test_api_custom_voice_output_wav"

    ensure_dir(OUT_DIR)

    print(f"Testing against API: {API_BASE_URL}")
    print(f"Model: {MODEL_PATH}")
    print(f"Output directory: {OUT_DIR}")

    print("\n" + "=" * 70)
    print("Starting API-based Custom Voice tests")
    print("=" * 70)

    # -------- Single (with instruct) --------
    print("\n[CustomVoice Single with instruct] Running...")
    try:
        wavs, sr, elapsed_time = send_custom_voice_request(
            api_url=API_BASE_URL,
            text="其实我真的有发现，我是一个特别善于观察别人情绪的人。",
            language="Chinese",
            speaker="Vivian",
            instruct="用特别愤怒的语气说",
            model=MODEL_PATH,
        )

        print(
            f"[CustomVoice Single] time: {elapsed_time:.3f}s, n_wavs={len(wavs)}, sr={sr}"
        )

        output_path = os.path.join(OUT_DIR, "qwen3_tts_test_api_custom_single.wav")
        sf.write(output_path, wavs[0], sr)
        print(f"[CustomVoice Single] Saved: {output_path}")

    except Exception as e:
        print(f"[CustomVoice Single] ERROR: {e}")

    # -------- Batch (some empty instruct) --------
    print("\n[CustomVoice Batch] Running...")
    try:
        texts = [
            "其实我真的有发现，我是一个特别善于观察别人情绪的人。",
            "She said she would be here by noon.",
        ]
        languages = ["Chinese", "English"]
        speakers = ["Vivian", "Ryan"]
        instructs = ["", "Very happy."]

        all_wavs = []
        total_time = 0.0

        for i in range(len(texts)):
            wavs, sr, elapsed_time = send_custom_voice_request(
                api_url=API_BASE_URL,
                text=texts[i],
                language=languages[i],
                speaker=speakers[i],
                instruct=instructs[i] if instructs[i] else None,
                model=MODEL_PATH,
            )
            all_wavs.extend(wavs)
            total_time += elapsed_time

        print(
            f"[CustomVoice Batch] time: {total_time:.3f}s, n_wavs={len(all_wavs)}, sr={sr}"
        )

        for i, w in enumerate(all_wavs):
            output_path = os.path.join(
                OUT_DIR, f"qwen3_tts_test_api_custom_batch_{i}.wav"
            )
            sf.write(output_path, w, sr)
            print(f"[CustomVoice Batch] Saved: {output_path}")

    except Exception as e:
        print(f"[CustomVoice Batch] ERROR: {e}")

    print("\n" + "=" * 70)
    print("API-based Custom Voice tests completed")
    print(f"Output files saved to: {OUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
