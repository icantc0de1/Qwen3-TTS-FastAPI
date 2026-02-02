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
"""API-based test for Qwen3-TTS 0.6B Base model.

This script tests the base model (0.6B) voice cloning via FastAPI server
instead of direct model inference, allowing comparison of timing between
API and direct inference approaches.
"""

import os
import time
import base64
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


def download_audio(url: str) -> bytes:
    """Download audio file from URL."""
    response: requests.Response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.content


def audio_to_base64(audio_bytes: bytes) -> str:
    """Convert audio bytes to base64 string."""
    return base64.b64encode(audio_bytes).decode("utf-8")


def send_api_request(
    api_url: str,
    text: str | List[str],
    language: str | List[str],
    ref_audio_b64: str | List[str] | None = None,
    ref_text: str | List[str] | None = None,
    model: str = "qwen3-tts-12hz-0.6b-base",
    response_format: str = "wav",
) -> Tuple[List[np.ndarray], int, float]:
    """Send request to TTS API and return audio data with timing.

    Args:
        api_url: Base URL of the FastAPI server
        text: Text to synthesize (single string or list for batch)
        language: Language code(s) for synthesis
        ref_audio_b64: Base64-encoded reference audio (single or list)
        ref_text: Reference text for ICL mode (single or list)
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
    else:
        input_text = text
        lang = language if isinstance(language, str) else language[0]

    payload = {
        "model": model,
        "input": input_text,
        "voice": "alloy",
        "response_format": response_format,
        "language": lang,
        "streaming_mode": "full",
    }

    # Add reference audio and text if provided
    if ref_audio_b64:
        if isinstance(ref_audio_b64, list):
            payload["ref_audio"] = ref_audio_b64[0]  # Use first for single
        else:
            payload["ref_audio"] = ref_audio_b64

    if ref_text:
        if isinstance(ref_text, list):
            payload["ref_text"] = ref_text[0]
        else:
            payload["ref_text"] = ref_text

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


def run_case(
    api_url: str,
    out_dir: str,
    case_name: str,
    text: str | List[str],
    language: str | List[str],
    ref_audio_b64: str | None = None,
    ref_text: str | None = None,
    model: str = "qwen3-tts-12hz-0.6b-base",
) -> None:
    """Run a single test case and save results.

    Args:
        api_url: Base URL of the FastAPI server
        out_dir: Directory to save output files
        case_name: Name of the test case
        text: Text to synthesize
        language: Language for synthesis
        ref_audio_b64: Base64-encoded reference audio for cloning
        ref_text: Reference text for ICL mode
        model: Model identifier
    """
    print(f"\n[{case_name}] Running...")

    try:
        wavs, sr, elapsed_time = send_api_request(
            api_url=api_url,
            text=text,
            language=language,
            ref_audio_b64=ref_audio_b64,
            ref_text=ref_text,
            model=model,
        )

        print(f"[{case_name}] time: {elapsed_time:.3f}s, n_wavs={len(wavs)}, sr={sr}")

        for i, w in enumerate(wavs):
            output_path = os.path.join(out_dir, f"{case_name}_{i}.wav")
            sf.write(output_path, w, sr)
            print(f"[{case_name}] Saved: {output_path}")

    except Exception as e:
        print(f"[{case_name}] ERROR: {e}")


def main() -> None:
    """Main entry point for API-based base model tests."""
    # Configuration
    API_BASE_URL = get_api_url()
    MODEL_PATH = "qwen3-tts-12hz-0.6b-base"
    OUT_DIR = "qwen3_tts_test_api_base_small_output_wav"

    ensure_dir(OUT_DIR)

    print(f"Testing against API: {API_BASE_URL}")
    print(f"Model: {MODEL_PATH}")
    print(f"Output directory: {OUT_DIR}")

    # Reference audio URLs
    ref_audio_url_1 = (
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone_2.wav"
    )
    ref_audio_url_2 = (
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone_1.wav"
    )

    # Download and encode reference audios
    print("\nDownloading reference audios...")
    ref_audio_bytes_1 = download_audio(ref_audio_url_1)
    ref_audio_bytes_2 = download_audio(ref_audio_url_2)
    ref_audio_single = audio_to_base64(ref_audio_bytes_1)
    _ = [
        audio_to_base64(ref_audio_bytes_1),
        audio_to_base64(ref_audio_bytes_2),
    ]

    ref_text_single = "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you."
    _ = [
        "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you.",
        "甚至出现交易几乎停滞的情况。",
    ]

    # Synthesis targets
    syn_text_single = (
        "Good one. Okay, fine, I'm just gonna leave this sock monkey here. Goodbye."
    )
    syn_lang_single = "English"

    syn_text_batch = [
        "Good one. Okay, fine, I'm just gonna leave this sock monkey here. Goodbye.",
        "其实我真的有发现，我是一个特别善于观察别人情绪的人。",
    ]
    syn_lang_batch = ["English", "Chinese"]

    # Run test cases
    print("\n" + "=" * 70)
    print("Starting API-based base model (0.6B) tests")
    print("=" * 70)

    # Case 1: prompt single + synth single, direct
    run_case(
        API_BASE_URL,
        OUT_DIR,
        "case1_promptSingle_synSingle_direct_icl",
        text=syn_text_single,
        language=syn_lang_single,
        ref_audio_b64=ref_audio_single,
        ref_text=ref_text_single,
        model=MODEL_PATH,
    )

    # Case 1b: prompt single + synth single, xvec_only mode
    run_case(
        API_BASE_URL,
        OUT_DIR,
        "case1b_promptSingle_synSingle_direct_xvec_only",
        text=syn_text_single,
        language=syn_lang_single,
        ref_audio_b64=ref_audio_single,
        ref_text=None,
        model=MODEL_PATH,
    )

    # Case 2: prompt single + synth batch
    run_case(
        API_BASE_URL,
        OUT_DIR,
        "case2_promptSingle_synBatch_direct_icl",
        text=syn_text_batch,
        language=syn_lang_batch,
        ref_audio_b64=ref_audio_single,
        ref_text=ref_text_single,
        model=MODEL_PATH,
    )

    # Case 2b: xvec only mode with batch
    run_case(
        API_BASE_URL,
        OUT_DIR,
        "case2b_promptSingle_synBatch_direct_xvec_only",
        text=syn_text_batch,
        language=syn_lang_batch,
        ref_audio_b64=ref_audio_single,
        ref_text=None,
        model=MODEL_PATH,
    )

    print("\n" + "=" * 70)
    print("API-based base model (0.6B) tests completed")
    print(f"Output files saved to: {OUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
