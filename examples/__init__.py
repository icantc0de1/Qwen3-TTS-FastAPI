"""API-based tests for Qwen3-TTS FastAPI server.

These tests mirror the official Qwen3-TTS batch/single inference tests
but send requests to the FastAPI server instead of using direct model inference.

Usage:
    1. Start the FastAPI server:
       uv run uvicorn api.main:app --host 0.0.0.0 --port 8000

    2. Run the tests:
       uv run python tests/test_model_base_small.py
       uv run python tests/test_model_base_large.py
       uv run python tests/test_model_custom_voice.py
       uv run python tests/test_model_voice_design.py

    3. Or set a custom API URL:
       TTS_API_URL=http://localhost:8080 uv run python tests/test_model_base_small.py

Comparison with examples/:
    The examples/ directory contains tests that use direct model inference
    via Qwen3TTSModel.from_pretrained(). These API tests provide a way to
    compare timing between direct inference and API-based inference.
"""
