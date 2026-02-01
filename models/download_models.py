#!/usr/bin/env python3
"""
Model download script for Qwen3-TTS-FastAPI.

Downloads Qwen3-TTS models from HuggingFace.
Supports selective downloads, renaming to project conventions, and batch operations.

Usage:
    python download_models.py --models base,custom-voice
    python download_models.py --all
    python download_models.py --models base-large --rename

Available models:
    - base (0.6B): Voice cloning base model (~2.5 GB)
    - base-large (1.7B): Voice cloning large model (~4.5 GB)
    - custom-voice (0.6B): Custom voice small model (~2.5 GB)
    - custom-voice-large (1.7B): Custom voice large model (~4.5 GB)
    - voice-design (1.7B): Voice design model (~4.5 GB)
    - tokenizer: Speech tokenizer (~500 MB)

Total size if downloading all models: ~19 GB
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

from huggingface_hub import hf_hub_download, list_repo_files
from loguru import logger

# Model definitions with HuggingFace repo IDs and sizes
MODELS = {
    "base": {
        "repo_id": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "size_gb": 2.52,
        "description": "0.6B Base model for voice cloning",
        "files": [
            "model.safetensors",
            "config.json",
            "generation_config.json",
            "preprocessor_config.json",
            "tokenizer_config.json",
            "vocab.json",
        ],
    },
    "base-large": {
        "repo_id": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "size_gb": 4.54,
        "description": "1.7B Base model for voice cloning",
        "files": [
            "model.safetensors",
            "config.json",
            "generation_config.json",
            "preprocessor_config.json",
            "tokenizer_config.json",
            "vocab.json",
        ],
    },
    "custom-voice": {
        "repo_id": "Qwen/Qwen3-TTS-12Hz-0.6B-Custom-Voice",
        "size_gb": 2.52,
        "description": "0.6B Custom Voice model",
        "files": [
            "model.safetensors",
            "config.json",
            "generation_config.json",
            "preprocessor_config.json",
            "tokenizer_config.json",
            "vocab.json",
        ],
    },
    "custom-voice-large": {
        "repo_id": "Qwen/Qwen3-TTS-12Hz-1.7B-Custom-Voice",
        "size_gb": 4.54,
        "description": "1.7B Custom Voice model",
        "files": [
            "model.safetensors",
            "config.json",
            "generation_config.json",
            "preprocessor_config.json",
            "tokenizer_config.json",
            "vocab.json",
        ],
    },
    "voice-design": {
        "repo_id": "Qwen/Qwen3-TTS-12Hz-1.7B-Voice-Design",
        "size_gb": 4.54,
        "description": "1.7B Voice Design model",
        "files": [
            "model.safetensors",
            "config.json",
            "generation_config.json",
            "preprocessor_config.json",
            "tokenizer_config.json",
            "vocab.json",
        ],
    },
    "tokenizer": {
        "repo_id": "Qwen/Qwen3-TTS-Tokenizer-12Hz",
        "size_gb": 0.50,
        "description": "Speech tokenizer",
        "files": [
            "model.safetensors",
            "config.json",
            "preprocessor_config.json",
            "configuration.json",
        ],
    },
}

# Project naming convention mapping
PROJECT_NAMES = {
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base": "base",
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base": "base-large",
    "Qwen/Qwen3-TTS-12Hz-0.6B-Custom-Voice": "custom-voice",
    "Qwen/Qwen3-TTS-12Hz-1.7B-Custom-Voice": "custom-voice-large",
    "Qwen/Qwen3-TTS-12Hz-1.7B-Voice-Design": "voice-design",
    "Qwen/Qwen3-TTS-Tokenizer-12Hz": "tokenizer",
}


def download_model(
    model_key: str,
    target_dir: Path,
    rename_to_project: bool = False,
) -> bool:
    """Download a single model.

    Args:
        model_key: Key from MODELS dict
        target_dir: Directory to download to
        rename_to_project: Whether to rename to project naming convention

    Returns:
        True if successful, False otherwise
    """
    if model_key not in MODELS:
        logger.error(f"Unknown model: {model_key}")
        return False

    model_info = MODELS[model_key]
    repo_id = model_info["repo_id"]

    # Determine target directory name
    if rename_to_project:
        target_name = PROJECT_NAMES.get(repo_id, model_key)
    else:
        target_name = model_key

    model_dir = target_dir / target_name
    model_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading {model_key} ({model_info['description']})...")
    logger.info(f"Source: {repo_id}")
    logger.info(f"Target: {model_dir}")
    logger.info(f"Expected size: {model_info['size_gb']:.2f} GB")

    try:
        # Download each file
        for filename in model_info["files"]:
            logger.info(f"  Downloading {filename}...")

            # Download file
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(model_dir),
                local_dir_use_symlinks=False,
            )

        logger.success(f"✓ Successfully downloaded {model_key} to {model_dir}")
        return True

    except Exception as e:
        logger.error(f"✗ Failed to download {model_key}: {e}")
        return False


def get_total_size(model_keys: list[str]) -> float:
    """Calculate total size of selected models in GB."""
    total = 0.0
    for key in model_keys:
        if key in MODELS:
            total += MODELS[key]["size_gb"]
    return total


def list_available_models():
    """Print list of available models."""
    print("\nAvailable models:")
    print("-" * 70)
    for key, info in MODELS.items():
        print(f"  {key:20} - {info['description']:30} (~{info['size_gb']:.2f} GB)")
    print("-" * 70)
    total_size = get_total_size(list(MODELS.keys()))
    print(f"  {'Total (all models)':20} - {'':30} (~{total_size:.2f} GB)")
    print()


def prompt_yes_no(question: str, default: str = "yes") -> bool:
    """Ask a yes/no question and return the answer."""
    valid = {"yes": True, "y": True, "no": False, "n": False}

    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError(f"Invalid default answer: '{default}'")

    while True:
        choice = input(question + prompt).lower().strip()
        if choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            print("Please respond with 'yes' or 'no' (or 'y' or 'n').")


def main():
    parser = argparse.ArgumentParser(
        description="Download Qwen3-TTS models for Qwen3-TTS-FastAPI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download specific models
  python download_models.py --models base,custom-voice
  
  # Download all models
  python download_models.py --all
  
  # Download with project naming convention
  python download_models.py --models base-large --rename

Available models: base, base-large, custom-voice, custom-voice-large, voice-design, tokenizer
        """,
    )

    parser.add_argument(
        "--models",
        type=str,
        help="Comma-separated list of models to download (e.g., 'base,custom-voice')",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all available models",
    )
    parser.add_argument(
        "--rename",
        action="store_true",
        help="Rename models to match project naming convention (base, base-large, etc.)",
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        default=".",
        help="Target directory for downloads (default: current directory)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models and exit",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Automatically answer yes to all prompts (non-interactive mode)",
    )

    args = parser.parse_args()

    # List models and exit
    if args.list:
        list_available_models()
        return 0

    # Validate arguments
    if not args.models and not args.all:
        parser.print_help()
        list_available_models()
        return 1

    # Determine which models to download
    if args.all:
        models_to_download = list(MODELS.keys())
    else:
        models_to_download = [m.strip() for m in args.models.split(",")]

        # Validate model names
        invalid_models = [m for m in models_to_download if m not in MODELS]
        if invalid_models:
            logger.error(f"Invalid model names: {', '.join(invalid_models)}")
            list_available_models()
            return 1

    # Calculate total size
    total_size = get_total_size(models_to_download)

    # Display download plan
    print("\n" + "=" * 70)
    print("Qwen3-TTS Model Download")
    print("=" * 70)
    print(f"\nModels to download ({len(models_to_download)}):")
    for model in models_to_download:
        info = MODELS[model]
        target_name = (
            PROJECT_NAMES.get(info["repo_id"], model) if args.rename else model
        )
        print(f"  • {model} -> {target_name} ({info['size_gb']:.2f} GB)")

    print(f"\nTotal download size: {total_size:.2f} GB")
    print(f"Target directory: {Path(args.target_dir).absolute()}")
    print(f"Rename to project convention: {'Yes' if args.rename else 'No'}")
    print()

    # Ask for confirmation
    if not args.yes:
        if not prompt_yes_no("Proceed with download?", default="yes"):
            print("Download cancelled.")
            return 0

    # Ask about renaming if not specified
    rename = args.rename
    if not args.rename and not args.yes:
        rename = prompt_yes_no(
            "Rename models to project naming convention (base, base-large, etc.)?",
            default="yes",
        )

    # Perform downloads
    print()
    target_path = Path(args.target_dir)

    success_count = 0
    fail_count = 0

    for model in models_to_download:
        if download_model(model, target_path, rename):
            success_count += 1
        else:
            fail_count += 1
        print()

    # Summary
    print("=" * 70)
    print("Download Summary")
    print("=" * 70)
    print(f"Successful: {success_count}/{len(models_to_download)}")
    print(f"Failed: {fail_count}/{len(models_to_download)}")

    if fail_count == 0:
        print("\n✓ All models downloaded successfully!")
        print(f"\nModels are ready in: {target_path.absolute()}")
        print("\nYou can now start the server:")
        print("  uv run uvicorn api.main:app --reload")
        return 0
    else:
        print(f"\n✗ {fail_count} model(s) failed to download.")
        print("Please check your internet connection and try again.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
