#!/bin/bash
#
# Installation script for Qwen3-TTS-FastAPI (CPU-only version)
#
# This script installs PyTorch CPU-only version and all other dependencies.
# Use this script if you don't have an NVIDIA GPU or don't need GPU acceleration.
# Note: CPU inference is significantly slower than GPU.
#
# Usage: ./install-cpu.sh
#

set -e  # Exit on error

echo "=========================================="
echo "Qwen3-TTS-FastAPI Installation"
echo "CPU-Only Version"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo -e "${RED}Error: uv is not installed.${NC}"
    echo "Please install uv first:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo -e "${GREEN}✓ uv found${NC}"

# Manual confirmation
echo ""
echo -e "${YELLOW}WARNING: This is the CPU-only version.${NC}"
echo ""
echo "CPU-only installation means:"
echo "  • No GPU acceleration (much slower inference)"
echo "  • Higher CPU usage"
echo "  • Longer processing times for TTS"
echo ""
echo "This is suitable for:"
echo "  • Testing and development"
echo "  • Systems without NVIDIA GPUs"
echo "  • Low-traffic deployments"
echo ""
echo "Use CUDA versions (cu126/cu128/cu130) for production with GPUs."
echo ""

read -p "Do you want to proceed with CPU-only installation? (y/N) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Installation cancelled."
    exit 0
fi

# Get project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo ""
echo "Installing in: $PROJECT_ROOT"
echo ""

# Step 1: Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv --python 3.12
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment already exists${NC}"
fi

# Step 2: Activate virtual environment
source .venv/bin/activate

# Step 3: Install PyTorch CPU-only version first
echo ""
echo "Step 1/3: Installing PyTorch (CPU-only)..."
echo "This may take several minutes..."
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Verify PyTorch installation
echo ""
echo "Verifying PyTorch installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'MPS available (Apple Silicon): {torch.backends.mps.is_available() if hasattr(torch.backends, \"mps\") else \"N/A\"}')"

echo -e "${GREEN}✓ PyTorch (CPU-only) installed${NC}"

# Step 4: Install the project and remaining dependencies
echo ""
echo "Step 2/3: Installing qwen-tts and remaining dependencies..."
uv pip install -e .

echo -e "${GREEN}✓ All dependencies installed${NC}"

# Step 5: Install dev dependencies (optional)
echo ""
echo "Step 3/3: Installing development dependencies (optional)..."
read -p "Install dev dependencies (pytest, black, ruff, mypy)? (y/N) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    uv pip install -e ".[dev]"
    echo -e "${GREEN}✓ Development dependencies installed${NC}"
else
    echo "Skipping dev dependencies."
fi

# Final verification
echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo -e "${YELLOW}Note: Running in CPU-only mode.${NC}"
echo "Inference will be slower than GPU acceleration."
echo ""
echo "To start the server:"
echo "  source .venv/bin/activate"
echo "  uv run uvicorn api.main:app --reload"
echo ""
echo "To download models:"
echo "  cd models"
echo "  python download_models.py --help"
echo ""

# Check for Apple Silicon MPS support
if python -c "import torch; exit(0 if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 1)" 2>/dev/null; then
    echo -e "${GREEN}✓ Apple Silicon MPS (Metal Performance Shaders) detected!${NC}"
    echo "  MPS will be used for acceleration on Apple Silicon Macs."
fi

echo ""
echo -e "${GREEN}✓ Installation finished successfully!${NC}"
echo ""
echo "Performance Tips:"
echo "  • Use smaller models (0.6B) instead of large (1.7B) for faster inference"
echo "  • Consider using DEFAULT_MODEL_SIZE=small in your .env file"
echo "  • Enable CLEANUP_ENABLED=true to manage memory efficiently"
