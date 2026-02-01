#!/bin/bash
#
# Installation script for Qwen3-TTS-FastAPI with CUDA 12.8 support
#
# This script installs PyTorch with CUDA 12.8 support and all other dependencies.
# Run this script if your system has NVIDIA drivers with CUDA 12.8 support.
# CUDA 12.8 is recommended for RTX 50-series cards (RTX 5090, 5080, etc.)
#
# Usage: ./install-cu128.sh
#

set -e  # Exit on error

echo "=========================================="
echo "Qwen3-TTS-FastAPI Installation"
echo "CUDA 12.8 Version"
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
echo "This will install:"
echo "  • PyTorch with CUDA 12.8 support"
echo "  • torchaudio with CUDA 12.8"
echo "  • qwen-tts and all dependencies"
echo "  • FastAPI and related packages"
echo ""
echo -e "${YELLOW}Note:${NC} CUDA 12.8 is recommended for RTX 50-series GPUs"
echo "Make sure your NVIDIA drivers support CUDA 12.8"
echo "You can check with: nvidia-smi"
echo ""

read -p "Do you want to proceed? (y/N) " -n 1 -r
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

# Step 3: Install PyTorch with CUDA 12.8 first
echo ""
echo "Step 1/3: Installing PyTorch with CUDA 12.8..."
echo "This may take several minutes..."
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

# Verify PyTorch installation
echo ""
echo "Verifying PyTorch installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else "N/A"}')"

echo -e "${GREEN}✓ PyTorch with CUDA 12.8 installed${NC}"

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
echo "To start the server:"
echo "  source .venv/bin/activate"
echo "  uv run uvicorn api.main:app --reload"
echo ""
echo "To download models:"
echo "  cd models"
echo "  python download_models.py --help"
echo ""

# Check if CUDA is actually available
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo -e "${GREEN}✓ CUDA is available and working!${NC}"
    python -c "import torch; print(f'CUDA Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
else
    echo -e "${YELLOW}⚠ Warning: CUDA is not available.${NC}"
    echo "PyTorch installed but cannot access GPU."
    echo "You may need to:"
    echo "  1. Check NVIDIA drivers: nvidia-smi"
    echo "  2. Verify CUDA installation"
    echo "  3. Try the CPU-only install script instead"
fi

echo ""
echo -e "${GREEN}✓ Installation finished successfully!${NC}"
