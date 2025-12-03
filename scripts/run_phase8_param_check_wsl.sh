#!/bin/bash
# Phase 8 Max Parameters Verification - WSL Runner
# Usage: wsl -d ubuntu bash scripts/run_phase8_param_check_wsl.sh

set -e

echo "=========================================="
echo "Phase 8 Max Parameters Check (WSL)"
echo "=========================================="
echo ""

# Get the Windows path and convert to WSL path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Project Root: $PROJECT_ROOT"
echo ""

# Change to project directory
cd "$PROJECT_ROOT"

# Check for virtual environment
VENV_PATH="$PROJECT_ROOT/venv_ubuntu"
if [ -d "$VENV_PATH" ]; then
    echo "Found virtual environment: $VENV_PATH"
    source "$VENV_PATH/bin/activate"
    PYTHON="$VENV_PATH/bin/python"
    echo "✓ Virtual environment activated"
else
    echo "⚠️  Virtual environment not found, using system Python"
    PYTHON="python3"
fi
echo ""

# Check Python
echo "Checking Python..."
if ! command -v $PYTHON &> /dev/null; then
    echo "❌ Error: Python not found"
    exit 1
fi

PYTHON_VERSION=$($PYTHON --version)
echo "✓ Python: $PYTHON_VERSION"
echo ""

# Check PyTorch
echo "Checking PyTorch..."
if ! $PYTHON -c "import torch" 2>/dev/null; then
    echo "❌ Error: PyTorch not installed"
    echo "Please install: pip install torch"
    exit 1
fi

TORCH_VERSION=$($PYTHON -c "import torch; print(torch.__version__)")
echo "✓ PyTorch: $TORCH_VERSION"
echo ""

# Check CUDA (optional)
if $PYTHON -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    CUDA_VERSION=$($PYTHON -c "import torch; print(torch.version.cuda)")
    GPU_NAME=$($PYTHON -c "import torch; print(torch.cuda.get_device_name(0))")
    echo "✓ CUDA: $CUDA_VERSION"
    echo "✓ GPU: $GPU_NAME"
else
    echo "⚠️  CUDA not available (CPU mode)"
fi
echo ""

# Run verification script
echo "Running parameter verification..."
echo ""

$PYTHON scripts/verify_phase8_max_params.py

echo ""
echo "=========================================="
echo "Verification Complete!"
echo "=========================================="
