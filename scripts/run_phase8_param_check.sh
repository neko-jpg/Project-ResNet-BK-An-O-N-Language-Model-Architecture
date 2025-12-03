#!/bin/bash
# Phase 8 Parameter Check - WSL Execution Script
# このスクリプトはWSL環境でPhase 8のパラメータ数を確認します

set -e  # Exit on error

echo "=========================================="
echo "Phase 8 Parameter Check (WSL)"
echo "=========================================="
echo ""

# Check if we're in WSL
if ! grep -qi microsoft /proc/version 2>/dev/null; then
    echo "⚠️  Warning: This doesn't appear to be WSL environment"
    echo "Continuing anyway..."
    echo ""
fi

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 not found"
    echo "Please install Python 3.8+ in WSL"
    exit 1
fi

echo "✓ Python found: $(python3 --version)"
echo ""

# Check if venv exists
if [ -d "venv_ubuntu" ]; then
    echo "✓ Virtual environment found: venv_ubuntu"
    source venv_ubuntu/bin/activate
    echo "✓ Virtual environment activated"
elif [ -d ".venv" ]; then
    echo "✓ Virtual environment found: .venv"
    source .venv/bin/activate
    echo "✓ Virtual environment activated"
else
    echo "⚠️  No virtual environment found"
    echo "Using system Python..."
fi

echo ""

# Check PyTorch
if python3 -c "import torch" 2>/dev/null; then
    echo "✓ PyTorch installed"
    python3 -c "import torch; print(f'  Version: {torch.__version__}')"
else
    echo "❌ Error: PyTorch not installed"
    echo "Please install PyTorch: pip install torch"
    exit 1
fi

echo ""

# Check config file
CONFIG_FILE="${1:-configs/phase8_max_push.yaml}"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "✓ Config file found: $CONFIG_FILE"
echo ""

# Run parameter check
echo "=========================================="
echo "Running Parameter Check..."
echo "=========================================="
echo ""

python3 scripts/check_phase8_params_wsl.py --config "$CONFIG_FILE"

echo ""
echo "=========================================="
echo "Parameter Check Complete!"
echo "=========================================="
