#!/bin/bash
# Phase 8 Comprehensive Benchmark for WSL Ubuntu with Triton

set -e

echo "=========================================="
echo "Phase 8 Benchmark - WSL Ubuntu + Triton"
echo "=========================================="

# 環境チェック
echo ""
echo "Checking environment..."

# Python環境の確認
if [ -d "venv_ubuntu" ]; then
    echo "✓ Found venv_ubuntu"
    source venv_ubuntu/bin/activate
else
    echo "✗ venv_ubuntu not found!"
    echo "Please create virtual environment first:"
    echo "  python3 -m venv venv_ubuntu"
    echo "  source venv_ubuntu/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# CUDA確認
echo ""
echo "Checking CUDA..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

# Triton確認
echo ""
echo "Checking Triton..."
python3 -c "import triton; print(f'Triton version: {triton.__version__}')" || {
    echo "✗ Triton not found! Installing..."
    pip install triton
}

# ベンチマーク実行
echo ""
echo "=========================================="
echo "Running comprehensive benchmark..."
echo "=========================================="

python3 scripts/benchmark_phase8_comprehensive.py

echo ""
echo "=========================================="
echo "Benchmark completed!"
echo "=========================================="
echo "Results saved to: results/benchmarks/phase8_comprehensive_benchmark.json"
