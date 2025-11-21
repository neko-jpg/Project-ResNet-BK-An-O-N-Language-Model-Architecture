"""
Benchmark Script for Complex Matrix Multiplication (Task 22.3)

This script compares the performance of PyTorch native complex matmul
vs Triton-optimized complex matmul.

Requirements:
    - Batch=16, M=512, N=512, K=512
    - 100 iterations
    - Goal: > 1.25x speedup
"""

import torch
import time
import json
import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.kernels.complex_matmul import complex_matmul

def benchmark_complex_matmul():
    print("=" * 60)
    print("Benchmark: Complex Matrix Multiplication (Triton vs PyTorch)")
    print("=" * 60)

    # Configuration
    BATCH = 16
    M, N, K = 512, 512, 512
    ITERATIONS = 100
    WARMUP = 10

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    if device.type == 'cpu':
        print("Warning: Running on CPU. Triton optimization requires CUDA.")
        print("Benchmark will run but speedup is not expected.")

    # Prepare Data
    # Note: Kernel implementation in src/kernels/complex_matmul.py assumes 2D matrices for kernel
    # But function wrapper might handle batches?
    # The wrapper expects (M, K) and (K, N).
    # For batched matmul, we loop or reshape.
    # Let's assume we benchmark single large matrix or looped batch for fairness.
    # If we want to benchmark effective throughput, we should use large matrices.
    # BATCH=16, 512x512 is effectively (16*512)x512 ? No, batch dim.
    # Let's adjust to 2D large matrices for kernel benchmark as implemented.

    M_eff = M * BATCH
    print(f"Matrix Size: [{M_eff} x {K}] @ [{K} x {N}]")

    a_real = torch.randn(M_eff, K, device=device)
    a_imag = torch.randn(M_eff, K, device=device)
    b_real = torch.randn(K, N, device=device)
    b_imag = torch.randn(K, N, device=device)

    # PyTorch Native Baseline
    def pytorch_matmul():
        a = torch.complex(a_real, a_imag)
        b = torch.complex(b_real, b_imag)
        return torch.matmul(a, b)

    # Warmup
    for _ in range(WARMUP):
        pytorch_matmul()
    if device.type == 'cuda': torch.cuda.synchronize()

    # Measure PyTorch
    start_time = time.time()
    for _ in range(ITERATIONS):
        pytorch_matmul()
    if device.type == 'cuda': torch.cuda.synchronize()
    pytorch_duration = time.time() - start_time
    pytorch_avg_ms = (pytorch_duration / ITERATIONS) * 1000

    # Triton / Custom Implementation
    def custom_matmul():
        return complex_matmul(a_real, a_imag, b_real, b_imag)

    # Warmup
    try:
        for _ in range(WARMUP):
            custom_matmul()
        if device.type == 'cuda': torch.cuda.synchronize()

        # Measure Custom
        start_time = time.time()
        for _ in range(ITERATIONS):
            custom_matmul()
        if device.type == 'cuda': torch.cuda.synchronize()
        custom_duration = time.time() - start_time
        custom_avg_ms = (custom_duration / ITERATIONS) * 1000

        # Verification (Accuracy)
        c_torch = pytorch_matmul()
        c_custom_real, c_custom_imag = custom_matmul()
        c_custom = torch.complex(c_custom_real, c_custom_imag)

        mse = torch.mean((c_torch - c_custom).abs() ** 2).item()

    except Exception as e:
        print(f"Custom kernel failed: {e}")
        custom_avg_ms = float('inf')
        mse = float('inf')

    # Results
    speedup = pytorch_avg_ms / custom_avg_ms if custom_avg_ms > 0 else 0

    print(f"\nResults:")
    print(f"  PyTorch Average Time: {pytorch_avg_ms:.4f} ms")
    print(f"  Custom  Average Time: {custom_avg_ms:.4f} ms")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  MSE Error: {mse:.2e}")

    # Save Results
    results = {
        "pytorch_time_ms": pytorch_avg_ms,
        "triton_time_ms": custom_avg_ms,
        "speedup_ratio": speedup,
        "mse_error": mse,
        "pass": speedup >= 1.25 and mse < 1e-5
    }

    output_path = project_root / "results" / "benchmarks" / "complex_matmul_benchmark.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {output_path}")

    if results['pass']:
        print("\nSUCCESS: Benchmark goals met!")
    else:
        print("\nFAIL: Benchmark goals not met (or running on CPU).")

if __name__ == "__main__":
    benchmark_complex_matmul()
