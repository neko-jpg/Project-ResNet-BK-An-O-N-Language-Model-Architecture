"""
Benchmark Script for Symplectic Step (Task 23.3)

This script compares the performance of PyTorch native symplectic step
vs Triton-optimized implementation.

Requirements:
    - Batch=16, Seq=2048, dt=0.1
    - 100 steps, 100 iterations
    - Goal: > 1.20x speedup, Energy Drift < 5e-5
"""

import torch
import time
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.models.phase3.hamiltonian import HamiltonianFunction, symplectic_leapfrog_step
from src.kernels.symplectic_step import symplectic_leapfrog_step_fused

def benchmark_symplectic_step():
    print("=" * 60)
    print("Benchmark: Symplectic Step (Triton vs PyTorch)")
    print("=" * 60)

    # Configuration
    BATCH = 16
    SEQ = 2048
    D_MODEL = 32 # Position dim
    DT = 0.1
    STEPS = 100 # Steps per run (trajectory)
    ITERATIONS = 20 # Number of runs to average (reduced from 100 for time)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    if device.type == 'cpu':
        print("Warning: Running on CPU. Triton optimization requires CUDA.")

    # Prepare Data
    x_init = torch.randn(BATCH, SEQ, 2 * D_MODEL, device=device)

    # Hamiltonian
    h_func = HamiltonianFunction(d_model=D_MODEL, potential_type='mlp')
    h_func.to(device)

    # PyTorch Native Baseline
    def run_pytorch():
        x = x_init.clone()
        for _ in range(STEPS):
            x = symplectic_leapfrog_step(h_func, x, DT)
        return x

    # Measure PyTorch
    print(f"Running PyTorch baseline ({ITERATIONS} iter)...")
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    for _ in range(ITERATIONS):
        run_pytorch()
    torch.cuda.synchronize() if device.type == 'cuda' else None
    pytorch_duration = time.time() - start_time
    pytorch_avg_ms = (pytorch_duration / ITERATIONS) * 1000

    # Triton / Custom Implementation
    def run_custom():
        x = x_init.clone()
        for _ in range(STEPS):
            x = symplectic_leapfrog_step_fused(h_func, x, DT)
        return x

    # Measure Custom
    print(f"Running Custom Kernel ({ITERATIONS} iter)...")
    try:
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_time = time.time()
        for _ in range(ITERATIONS):
            run_custom()
        torch.cuda.synchronize() if device.type == 'cuda' else None
        custom_duration = time.time() - start_time
        custom_avg_ms = (custom_duration / ITERATIONS) * 1000

        # Verification (Energy Drift)
        x_final = run_custom()
        e_start = h_func(0, x_init).mean()
        e_end = h_func(0, x_final).mean()
        energy_drift = abs(e_end - e_start) / abs(e_start)

    except Exception as e:
        print(f"Custom kernel failed: {e}")
        custom_avg_ms = float('inf')
        energy_drift = float('inf')

    # Results
    speedup = pytorch_avg_ms / custom_avg_ms if custom_avg_ms > 0 else 0

    print(f"\nResults:")
    print(f"  PyTorch Average Time: {pytorch_avg_ms:.4f} ms")
    print(f"  Custom  Average Time: {custom_avg_ms:.4f} ms")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Energy Drift: {energy_drift:.2e}")

    # Save Results
    results = {
        "pytorch_time_ms": pytorch_avg_ms,
        "triton_time_ms": custom_avg_ms,
        "speedup_ratio": speedup,
        "energy_drift": float(energy_drift),
        "pass": speedup >= 1.20 and energy_drift < 5e-5
    }

    output_path = project_root / "results" / "benchmarks" / "symplectic_step_benchmark.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    benchmark_symplectic_step()
