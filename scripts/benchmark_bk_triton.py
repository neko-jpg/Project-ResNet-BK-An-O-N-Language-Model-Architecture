"""
BK-Core Triton Performance Benchmark

Compares PyTorch vmap implementation vs Triton kernel implementation.

Measurement conditions:
- Batch size: 16
- Sequence length: 4096
- Number of runs: 100
- Device: CUDA (if available)

Success criteria:
- Triton version must be 3.0x+ faster than PyTorch version
- Results logged to JSON format
"""

import torch
import time
import json
import sys
from pathlib import Path
from datetime import datetime
import platform

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.bk_core import BKCoreFunction, set_triton_mode, get_triton_mode


def benchmark_bk_core(
    batch_size: int = 16,
    seq_len: int = 4096,
    num_runs: int = 100,
    warmup_runs: int = 10,
    device: str = "cuda",
):
    """
    Benchmark BK-Core with both PyTorch and Triton implementations.
    
    Args:
        batch_size: Batch size for testing
        seq_len: Sequence length
        num_runs: Number of benchmark runs
        warmup_runs: Number of warmup runs (not counted)
        device: Device to run on
    
    Returns:
        results: Dictionary with benchmark results
    """
    print(f"BK-Core Performance Benchmark")
    print(f"=" * 60)
    print(f"Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Number of runs: {num_runs}")
    print(f"  Device: {device}")
    print()
    
    # Check device availability
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"
    
    # Generate test data
    torch.manual_seed(42)
    he_diag = torch.randn(batch_size, seq_len, device=device)
    h0_super = torch.randn(batch_size, seq_len - 1, device=device)
    h0_sub = torch.randn(batch_size, seq_len - 1, device=device)
    z = torch.tensor(0.1 + 0.1j, dtype=torch.complex64, device=device)
    
    # Get GPU info if available
    gpu_name = "Unknown"
    if device == "cuda" and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
    
    results = {
        "config": {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "num_runs": num_runs,
            "warmup_runs": warmup_runs,
            "device": device,
            "gpu_name": gpu_name,
            "platform": platform.system(),
            "timestamp": datetime.now().isoformat(),
        },
        "pytorch": {},
        "triton": {},
        "speedup": 0.0,
        "success": False,
    }
    
    # ========================================================================
    # Benchmark PyTorch Implementation
    # ========================================================================
    print("Benchmarking PyTorch (vmap) implementation...")
    set_triton_mode(False)
    
    # Warmup
    for _ in range(warmup_runs):
        _ = BKCoreFunction.apply(he_diag, h0_super, h0_sub, z, False)
        if device == "cuda":
            torch.cuda.synchronize()
    
    # Benchmark
    times_pytorch = []
    for i in range(num_runs):
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        
        output = BKCoreFunction.apply(he_diag, h0_super, h0_sub, z, False)
        
        if device == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()
        
        times_pytorch.append((end - start) * 1000)  # Convert to ms
        
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i + 1}/{num_runs}")
    
    pytorch_mean = sum(times_pytorch) / len(times_pytorch)
    pytorch_std = (sum((t - pytorch_mean) ** 2 for t in times_pytorch) / len(times_pytorch)) ** 0.5
    
    results["pytorch"] = {
        "mean_ms": pytorch_mean,
        "std_ms": pytorch_std,
        "min_ms": min(times_pytorch),
        "max_ms": max(times_pytorch),
    }
    
    print(f"  Mean time: {pytorch_mean:.3f} ± {pytorch_std:.3f} ms")
    print()
    
    # ========================================================================
    # Benchmark Triton Implementation
    # ========================================================================
    print("Benchmarking Triton implementation...")
    
    # Check if Triton is available
    try:
        from src.kernels.bk_scan import is_triton_available
        if not is_triton_available():
            print("  Triton not available, skipping...")
            results["triton"]["available"] = False
            return results
    except Exception as e:
        print(f"  Failed to import Triton: {e}")
        results["triton"]["available"] = False
        return results
    
    set_triton_mode(True)
    results["triton"]["available"] = True
    
    # Warmup
    for _ in range(warmup_runs):
        _ = BKCoreFunction.apply(he_diag, h0_super, h0_sub, z, True)
        if device == "cuda":
            torch.cuda.synchronize()
    
    # Benchmark
    times_triton = []
    for i in range(num_runs):
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        
        output = BKCoreFunction.apply(he_diag, h0_super, h0_sub, z, True)
        
        if device == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()
        
        times_triton.append((end - start) * 1000)  # Convert to ms
        
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i + 1}/{num_runs}")
    
    triton_mean = sum(times_triton) / len(times_triton)
    triton_std = (sum((t - triton_mean) ** 2 for t in times_triton) / len(times_triton)) ** 0.5
    
    results["triton"] = {
        "available": True,
        "mean_ms": triton_mean,
        "std_ms": triton_std,
        "min_ms": min(times_triton),
        "max_ms": max(times_triton),
    }
    
    print(f"  Mean time: {triton_mean:.3f} ± {triton_std:.3f} ms")
    print()
    
    # ========================================================================
    # Compute Speedup
    # ========================================================================
    speedup = pytorch_mean / triton_mean
    results["speedup"] = speedup
    results["success"] = speedup >= 3.0
    
    # Add KPI achievement metrics
    results["kpi_achievement"] = {
        "target_speedup": 3.0,
        "achieved_speedup": speedup,
        "speedup_ratio": speedup / 3.0,
        "status": "PASS" if speedup >= 3.0 else "FAIL",
    }
    
    print(f"Results:")
    print(f"  PyTorch: {pytorch_mean:.3f} ms")
    print(f"  Triton:  {triton_mean:.3f} ms")
    print(f"  Speedup: {speedup:.2f}x")
    print()
    
    if results["success"]:
        print(f"✓ SUCCESS: Triton is {speedup:.2f}x faster (target: 3.0x+)")
    else:
        print(f"✗ FAILED: Triton is only {speedup:.2f}x faster (target: 3.0x+)")
    
    return results


def main():
    """Run benchmark and save results."""
    # Run benchmark
    results = benchmark_bk_core(
        batch_size=16,
        seq_len=4096,
        num_runs=100,
        warmup_runs=10,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    
    # Save results
    output_dir = Path(__file__).parent.parent / "results" / "benchmarks"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "bk_triton_benchmark.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print()
    print(f"Results saved to: {output_file}")
    
    # Exit with appropriate code
    sys.exit(0 if results.get("success", False) else 1)


if __name__ == "__main__":
    main()
