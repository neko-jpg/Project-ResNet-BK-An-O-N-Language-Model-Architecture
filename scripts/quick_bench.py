#!/usr/bin/env python3
"""Quick Triton benchmark test."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import time
import statistics
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

def bench(name, func, args, iterations=30):
    # Warmup
    for _ in range(5):
        _ = func(*args)
        if device.type == 'cuda':
            torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(iterations):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = func(*args)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    
    mean_ms = statistics.mean(times)
    std_ms = statistics.stdev(times)
    print(f"  {name}: {mean_ms:.3f}ms ±{std_ms:.3f}")
    return mean_ms

print("\n=== SSM Benchmark ===")
batch, seq, d_model, d_state = 8, 512, 256, 64
x = torch.randn(batch, seq, d_model, device=device)

try:
    # Baseline SSM (sequential)
    from src.models.phase8.ar_ssm_fusion import AdaptiveRankSSM
    ssm_seq = AdaptiveRankSSM(d_model, d_state, max_rank=16).to(device).eval()

    def ssm_seq_fn():
        with torch.no_grad():
            return ssm_seq(x)

    baseline = bench("Sequential SSM", ssm_seq_fn, ())

    # Optimized SSM (parallel scan)
    from src.kernels.low_rank_ssm_scan import LowRankSSMScan
    ssm_par = LowRankSSMScan(d_model, d_state, rank=16).to(device).eval()

    def ssm_par_fn():
        with torch.no_grad():
            return ssm_par(x)

    optimized = bench("Parallel SSM", ssm_par_fn, ())
    print(f"  >> Speedup: {baseline/optimized:.2f}x")
except Exception as e:
    print(f"  Error: {e}")

print("\n=== Möbius Addition Benchmark ===")
x = torch.randn(batch, seq, d_model, device=device) * 0.4
a = torch.randn(batch, seq, d_model, device=device) * 0.4

try:
    # Baseline (PyTorch)
    def mobius_baseline():
        x2 = (x * x).sum(dim=-1, keepdim=True)
        a2 = (a * a).sum(dim=-1, keepdim=True)
        xa = (x * a).sum(dim=-1, keepdim=True)
        numer = (1 + 2 * xa + a2) * x + (1 - x2) * a
        denom = 1 + 2 * xa + x2 * a2
        return numer / torch.clamp(denom, min=1e-7)

    baseline = bench("PyTorch", mobius_baseline, ())

    # Optimized (Triton)
    from src.kernels.hyperbolic_mobius_chain import mobius_add_fused
    def mobius_triton():
        return mobius_add_fused(x, a, 1.0)

    optimized = bench("Triton", mobius_triton, ())
    print(f"  >> Speedup: {baseline/optimized:.2f}x")
except Exception as e:
    print(f"  Error: {e}")

print("\n=== Hyperbolic Distance Benchmark ===")
x = torch.randn(batch, seq, d_model, device=device) * 0.4

try:
    # Baseline
    def dist_baseline():
        x_norm = x.norm(dim=-1)
        scaled = torch.clamp(x_norm, max=0.999)
        return 2.0 * torch.atanh(scaled)

    baseline = bench("PyTorch", dist_baseline, ())

    # Optimized (Triton)
    from src.kernels.hyperbolic_distance_batch import BatchedHyperbolicDistance
    dist_opt = BatchedHyperbolicDistance(curvature=1.0, use_triton=True)

    def dist_triton():
        return dist_opt(x)

    optimized = bench("Triton", dist_triton, ())
    print(f"  >> Speedup: {baseline/optimized:.2f}x")
except Exception as e:
    print(f"  Triton Error: {e}")
    # Fall back to PyTorch-only
    from src.kernels.hyperbolic_distance_batch import BatchedHyperbolicDistance
    dist_pytorch = BatchedHyperbolicDistance(curvature=1.0, use_triton=False)
    def dist_pt():
        return dist_pytorch(x)
    optimized = bench("PyTorch (module)", dist_pt, ())

print("\n=== Cache Benchmark ===")
x = torch.randn(batch, seq, d_model, device=device)

try:
    def compute_heavy(x):
        result = torch.zeros(x.shape[0], x.shape[1], device=x.device)
        for _ in range(10):
            result = result + x.norm(dim=-1)
        return result

    baseline = bench("No Cache", lambda: compute_heavy(x), ())

    # With cache (repeated input hits cache)
    from src.kernels.green_function_cache import GreenFunctionCache
    cache = GreenFunctionCache(cache_size=128)
    cache.get_or_compute(x, compute_heavy)  # Prime

    def cached():
        return cache.get_or_compute(x, compute_heavy)

    optimized = bench("With Cache", cached, ())
    print(f"  >> Speedup: {baseline/optimized:.2f}x")
    print(f"  Cache stats: {cache.get_stats()}")
except Exception as e:
    print(f"  Error: {e}")

print("\n=== Summary ===")
print("Done!")
