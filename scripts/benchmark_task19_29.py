#!/usr/bin/env python
"""
Benchmark script for Tasks 19-29 (Phase 8 Hyperbolic SSM and related components)

This script validates the implementation of:
- Task 19: Hyperbolic SSM
- Task 24: KV Cache Compression (24.2, 24.3)
- Task 25: Checkpoint
- Task 26: Dynamic Curvature Adaptation
- Task 27: Koopman-Hyperbolic Bridge
- Task 28: Numerical Safety Guards
- Task 29: Checkpoint
"""

import torch
import json
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

def benchmark_hyperbolic_ssm():
    """Benchmark Hyperbolic SSM throughput."""
    from src.models.phase8.hyperbolic_ssm import (
        HyperbolicSSMConfig, HyperbolicSSM, measure_throughput
    )
    
    results = {"task": "19", "name": "Hyperbolic SSM", "tests": []}
    
    config = HyperbolicSSMConfig(d_model=256, d_state=64)
    model = HyperbolicSSM(config)
    
    # Test different sequence lengths
    for seq_len in [256, 512, 1024]:
        x = torch.randn(4, seq_len, 256)
        
        # Warmup
        for _ in range(3):
            _ = model(x)
        
        # Measure
        start = time.time()
        for _ in range(10):
            output, _ = model(x)
        elapsed = time.time() - start
        
        tokens_per_sec = (4 * seq_len * 10) / elapsed
        
        results["tests"].append({
            "seq_len": seq_len,
            "tokens_per_sec": tokens_per_sec,
            "avg_time_ms": elapsed / 10 * 1000,
            "output_shape": list(output.shape),
        })
    
    # Get diagnostics
    diag = model.get_diagnostics()
    results["diagnostics"] = diag.to_dict()
    
    return results


def benchmark_kv_cache():
    """Benchmark KV Cache Compression."""
    from src.models.phase8.kv_cache import (
        KVCacheConfig, CompressedKVCache, create_compressed_kv_cache
    )
    
    results = {"task": "24", "name": "KV Cache Compression", "tests": []}
    
    # Test with different configurations
    for use_quant in [True, False]:
        for compression_ratio in [0.5, 0.25]:
            cache = create_compressed_kv_cache(
                d_model=256,
                max_cache_size=128,
                compression_ratio=compression_ratio,
                use_quantization=use_quant,
            )
            
            # Add some KV pairs
            for _ in range(5):
                k = torch.randn(2, 32, 256)
                v = torch.randn(2, 32, 256)
                cache.update(k, v)
            
            stats = cache.get_compression_stats()
            
            results["tests"].append({
                "use_quantization": use_quant,
                "compression_ratio": compression_ratio,
                "stats": stats,
            })
    
    return results


def benchmark_curvature_adapter():
    """Benchmark Dynamic Curvature Adaptation."""
    from src.models.phase8.curvature import CurvatureAdapter
    
    results = {"task": "26", "name": "Curvature Adapter", "tests": []}
    
    adapter = CurvatureAdapter(d_model=64, c_min=0.1, c_max=2.0)
    
    # Test with different input distributions
    for scale in [0.1, 0.5, 1.0, 2.0]:
        x = torch.randn(4, 32, 64) * scale
        c = adapter(x)
        results["tests"].append({
            "input_scale": scale,
            "curvature": float(c),
        })
    
    return results


def benchmark_koopman_bridge():
    """Benchmark Koopman-Hyperbolic Bridge."""
    from src.models.phase8.koopman_bridge import KoopmanBridge
    
    results = {"task": "27", "name": "Koopman Bridge", "tests": []}
    
    bridge = KoopmanBridge(d_model=64)
    
    # Test eigenfunction mapping
    eigenfunctions = torch.randn(4, 32, 64)
    eigenvalues = torch.randn(4, 64) + 1j * torch.randn(4, 64)
    
    start = time.time()
    hyperbolic_coords = bridge(eigenfunctions, eigenvalues)
    elapsed = time.time() - start
    
    results["tests"].append({
        "input_shape": list(eigenfunctions.shape),
        "output_shape": list(hyperbolic_coords.shape),
        "time_ms": elapsed * 1000,
        "output_norm_mean": float(hyperbolic_coords.norm(dim=-1).mean()),
        "output_max_norm": float(hyperbolic_coords.norm(dim=-1).max()),
    })
    
    return results


def benchmark_numerical_guard():
    """Benchmark Numerical Safety Guards."""
    from src.models.phase8.guard import NumericalGuard
    
    results = {"task": "28", "name": "Numerical Guard", "tests": []}
    
    guard = NumericalGuard(max_norm=0.99)
    
    # Test with vectors near boundary
    x_safe = torch.randn(4, 32, 64) * 0.3
    x_boundary = torch.randn(4, 32, 64) * 1.5
    
    y_safe = guard(x_safe)
    y_boundary = guard(x_boundary)
    
    results["tests"].append({
        "safe_input_norm": float(x_safe.norm(dim=-1).mean()),
        "safe_output_norm": float(y_safe.norm(dim=-1).mean()),
        "boundary_input_norm": float(x_boundary.norm(dim=-1).mean()),
        "boundary_output_norm": float(y_boundary.norm(dim=-1).mean()),
        "boundary_clamped": float((y_boundary.norm(dim=-1) <= 0.99 + 1e-6).float().mean()),
        "collapse_count": guard.collapse_count,
    })
    
    return results


def main():
    """Run all benchmarks and save results."""
    print("=" * 60)
    print("Task 19-29 Benchmark Suite")
    print("=" * 60)
    
    all_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tasks": [],
    }
    
    # Run benchmarks
    benchmarks = [
        ("Hyperbolic SSM", benchmark_hyperbolic_ssm),
        ("KV Cache", benchmark_kv_cache),
        ("Curvature Adapter", benchmark_curvature_adapter),
        ("Koopman Bridge", benchmark_koopman_bridge),
        ("Numerical Guard", benchmark_numerical_guard),
    ]
    
    for name, func in benchmarks:
        print(f"\nRunning {name} benchmark...")
        try:
            result = func()
            all_results["tasks"].append(result)
            print(f"  ✓ {name} completed")
        except Exception as e:
            print(f"  ✗ {name} failed: {e}")
            all_results["tasks"].append({
                "name": name,
                "error": str(e),
            })
    
    # Save results
    output_path = Path("results/benchmarks/TASK19_29_BENCHMARK_RESULTS.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Results saved to {output_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    for task in all_results["tasks"]:
        if "error" in task:
            print(f"  ✗ {task['name']}: FAILED")
        else:
            print(f"  ✓ Task {task['task']}: {task['name']} - {len(task['tests'])} tests")
    
    return all_results


if __name__ == "__main__":
    main()
