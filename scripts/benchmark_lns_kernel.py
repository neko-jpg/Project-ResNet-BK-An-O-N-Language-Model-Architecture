"""
Benchmark LNS Kernel Performance

This script benchmarks the LNS (Logarithmic Number System) kernel
against standard torch.matmul to measure:
1. Speedup (throughput improvement)
2. Numerical accuracy loss from max-log approximation
3. Memory usage
4. Power consumption (if available)

物理的直観:
LNSカーネルは乗算器(FMA)を加算器(ADD)に置き換えることで、
計算コストと消費電力を削減します。このベンチマークでは、
その効果を定量的に測定します。

Requirements: 8.3, 8.5, 3.6, 11.6, 12.1
"""

import torch
import time
import json
from pathlib import Path
from typing import Dict, List
import argparse

try:
    from src.kernels.lns_kernel import lns_matmul, TRITON_AVAILABLE
except ImportError:
    print("Error: Cannot import lns_matmul. Make sure src/kernels is in PYTHONPATH")
    exit(1)


def benchmark_speedup(
    matrix_sizes: List[tuple] = [(128, 128, 128), (256, 256, 256), (512, 512, 512), (1024, 1024, 1024)],
    num_warmup: int = 10,
    num_iterations: int = 100,
    dtype: torch.dtype = torch.float16,
) -> List[Dict]:
    """
    Benchmark LNS kernel speedup vs standard torch.matmul.
    
    物理的直観:
    行列サイズを変えながら、LNSカーネルと標準matmulの速度を比較。
    理論的には、大きな行列ほどLNSの優位性が顕著になるはずです。
    
    Args:
        matrix_sizes: List of (M, K, N) tuples for matrix dimensions
        num_warmup: Number of warmup iterations
        num_iterations: Number of benchmark iterations
        dtype: Data type for matrices
    
    Returns:
        List of benchmark results
    
    Requirements: 8.3, 8.5
    """
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark")
        return []
    
    if not TRITON_AVAILABLE:
        print("Triton not available, skipping LNS benchmark")
        return []
    
    results = []
    
    print("\n" + "="*80)
    print("LNS Kernel Speedup Benchmark")
    print("="*80)
    print(f"{'Matrix Size':<20} {'torch.matmul':<15} {'LNS Kernel':<15} {'Speedup':<10}")
    print("-"*80)
    
    for M, K, N in matrix_sizes:
        # Create test matrices (positive values for log domain)
        a = torch.abs(torch.randn(M, K, device='cuda', dtype=dtype)) + 0.1
        b = torch.abs(torch.randn(K, N, device='cuda', dtype=dtype)) + 0.1
        
        # Convert to log domain for LNS
        log_a = torch.log(a)
        log_b = torch.log(b)
        
        # Warmup
        for _ in range(num_warmup):
            _ = torch.matmul(a, b)
            _ = lns_matmul(log_a, log_b)
        torch.cuda.synchronize()
        
        # Benchmark torch.matmul
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(num_iterations):
            _ = torch.matmul(a, b)
        end.record()
        torch.cuda.synchronize()
        matmul_time = start.elapsed_time(end) / num_iterations
        
        # Benchmark LNS kernel
        start.record()
        for _ in range(num_iterations):
            _ = lns_matmul(log_a, log_b)
        end.record()
        torch.cuda.synchronize()
        lns_time = start.elapsed_time(end) / num_iterations
        
        speedup = matmul_time / lns_time
        
        result = {
            'matrix_size': f"{M}x{K}x{N}",
            'M': M,
            'K': K,
            'N': N,
            'matmul_time_ms': matmul_time,
            'lns_time_ms': lns_time,
            'speedup': speedup,
        }
        results.append(result)
        
        print(f"{result['matrix_size']:<20} {matmul_time:>12.3f}ms {lns_time:>12.3f}ms {speedup:>8.2f}x")
    
    print("-"*80)
    print()
    
    return results


def benchmark_accuracy(
    matrix_sizes: List[tuple] = [(128, 128, 128), (256, 256, 256), (512, 512, 512)],
    num_trials: int = 10,
    dtype: torch.dtype = torch.float16,
) -> List[Dict]:
    """
    Measure numerical accuracy loss from max-log approximation.
    
    物理的直観:
    Max-Log近似は支配的な項のみを保持するため、精度が低下します。
    この関数では、標準matmulとの誤差を定量的に測定します。
    
    Args:
        matrix_sizes: List of (M, K, N) tuples
        num_trials: Number of trials for averaging
        dtype: Data type for matrices
    
    Returns:
        List of accuracy results
    
    Requirements: 3.6
    """
    if not torch.cuda.is_available() or not TRITON_AVAILABLE:
        print("CUDA or Triton not available, skipping accuracy benchmark")
        return []
    
    results = []
    
    print("\n" + "="*80)
    print("LNS Kernel Accuracy Benchmark")
    print("="*80)
    print(f"{'Matrix Size':<20} {'Mean Abs Error':<20} {'Relative Error':<20}")
    print("-"*80)
    
    for M, K, N in matrix_sizes:
        errors_abs = []
        errors_rel = []
        
        for _ in range(num_trials):
            # Create test matrices
            a = torch.abs(torch.randn(M, K, device='cuda', dtype=dtype)) + 0.1
            b = torch.abs(torch.randn(K, N, device='cuda', dtype=dtype)) + 0.1
            
            # Standard matmul (ground truth)
            c_true = torch.matmul(a, b)
            
            # LNS matmul (approximation)
            log_a = torch.log(a)
            log_b = torch.log(b)
            log_c = lns_matmul(log_a, log_b)
            c_lns = torch.exp(log_c)
            
            # Compute errors
            abs_error = torch.abs(c_true - c_lns).mean().item()
            rel_error = (torch.abs(c_true - c_lns) / (torch.abs(c_true) + 1e-8)).mean().item()
            
            errors_abs.append(abs_error)
            errors_rel.append(rel_error)
        
        mean_abs_error = sum(errors_abs) / len(errors_abs)
        mean_rel_error = sum(errors_rel) / len(errors_rel)
        
        result = {
            'matrix_size': f"{M}x{K}x{N}",
            'M': M,
            'K': K,
            'N': N,
            'mean_abs_error': mean_abs_error,
            'mean_rel_error': mean_rel_error,
            'rel_error_percent': mean_rel_error * 100,
        }
        results.append(result)
        
        print(f"{result['matrix_size']:<20} {mean_abs_error:>18.6f} {mean_rel_error*100:>17.2f}%")
    
    print("-"*80)
    print()
    
    return results


def benchmark_memory(
    matrix_sizes: List[tuple] = [(512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048)],
    dtype: torch.dtype = torch.float16,
) -> List[Dict]:
    """
    Measure memory usage of LNS kernel vs standard matmul.
    
    Args:
        matrix_sizes: List of (M, K, N) tuples
        dtype: Data type for matrices
    
    Returns:
        List of memory usage results
    
    Requirements: 5.4
    """
    if not torch.cuda.is_available() or not TRITON_AVAILABLE:
        print("CUDA or Triton not available, skipping memory benchmark")
        return []
    
    results = []
    
    print("\n" + "="*80)
    print("LNS Kernel Memory Usage Benchmark")
    print("="*80)
    print(f"{'Matrix Size':<20} {'torch.matmul':<20} {'LNS Kernel':<20} {'Difference':<15}")
    print("-"*80)
    
    for M, K, N in matrix_sizes:
        # Measure torch.matmul memory
        torch.cuda.reset_peak_memory_stats()
        a = torch.abs(torch.randn(M, K, device='cuda', dtype=dtype)) + 0.1
        b = torch.abs(torch.randn(K, N, device='cuda', dtype=dtype)) + 0.1
        _ = torch.matmul(a, b)
        torch.cuda.synchronize()
        matmul_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
        
        # Measure LNS kernel memory
        torch.cuda.reset_peak_memory_stats()
        log_a = torch.log(a)
        log_b = torch.log(b)
        _ = lns_matmul(log_a, log_b)
        torch.cuda.synchronize()
        lns_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
        
        memory_diff = lns_memory - matmul_memory
        
        result = {
            'matrix_size': f"{M}x{K}x{N}",
            'M': M,
            'K': K,
            'N': N,
            'matmul_memory_mb': matmul_memory,
            'lns_memory_mb': lns_memory,
            'memory_diff_mb': memory_diff,
        }
        results.append(result)
        
        print(f"{result['matrix_size']:<20} {matmul_memory:>17.2f}MB {lns_memory:>17.2f}MB {memory_diff:>12.2f}MB")
    
    print("-"*80)
    print()
    
    return results


def save_results(results: Dict, output_path: Path):
    """Save benchmark results to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark LNS Kernel Performance")
    parser.add_argument('--output', type=str, default='results/benchmarks/lns_kernel_benchmark.json',
                        help='Output path for benchmark results')
    parser.add_argument('--num-warmup', type=int, default=10,
                        help='Number of warmup iterations')
    parser.add_argument('--num-iterations', type=int, default=100,
                        help='Number of benchmark iterations')
    parser.add_argument('--num-trials', type=int, default=10,
                        help='Number of trials for accuracy measurement')
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("LNS Kernel Benchmark Suite")
    print("="*80)
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    print(f"Triton Available: {TRITON_AVAILABLE}")
    print("="*80)
    
    # Run benchmarks
    speedup_results = benchmark_speedup(
        num_warmup=args.num_warmup,
        num_iterations=args.num_iterations,
    )
    
    accuracy_results = benchmark_accuracy(
        num_trials=args.num_trials,
    )
    
    memory_results = benchmark_memory()
    
    # Combine results
    all_results = {
        'speedup': speedup_results,
        'accuracy': accuracy_results,
        'memory': memory_results,
        'metadata': {
            'cuda_available': torch.cuda.is_available(),
            'triton_available': TRITON_AVAILABLE,
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'num_warmup': args.num_warmup,
            'num_iterations': args.num_iterations,
            'num_trials': args.num_trials,
        }
    }
    
    # Save results
    save_results(all_results, Path(args.output))
    
    # Print summary
    print("\n" + "="*80)
    print("Benchmark Summary")
    print("="*80)
    if speedup_results:
        avg_speedup = sum(r['speedup'] for r in speedup_results) / len(speedup_results)
        print(f"Average Speedup: {avg_speedup:.2f}x")
    if accuracy_results:
        avg_rel_error = sum(r['mean_rel_error'] for r in accuracy_results) / len(accuracy_results)
        print(f"Average Relative Error: {avg_rel_error*100:.2f}%")
    if memory_results:
        avg_memory_diff = sum(r['memory_diff_mb'] for r in memory_results) / len(memory_results)
        print(f"Average Memory Difference: {avg_memory_diff:.2f}MB")
    print("="*80)


if __name__ == '__main__':
    main()
