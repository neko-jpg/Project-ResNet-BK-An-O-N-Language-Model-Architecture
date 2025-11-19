"""
Benchmark script for Fused Associative Scan kernel.

Compares performance of fused_associative_scan vs torch.cumsum
across different sequence lengths and configurations.

Requirements: 8.3, 8.5, 6.3
"""

import torch
import time
import json
from pathlib import Path
import argparse

# Import benchmark function
try:
    from src.kernels.associative_scan import benchmark_scan, fused_associative_scan, TRITON_AVAILABLE
except ImportError:
    print("Error: Could not import fused_associative_scan")
    print("Make sure you're running from the project root directory")
    exit(1)


def benchmark_different_configs():
    """
    Benchmark fused scan with different configurations.
    
    Tests:
    - Different sequence lengths: 512, 1024, 2048, 4096, 8192
    - Different model dimensions: 256, 512, 1024
    - Different batch sizes: 1, 2, 4, 8
    
    Requirement 8.3: Test sequence lengths: 512, 1024, 2048, 4096, 8192
    Requirement 8.5: Optimize block sizes for different GPU architectures
    """
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping benchmark.")
        return {}
    
    if not TRITON_AVAILABLE:
        print("Triton not available. Skipping benchmark.")
        return {}
    
    print("=" * 80)
    print("Fused Associative Scan Benchmark")
    print("=" * 80)
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    print("=" * 80)
    
    results = {
        'device': torch.cuda.get_device_name(),
        'cuda_version': torch.version.cuda,
        'pytorch_version': torch.__version__,
        'benchmarks': []
    }
    
    # Test configurations
    seq_lengths = [512, 1024, 2048, 4096, 8192]
    d_models = [256, 512, 1024]
    batch_sizes = [1, 2, 4, 8]
    
    num_warmup = 10
    num_iterations = 100
    
    print("\n" + "=" * 80)
    print("Benchmark 1: Varying Sequence Length (d_model=512, batch_size=4)")
    print("=" * 80)
    
    for seq_len in seq_lengths:
        d_model = 512
        batch_size = 4
        
        x = torch.randn(batch_size, seq_len, d_model, device='cuda', dtype=torch.float32)
        
        # Warmup
        for _ in range(num_warmup):
            _ = torch.cumsum(x, dim=1)
            _ = fused_associative_scan(x, dim=1)
        torch.cuda.synchronize()
        
        # Benchmark torch.cumsum
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(num_iterations):
            _ = torch.cumsum(x, dim=1)
        end.record()
        torch.cuda.synchronize()
        cumsum_time = start.elapsed_time(end) / num_iterations
        
        # Benchmark fused_associative_scan
        start.record()
        for _ in range(num_iterations):
            _ = fused_associative_scan(x, dim=1)
        end.record()
        torch.cuda.synchronize()
        fused_time = start.elapsed_time(end) / num_iterations
        
        speedup = cumsum_time / fused_time
        
        result = {
            'config': 'varying_seq_len',
            'seq_len': seq_len,
            'd_model': d_model,
            'batch_size': batch_size,
            'cumsum_time_ms': cumsum_time,
            'fused_time_ms': fused_time,
            'speedup': speedup,
        }
        results['benchmarks'].append(result)
        
        print(f"Seq Length: {seq_len:5d} | "
              f"torch.cumsum: {cumsum_time:6.3f}ms | "
              f"Fused Scan: {fused_time:6.3f}ms | "
              f"Speedup: {speedup:4.2f}x")
    
    print("\n" + "=" * 80)
    print("Benchmark 2: Varying Model Dimension (seq_len=2048, batch_size=4)")
    print("=" * 80)
    
    for d_model in d_models:
        seq_len = 2048
        batch_size = 4
        
        x = torch.randn(batch_size, seq_len, d_model, device='cuda', dtype=torch.float32)
        
        # Warmup
        for _ in range(num_warmup):
            _ = torch.cumsum(x, dim=1)
            _ = fused_associative_scan(x, dim=1)
        torch.cuda.synchronize()
        
        # Benchmark
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(num_iterations):
            _ = torch.cumsum(x, dim=1)
        end.record()
        torch.cuda.synchronize()
        cumsum_time = start.elapsed_time(end) / num_iterations
        
        start.record()
        for _ in range(num_iterations):
            _ = fused_associative_scan(x, dim=1)
        end.record()
        torch.cuda.synchronize()
        fused_time = start.elapsed_time(end) / num_iterations
        
        speedup = cumsum_time / fused_time
        
        result = {
            'config': 'varying_d_model',
            'seq_len': seq_len,
            'd_model': d_model,
            'batch_size': batch_size,
            'cumsum_time_ms': cumsum_time,
            'fused_time_ms': fused_time,
            'speedup': speedup,
        }
        results['benchmarks'].append(result)
        
        print(f"D Model: {d_model:4d} | "
              f"torch.cumsum: {cumsum_time:6.3f}ms | "
              f"Fused Scan: {fused_time:6.3f}ms | "
              f"Speedup: {speedup:4.2f}x")
    
    print("\n" + "=" * 80)
    print("Benchmark 3: Varying Batch Size (seq_len=2048, d_model=512)")
    print("=" * 80)
    
    for batch_size in batch_sizes:
        seq_len = 2048
        d_model = 512
        
        x = torch.randn(batch_size, seq_len, d_model, device='cuda', dtype=torch.float32)
        
        # Warmup
        for _ in range(num_warmup):
            _ = torch.cumsum(x, dim=1)
            _ = fused_associative_scan(x, dim=1)
        torch.cuda.synchronize()
        
        # Benchmark
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(num_iterations):
            _ = torch.cumsum(x, dim=1)
        end.record()
        torch.cuda.synchronize()
        cumsum_time = start.elapsed_time(end) / num_iterations
        
        start.record()
        for _ in range(num_iterations):
            _ = fused_associative_scan(x, dim=1)
        end.record()
        torch.cuda.synchronize()
        fused_time = start.elapsed_time(end) / num_iterations
        
        speedup = cumsum_time / fused_time
        
        result = {
            'config': 'varying_batch_size',
            'seq_len': seq_len,
            'd_model': d_model,
            'batch_size': batch_size,
            'cumsum_time_ms': cumsum_time,
            'fused_time_ms': fused_time,
            'speedup': speedup,
        }
        results['benchmarks'].append(result)
        
        print(f"Batch Size: {batch_size:2d} | "
              f"torch.cumsum: {cumsum_time:6.3f}ms | "
              f"Fused Scan: {fused_time:6.3f}ms | "
              f"Speedup: {speedup:4.2f}x")
    
    # Calculate average speedup
    avg_speedup = sum(r['speedup'] for r in results['benchmarks']) / len(results['benchmarks'])
    results['average_speedup'] = avg_speedup
    
    print("\n" + "=" * 80)
    print(f"Average Speedup: {avg_speedup:.2f}x")
    
    # Requirement 8.3: Verify 3x speedup target is achieved
    if avg_speedup >= 3.0:
        print("✓ Target speedup of 3x achieved!")
        results['target_achieved'] = True
    else:
        print(f"✗ Target speedup of 3x not achieved (got {avg_speedup:.2f}x)")
        results['target_achieved'] = False
    
    print("=" * 80)
    
    return results


def test_correctness():
    """
    Test that fused_associative_scan produces correct results.
    
    Compares output against torch.cumsum to verify correctness.
    
    Requirement 6.1: Test output correctness matches torch.cumsum
    """
    print("\n" + "=" * 80)
    print("Correctness Test")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("CUDA not available. Testing CPU fallback.")
        device = 'cpu'
    else:
        device = 'cuda'
    
    # Test different shapes
    test_cases = [
        (4, 128, 64),
        (2, 512, 128),
        (1, 1024, 256),
        (8, 2048, 512),
    ]
    
    all_passed = True
    
    for batch_size, seq_len, d_model in test_cases:
        x = torch.randn(batch_size, seq_len, d_model, device=device)
        
        # Compute with torch.cumsum
        expected = torch.cumsum(x, dim=1)
        
        # Compute with fused_associative_scan
        actual = fused_associative_scan(x, dim=1)
        
        # Check correctness
        max_diff = (expected - actual).abs().max().item()
        rel_error = max_diff / (expected.abs().max().item() + 1e-8)
        
        passed = rel_error < 1e-4
        all_passed = all_passed and passed
        
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status} | Shape: ({batch_size}, {seq_len}, {d_model}) | "
              f"Max Diff: {max_diff:.2e} | Rel Error: {rel_error:.2e}")
    
    print("=" * 80)
    if all_passed:
        print("✓ All correctness tests passed!")
    else:
        print("✗ Some correctness tests failed!")
    print("=" * 80)
    
    return all_passed


def main():
    parser = argparse.ArgumentParser(description='Benchmark Fused Associative Scan')
    parser.add_argument('--output', type=str, default='results/benchmarks/fused_scan_benchmark.json',
                       help='Output file for benchmark results')
    parser.add_argument('--skip-correctness', action='store_true',
                       help='Skip correctness tests')
    parser.add_argument('--skip-benchmark', action='store_true',
                       help='Skip performance benchmarks')
    
    args = parser.parse_args()
    
    # Run correctness tests
    if not args.skip_correctness:
        correctness_passed = test_correctness()
        if not correctness_passed:
            print("\nWarning: Correctness tests failed. Benchmark results may not be reliable.")
    
    # Run benchmarks
    if not args.skip_benchmark:
        results = benchmark_different_configs()
        
        # Save results
        if results:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
