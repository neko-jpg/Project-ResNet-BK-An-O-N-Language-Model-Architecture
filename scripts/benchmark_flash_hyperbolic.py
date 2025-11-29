#!/usr/bin/env python3
"""
Flash Hyperbolic Attention Benchmark Script

Phase 8のFlash Hyperbolic AttentionとPhase 7の比較ベンチマーク。

ターゲット:
- seq=1024, 2048, 4096, 8192でのスループット測定
- Phase 7比で2倍のスループット向上
- O(N)メモリスケーリングの検証

Requirements: 31.1-31.6

出力: results/benchmarks/phase8_flash_hyperbolic_benchmark.json
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import json
import time
import argparse
from datetime import datetime
from typing import Dict, Any, List

# Check CUDA availability
if not torch.cuda.is_available():
    print("CUDA is not available. This benchmark requires a GPU.")
    sys.exit(1)


def benchmark_attention(
    attention_fn,
    batch_size: int,
    num_heads: int,
    seq_len: int,
    d_head: int,
    num_warmup: int = 10,
    num_iterations: int = 100,
    device: str = 'cuda',
    dtype: torch.dtype = torch.float16,
) -> Dict[str, float]:
    """
    アテンション関数のベンチマーク
    
    Returns:
        Dictionary with timing and memory metrics
    """
    # Create inputs
    q = torch.randn(batch_size, num_heads, seq_len, d_head, device=device, dtype=dtype)
    k = torch.randn(batch_size, num_heads, seq_len, d_head, device=device, dtype=dtype)
    v = torch.randn(batch_size, num_heads, seq_len, d_head, device=device, dtype=dtype)
    c = torch.tensor(1.0, device=device)
    beta = torch.tensor(1.0, device=device)
    
    # Warmup
    torch.cuda.reset_peak_memory_stats()
    for _ in range(num_warmup):
        try:
            _ = attention_fn(q, k, v, c, beta, causal=True)
        except Exception as e:
            return {'error': str(e)}
    torch.cuda.synchronize()
    
    # Benchmark
    torch.cuda.reset_peak_memory_stats()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(num_iterations):
        _ = attention_fn(q, k, v, c, beta, causal=True)
    end_event.record()
    torch.cuda.synchronize()
    
    # Calculate metrics
    elapsed_ms = start_event.elapsed_time(end_event)
    avg_time_ms = elapsed_ms / num_iterations
    tokens_per_sec = (batch_size * seq_len * num_iterations) / (elapsed_ms / 1000)
    peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    
    # Cleanup
    del q, k, v
    torch.cuda.empty_cache()
    
    return {
        'avg_time_ms': avg_time_ms,
        'tokens_per_sec': tokens_per_sec,
        'peak_memory_mb': peak_memory_mb,
    }


def run_benchmark(
    seq_lengths: List[int] = [1024, 2048, 4096, 8192],
    batch_size: int = 2,
    num_heads: int = 8,
    d_head: int = 64,
    num_warmup: int = 10,
    num_iterations: int = 100,
) -> Dict[str, Any]:
    """
    Flash Hyperbolic vs Phase 7 ベンチマーク実行
    """
    # Import attention functions
    try:
        from src.kernels.flash_hyperbolic_triton import flash_hyperbolic_attention
        flash_available = True
    except ImportError as e:
        print(f"Flash Hyperbolic not available: {e}")
        flash_available = False
    
    try:
        from src.kernels.hyperbolic_attention_triton import hyperbolic_attention_triton
        phase7_available = True
    except ImportError as e:
        print(f"Phase 7 Hyperbolic not available: {e}")
        phase7_available = False
    
    try:
        from src.kernels.hyperbolic_attention_fast import fast_hyperbolic_attention
        fast_available = True
    except ImportError as e:
        print(f"Fast Hyperbolic not available: {e}")
        fast_available = False
    
    results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'device': torch.cuda.get_device_name(0),
            'cuda_version': torch.version.cuda,
            'pytorch_version': torch.__version__,
            'batch_size': batch_size,
            'num_heads': num_heads,
            'd_head': d_head,
            'num_warmup': num_warmup,
            'num_iterations': num_iterations,
        },
        'seq_lengths': seq_lengths,
        'flash_hyperbolic': {},
        'phase7_hyperbolic': {},
        'fast_hyperbolic': {},
        'speedup': {},
        'memory_reduction': {},
    }
    
    for seq_len in seq_lengths:
        print(f"\n=== Benchmarking seq_len={seq_len} ===")
        
        # Flash Hyperbolic (Phase 8)
        if flash_available:
            print(f"  Flash Hyperbolic...")
            flash_result = benchmark_attention(
                flash_hyperbolic_attention,
                batch_size, num_heads, seq_len, d_head,
                num_warmup, num_iterations
            )
            results['flash_hyperbolic'][str(seq_len)] = flash_result
            if 'error' not in flash_result:
                print(f"    Time: {flash_result['avg_time_ms']:.3f} ms")
                print(f"    Throughput: {flash_result['tokens_per_sec']:.0f} tokens/sec")
                print(f"    Memory: {flash_result['peak_memory_mb']:.1f} MB")
        
        # Phase 7 Hyperbolic
        if phase7_available:
            print(f"  Phase 7 Hyperbolic...")
            phase7_result = benchmark_attention(
                hyperbolic_attention_triton,
                batch_size, num_heads, seq_len, d_head,
                num_warmup, num_iterations
            )
            results['phase7_hyperbolic'][str(seq_len)] = phase7_result
            if 'error' not in phase7_result:
                print(f"    Time: {phase7_result['avg_time_ms']:.3f} ms")
                print(f"    Throughput: {phase7_result['tokens_per_sec']:.0f} tokens/sec")
                print(f"    Memory: {phase7_result['peak_memory_mb']:.1f} MB")
        
        # Fast Hyperbolic
        if fast_available:
            print(f"  Fast Hyperbolic...")
            fast_result = benchmark_attention(
                fast_hyperbolic_attention,
                batch_size, num_heads, seq_len, d_head,
                num_warmup, num_iterations
            )
            results['fast_hyperbolic'][str(seq_len)] = fast_result
            if 'error' not in fast_result:
                print(f"    Time: {fast_result['avg_time_ms']:.3f} ms")
                print(f"    Throughput: {fast_result['tokens_per_sec']:.0f} tokens/sec")
                print(f"    Memory: {fast_result['peak_memory_mb']:.1f} MB")
        
        # Calculate speedup and memory reduction
        if flash_available and phase7_available:
            flash_res = results['flash_hyperbolic'].get(str(seq_len), {})
            phase7_res = results['phase7_hyperbolic'].get(str(seq_len), {})
            
            if 'error' not in flash_res and 'error' not in phase7_res:
                speedup = phase7_res['avg_time_ms'] / flash_res['avg_time_ms']
                memory_reduction = 1 - (flash_res['peak_memory_mb'] / phase7_res['peak_memory_mb'])
                
                results['speedup'][str(seq_len)] = speedup
                results['memory_reduction'][str(seq_len)] = memory_reduction
                
                print(f"  Speedup: {speedup:.2f}x")
                print(f"  Memory Reduction: {memory_reduction*100:.1f}%")
    
    return results


def verify_memory_scaling(
    seq_lengths: List[int] = [1024, 2048, 4096, 8192],
    batch_size: int = 2,
    num_heads: int = 8,
    d_head: int = 64,
) -> Dict[str, Any]:
    """
    O(N)メモリスケーリングの検証
    
    Requirements: 31.3, 7.3
    """
    try:
        from src.kernels.flash_hyperbolic_triton import flash_hyperbolic_attention
    except ImportError:
        return {'error': 'Flash Hyperbolic not available'}
    
    results = {
        'seq_lengths': seq_lengths,
        'memory_mb': [],
        'memory_per_token_kb': [],
        'is_linear': False,
    }
    
    c = torch.tensor(1.0, device='cuda')
    beta = torch.tensor(1.0, device='cuda')
    
    for seq_len in seq_lengths:
        q = torch.randn(batch_size, num_heads, seq_len, d_head, device='cuda', dtype=torch.float16)
        k = torch.randn(batch_size, num_heads, seq_len, d_head, device='cuda', dtype=torch.float16)
        v = torch.randn(batch_size, num_heads, seq_len, d_head, device='cuda', dtype=torch.float16)
        
        torch.cuda.reset_peak_memory_stats()
        _ = flash_hyperbolic_attention(q, k, v, c, beta, causal=True)
        torch.cuda.synchronize()
        
        peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        memory_per_token_kb = (peak_memory_mb * 1024) / (batch_size * seq_len)
        
        results['memory_mb'].append(peak_memory_mb)
        results['memory_per_token_kb'].append(memory_per_token_kb)
        
        del q, k, v
        torch.cuda.empty_cache()
    
    # Check if memory scaling is approximately linear
    # Memory per token should be roughly constant for O(N) scaling
    if len(results['memory_per_token_kb']) >= 2:
        variance = max(results['memory_per_token_kb']) / min(results['memory_per_token_kb'])
        results['is_linear'] = variance < 2.0  # Allow 2x variance
        results['variance_ratio'] = variance
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Flash Hyperbolic Attention Benchmark')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--num-heads', type=int, default=8)
    parser.add_argument('--d-head', type=int, default=64)
    parser.add_argument('--seq-lengths', type=int, nargs='+', default=[1024, 2048, 4096, 8192])
    parser.add_argument('--num-warmup', type=int, default=10)
    parser.add_argument('--num-iterations', type=int, default=100)
    parser.add_argument('--output', type=str, default='results/benchmarks/phase8_flash_hyperbolic_benchmark.json')
    args = parser.parse_args()
    
    print("=" * 60)
    print("Flash Hyperbolic Attention Benchmark")
    print("=" * 60)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    
    # Run main benchmark
    print("\n" + "=" * 60)
    print("Running Throughput Benchmark")
    print("=" * 60)
    benchmark_results = run_benchmark(
        seq_lengths=args.seq_lengths,
        batch_size=args.batch_size,
        num_heads=args.num_heads,
        d_head=args.d_head,
        num_warmup=args.num_warmup,
        num_iterations=args.num_iterations,
    )
    
    # Run memory scaling verification
    print("\n" + "=" * 60)
    print("Verifying Memory Scaling")
    print("=" * 60)
    memory_results = verify_memory_scaling(
        seq_lengths=args.seq_lengths,
        batch_size=args.batch_size,
        num_heads=args.num_heads,
        d_head=args.d_head,
    )
    benchmark_results['memory_scaling'] = memory_results
    
    if 'error' not in memory_results:
        print(f"Memory scaling is {'O(N)' if memory_results['is_linear'] else 'NOT O(N)'}")
        print(f"Variance ratio: {memory_results.get('variance_ratio', 'N/A'):.2f}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    target_speedup = 2.0
    target_memory_seq8192 = 3000  # MB
    
    speedups = benchmark_results.get('speedup', {})
    if speedups:
        avg_speedup = sum(speedups.values()) / len(speedups)
        print(f"Average Speedup: {avg_speedup:.2f}x (Target: {target_speedup}x)")
        benchmark_results['summary'] = {
            'avg_speedup': avg_speedup,
            'target_speedup': target_speedup,
            'speedup_achieved': avg_speedup >= target_speedup,
        }
    
    flash_results = benchmark_results.get('flash_hyperbolic', {})
    if '8192' in flash_results and 'error' not in flash_results['8192']:
        memory_8192 = flash_results['8192']['peak_memory_mb']
        print(f"Memory at seq=8192: {memory_8192:.1f} MB (Target: <{target_memory_seq8192} MB)")
        benchmark_results['summary']['memory_8192_mb'] = memory_8192
        benchmark_results['summary']['memory_target_achieved'] = memory_8192 < target_memory_seq8192
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(benchmark_results, f, indent=2)
    print(f"\nResults saved to: {args.output}")
    
    return benchmark_results


if __name__ == '__main__':
    main()
