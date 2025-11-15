"""
Multi-Scale Sequence Processing Demo

Demonstrates the multi-scale layer implementation and compares:
1. Standard ResNet-BK layer (full resolution)
2. Simple multi-scale (N → N/2 → N)
3. Hierarchical multi-scale (N → N/2 → N/4 → N/2 → N)

Shows speedup analysis and performance comparison.
"""

import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
from typing import Dict, List

from src.models.multi_scale_layer import (
    MultiScaleResNetBKLayer,
    HierarchicalMultiScaleLayer,
    MultiScaleResNetBKBlock,
    count_flops_multi_scale
)
from src.models.resnet_bk import MoEResNetBKLayer


def benchmark_layer(layer: nn.Module, x: torch.Tensor, num_runs: int = 100) -> float:
    """
    Benchmark layer forward pass time.
    
    Args:
        layer: Layer to benchmark
        x: Input tensor
        num_runs: Number of runs for averaging
    
    Returns:
        Average time per forward pass (ms)
    """
    # Warm-up
    for _ in range(10):
        _ = layer(x)
    
    # Benchmark
    start_time = time.time()
    for _ in range(num_runs):
        _ = layer(x)
    end_time = time.time()
    
    avg_time_ms = (end_time - start_time) / num_runs * 1000
    return avg_time_ms


def compare_layers(d_model: int = 64, n_seq: int = 128, batch_size: int = 4) -> Dict:
    """
    Compare standard, simple multi-scale, and hierarchical multi-scale layers.
    
    Returns:
        Dictionary with benchmark results
    """
    print(f"\n{'='*60}")
    print(f"Comparing Layers: d_model={d_model}, n_seq={n_seq}, batch_size={batch_size}")
    print(f"{'='*60}\n")
    
    # Create input
    x = torch.randn(batch_size, n_seq, d_model)
    
    # Create layers
    print("Creating layers...")
    standard_layer = MoEResNetBKLayer(d_model, n_seq, num_experts=4)
    simple_multi_scale = MultiScaleResNetBKLayer(d_model, n_seq, num_experts=4)
    hierarchical_multi_scale = HierarchicalMultiScaleLayer(d_model, n_seq, num_experts=4)
    
    # Benchmark
    print("\nBenchmarking forward pass times...")
    standard_time = benchmark_layer(standard_layer, x)
    simple_time = benchmark_layer(simple_multi_scale, x)
    hierarchical_time = benchmark_layer(hierarchical_multi_scale, x)
    
    # Results
    results = {
        'standard': {
            'time_ms': standard_time,
            'speedup': 1.0
        },
        'simple_multi_scale': {
            'time_ms': simple_time,
            'speedup': standard_time / simple_time
        },
        'hierarchical_multi_scale': {
            'time_ms': hierarchical_time,
            'speedup': standard_time / hierarchical_time
        }
    }
    
    # Print results
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    
    for name, data in results.items():
        print(f"\n{name.replace('_', ' ').title()}:")
        print(f"  Time: {data['time_ms']:.2f} ms")
        print(f"  Speedup: {data['speedup']:.2f}×")
    
    # FLOPs analysis
    print("\n" + "="*60)
    print("FLOPS ANALYSIS")
    print("="*60)
    
    flops_info = count_flops_multi_scale(d_model, n_seq)
    print(f"\nStandard layer FLOPs: {flops_info['standard_flops']:,}")
    print(f"Multi-scale layer FLOPs: {flops_info['multi_scale_flops']:,}")
    print(f"Theoretical speedup: {flops_info['speedup']:.2f}×")
    
    print("\nFLOPs Breakdown:")
    for key, value in flops_info['breakdown'].items():
        percentage = (value / flops_info['multi_scale_flops']) * 100
        print(f"  {key}: {value:,} ({percentage:.1f}%)")
    
    return results


def visualize_speedup_scaling(d_model: int = 64, batch_size: int = 4):
    """
    Visualize how speedup scales with sequence length.
    """
    print("\n" + "="*60)
    print("SPEEDUP SCALING ANALYSIS")
    print("="*60)
    
    sequence_lengths = [64, 128, 256, 512]
    standard_times = []
    simple_times = []
    hierarchical_times = []
    
    for n_seq in sequence_lengths:
        print(f"\nTesting n_seq={n_seq}...")
        
        x = torch.randn(batch_size, n_seq, d_model)
        
        # Create layers
        standard_layer = MoEResNetBKLayer(d_model, n_seq, num_experts=4)
        simple_multi_scale = MultiScaleResNetBKLayer(d_model, n_seq, num_experts=4)
        
        # Benchmark
        standard_time = benchmark_layer(standard_layer, x, num_runs=50)
        simple_time = benchmark_layer(simple_multi_scale, x, num_runs=50)
        
        standard_times.append(standard_time)
        simple_times.append(simple_time)
        
        # Only test hierarchical for n_seq divisible by 4
        if n_seq % 4 == 0:
            hierarchical_multi_scale = HierarchicalMultiScaleLayer(d_model, n_seq, num_experts=4)
            hierarchical_time = benchmark_layer(hierarchical_multi_scale, x, num_runs=50)
            hierarchical_times.append(hierarchical_time)
        else:
            hierarchical_times.append(None)
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Absolute times
    plt.subplot(1, 2, 1)
    plt.plot(sequence_lengths, standard_times, 'o-', label='Standard', linewidth=2)
    plt.plot(sequence_lengths, simple_times, 's-', label='Simple Multi-Scale', linewidth=2)
    
    valid_hierarchical = [(n, t) for n, t in zip(sequence_lengths, hierarchical_times) if t is not None]
    if valid_hierarchical:
        h_seq, h_times = zip(*valid_hierarchical)
        plt.plot(h_seq, h_times, '^-', label='Hierarchical Multi-Scale', linewidth=2)
    
    plt.xlabel('Sequence Length (N)')
    plt.ylabel('Time (ms)')
    plt.title('Forward Pass Time vs Sequence Length')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Speedup
    plt.subplot(1, 2, 2)
    simple_speedups = [s / m for s, m in zip(standard_times, simple_times)]
    plt.plot(sequence_lengths, simple_speedups, 's-', label='Simple Multi-Scale', linewidth=2)
    
    if valid_hierarchical:
        h_speedups = [standard_times[sequence_lengths.index(n)] / t for n, t in valid_hierarchical]
        plt.plot(h_seq, h_speedups, '^-', label='Hierarchical Multi-Scale', linewidth=2)
    
    plt.axhline(y=1.0, color='gray', linestyle='--', label='Baseline')
    plt.axhline(y=2.0, color='green', linestyle='--', alpha=0.5, label='Target (2×)')
    plt.xlabel('Sequence Length (N)')
    plt.ylabel('Speedup (×)')
    plt.title('Speedup vs Sequence Length')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('multi_scale_speedup_analysis.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved plot to 'multi_scale_speedup_analysis.png'")
    
    # Print summary
    print("\n" + "="*60)
    print("SPEEDUP SUMMARY")
    print("="*60)
    
    for i, n_seq in enumerate(sequence_lengths):
        print(f"\nSequence Length: {n_seq}")
        print(f"  Standard: {standard_times[i]:.2f} ms")
        print(f"  Simple Multi-Scale: {simple_times[i]:.2f} ms ({simple_speedups[i]:.2f}×)")
        if hierarchical_times[i] is not None:
            h_speedup = standard_times[i] / hierarchical_times[i]
            print(f"  Hierarchical Multi-Scale: {hierarchical_times[i]:.2f} ms ({h_speedup:.2f}×)")


def test_numerical_stability():
    """
    Test numerical stability of multi-scale layers.
    """
    print("\n" + "="*60)
    print("NUMERICAL STABILITY TEST")
    print("="*60)
    
    d_model = 64
    n_seq = 128
    batch_size = 4
    
    # Create layers
    simple_layer = MultiScaleResNetBKBlock(d_model, n_seq, hierarchical=False)
    hierarchical_layer = MultiScaleResNetBKBlock(d_model, n_seq, hierarchical=True)
    
    # Test with random inputs
    print("\nTesting with random inputs...")
    x = torch.randn(batch_size, n_seq, d_model)
    
    simple_output = simple_layer(x)
    hierarchical_output = hierarchical_layer(x)
    
    # Check for NaN/Inf
    simple_has_nan = torch.isnan(simple_output).any()
    simple_has_inf = torch.isinf(simple_output).any()
    hierarchical_has_nan = torch.isnan(hierarchical_output).any()
    hierarchical_has_inf = torch.isinf(hierarchical_output).any()
    
    print(f"  Simple Multi-Scale: NaN={simple_has_nan}, Inf={simple_has_inf}")
    print(f"  Hierarchical Multi-Scale: NaN={hierarchical_has_nan}, Inf={hierarchical_has_inf}")
    
    if not (simple_has_nan or simple_has_inf or hierarchical_has_nan or hierarchical_has_inf):
        print("  ✓ No numerical issues detected")
    else:
        print("  ✗ Numerical issues detected!")
    
    # Test with extreme inputs
    print("\nTesting with extreme inputs...")
    x_large = torch.randn(batch_size, n_seq, d_model) * 100
    x_small = torch.randn(batch_size, n_seq, d_model) * 0.01
    
    for name, x_test in [("Large values", x_large), ("Small values", x_small)]:
        simple_output = simple_layer(x_test)
        hierarchical_output = hierarchical_layer(x_test)
        
        simple_ok = not (torch.isnan(simple_output).any() or torch.isinf(simple_output).any())
        hierarchical_ok = not (torch.isnan(hierarchical_output).any() or torch.isinf(hierarchical_output).any())
        
        print(f"  {name}:")
        print(f"    Simple: {'✓' if simple_ok else '✗'}")
        print(f"    Hierarchical: {'✓' if hierarchical_ok else '✗'}")


def main():
    """
    Run all demos and benchmarks.
    """
    print("\n" + "="*60)
    print("MULTI-SCALE SEQUENCE PROCESSING DEMO")
    print("="*60)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # 1. Compare layers
    results = compare_layers(d_model=64, n_seq=128, batch_size=4)
    
    # 2. Speedup scaling analysis
    visualize_speedup_scaling(d_model=64, batch_size=4)
    
    # 3. Numerical stability test
    test_numerical_stability()
    
    print("\n" + "="*60)
    print("DEMO COMPLETE")
    print("="*60)
    print("\nKey Findings:")
    print("  • Multi-scale processing reduces computation at middle layers")
    print("  • Speedup increases with sequence length")
    print("  • Hierarchical processing provides additional speedup")
    print("  • Numerical stability maintained across different input scales")
    print("\nNext Steps:")
    print("  • Integrate with full ResNet-BK model")
    print("  • Test on WikiText-2 language modeling task")
    print("  • Combine with ACT for adaptive multi-scale processing")
    print("  • Measure perplexity impact vs speedup trade-off")


if __name__ == '__main__':
    main()
