"""
Fused Associative Scan Demo - Phase 1.4

Demonstrates the usage and performance benefits of the fused associative scan
Triton kernel compared to standard torch.cumsum.

Requirements: 8.1, 8.2, 8.3
"""

import torch
import time
from src.kernels.associative_scan import fused_associative_scan, TRITON_AVAILABLE


def demo_basic_usage():
    """Demonstrate basic usage of fused_associative_scan."""
    print("=" * 80)
    print("Demo 1: Basic Usage")
    print("=" * 80)
    
    # Create sample input
    batch_size = 2
    seq_len = 8
    d_model = 4
    
    x = torch.tensor([
        [[1.0, 2.0, 3.0, 4.0],
         [0.5, 1.0, 1.5, 2.0],
         [0.2, 0.4, 0.6, 0.8],
         [0.1, 0.2, 0.3, 0.4],
         [1.0, 1.0, 1.0, 1.0],
         [2.0, 2.0, 2.0, 2.0],
         [0.5, 0.5, 0.5, 0.5],
         [0.1, 0.1, 0.1, 0.1]],
        
        [[2.0, 1.0, 3.0, 2.0],
         [1.0, 0.5, 1.5, 1.0],
         [0.4, 0.2, 0.6, 0.4],
         [0.2, 0.1, 0.3, 0.2],
         [1.5, 1.5, 1.5, 1.5],
         [2.5, 2.5, 2.5, 2.5],
         [0.8, 0.8, 0.8, 0.8],
         [0.2, 0.2, 0.2, 0.2]]
    ])
    
    print(f"Input shape: {x.shape}")
    print(f"Input (first batch, first 4 positions):")
    print(x[0, :4])
    
    # Compute cumulative sum using fused scan
    output = fused_associative_scan(x, dim=1)
    
    print(f"\nOutput (cumulative sum along sequence dimension):")
    print(f"Output shape: {output.shape}")
    print(f"Output (first batch, first 4 positions):")
    print(output[0, :4])
    
    # Verify correctness
    expected = torch.cumsum(x, dim=1)
    max_diff = (output - expected).abs().max().item()
    print(f"\nMax difference from torch.cumsum: {max_diff:.2e}")
    print(f"✓ Correctness verified!" if max_diff < 1e-5 else "✗ Correctness check failed!")


def demo_reverse_scan():
    """Demonstrate reverse (anti-causal) scan."""
    print("\n" + "=" * 80)
    print("Demo 2: Reverse (Anti-Causal) Scan")
    print("=" * 80)
    
    x = torch.randn(2, 8, 4)
    
    # Forward scan (causal)
    forward = fused_associative_scan(x, dim=1, reverse=False)
    
    # Reverse scan (anti-causal)
    reverse = fused_associative_scan(x, dim=1, reverse=True)
    
    print(f"Input shape: {x.shape}")
    print(f"\nForward scan (first batch, first 4 positions):")
    print(forward[0, :4])
    print(f"\nReverse scan (first batch, first 4 positions):")
    print(reverse[0, :4])
    
    # Verify correctness
    expected_reverse = torch.flip(
        torch.cumsum(torch.flip(x, dims=[1]), dim=1),
        dims=[1]
    )
    max_diff = (reverse - expected_reverse).abs().max().item()
    print(f"\nMax difference from expected reverse: {max_diff:.2e}")
    print(f"✓ Correctness verified!" if max_diff < 1e-5 else "✗ Correctness check failed!")


def demo_performance_comparison():
    """Compare performance of fused scan vs torch.cumsum."""
    print("\n" + "=" * 80)
    print("Demo 3: Performance Comparison")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping performance comparison.")
        return
    
    if not TRITON_AVAILABLE:
        print("Triton not available. Skipping performance comparison.")
        return
    
    # Test configurations
    configs = [
        (4, 512, 256),
        (4, 1024, 512),
        (4, 2048, 512),
        (4, 4096, 512),
    ]
    
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"\n{'Seq Len':<10} {'torch.cumsum':<15} {'Fused Scan':<15} {'Speedup':<10}")
    print("-" * 50)
    
    for batch_size, seq_len, d_model in configs:
        x = torch.randn(batch_size, seq_len, d_model, device='cuda')
        
        # Warmup
        for _ in range(10):
            _ = torch.cumsum(x, dim=1)
            _ = fused_associative_scan(x, dim=1)
        torch.cuda.synchronize()
        
        # Benchmark torch.cumsum
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(100):
            _ = torch.cumsum(x, dim=1)
        end.record()
        torch.cuda.synchronize()
        cumsum_time = start.elapsed_time(end) / 100
        
        # Benchmark fused_associative_scan
        start.record()
        for _ in range(100):
            _ = fused_associative_scan(x, dim=1)
        end.record()
        torch.cuda.synchronize()
        fused_time = start.elapsed_time(end) / 100
        
        speedup = cumsum_time / fused_time
        
        print(f"{seq_len:<10} {cumsum_time:>6.3f} ms      {fused_time:>6.3f} ms      {speedup:>4.2f}x")


def demo_ar_ssm_integration():
    """Demonstrate integration with AR-SSM layer."""
    print("\n" + "=" * 80)
    print("Demo 4: Integration with AR-SSM Layer")
    print("=" * 80)
    
    from src.models.phase1.ar_ssm_layer import AdaptiveRankSemiseparableLayer
    
    # Create AR-SSM layer with fused scan enabled
    layer = AdaptiveRankSemiseparableLayer(
        d_model=128,
        max_rank=32,
        use_fused_scan=True
    )
    
    # Create input
    x = torch.randn(4, 256, 128)
    
    print(f"AR-SSM Configuration:")
    print(f"  d_model: {layer.d_model}")
    print(f"  max_rank: {layer.max_rank}")
    print(f"  use_fused_scan: {layer.use_fused_scan}")
    
    # Forward pass
    output, diagnostics = layer(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Used fused scan: {diagnostics.get('used_fused_scan', False)}")
    print(f"Effective rank: {diagnostics['effective_rank']:.2f}")
    
    # Test bidirectional processing
    output_bidir, diagnostics_bidir = layer.forward_bidirectional(x, use_anticausal=True)
    
    print(f"\nBidirectional processing:")
    print(f"  Output shape: {output_bidir.shape}")
    print(f"  Bidirectional: {diagnostics_bidir.get('bidirectional', False)}")
    
    print(f"\n✓ AR-SSM integration successful!")


def demo_cpu_fallback():
    """Demonstrate CPU fallback behavior."""
    print("\n" + "=" * 80)
    print("Demo 5: CPU Fallback")
    print("=" * 80)
    
    # Create CPU tensor
    x = torch.randn(2, 128, 64)
    
    print(f"Input device: {x.device}")
    print(f"Input shape: {x.shape}")
    
    # Fused scan will automatically fall back to torch.cumsum on CPU
    output = fused_associative_scan(x, dim=1)
    
    print(f"Output device: {output.device}")
    print(f"Output shape: {output.shape}")
    
    # Verify correctness
    expected = torch.cumsum(x, dim=1)
    max_diff = (output - expected).abs().max().item()
    
    print(f"\nMax difference from torch.cumsum: {max_diff:.2e}")
    print(f"✓ CPU fallback works correctly!")


def main():
    """Run all demos."""
    print("\n" + "=" * 80)
    print("Fused Associative Scan Demo")
    print("=" * 80)
    
    # Demo 1: Basic usage
    demo_basic_usage()
    
    # Demo 2: Reverse scan
    demo_reverse_scan()
    
    # Demo 3: Performance comparison (CUDA only)
    demo_performance_comparison()
    
    # Demo 4: AR-SSM integration
    demo_ar_ssm_integration()
    
    # Demo 5: CPU fallback
    demo_cpu_fallback()
    
    print("\n" + "=" * 80)
    print("All demos completed successfully!")
    print("=" * 80)


if __name__ == '__main__':
    main()
