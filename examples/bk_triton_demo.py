"""
BK-Core Triton Acceleration Demo

Demonstrates how to use the Triton-accelerated BK-Core implementation.

This example shows:
1. How to enable/disable Triton acceleration
2. Performance comparison between PyTorch and Triton
3. Numerical equivalence verification
"""

import torch
import time
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.bk_core import (
    BKCoreFunction,
    set_triton_mode,
    get_triton_mode,
)


def demo_basic_usage():
    """Demonstrate basic usage of BK-Core with Triton."""
    print("=" * 70)
    print("BK-Core Triton Acceleration Demo")
    print("=" * 70)
    print()
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print()
    
    # Generate test data
    batch_size = 4
    seq_len = 1024
    
    torch.manual_seed(42)
    he_diag = torch.randn(batch_size, seq_len, device=device)
    h0_super = torch.randn(batch_size, seq_len - 1, device=device)
    h0_sub = torch.randn(batch_size, seq_len - 1, device=device)
    z = torch.tensor(0.1 + 0.1j, dtype=torch.complex64, device=device)
    
    print(f"Input shape: ({batch_size}, {seq_len})")
    print()
    
    # ========================================================================
    # Example 1: Auto-detection (default behavior)
    # ========================================================================
    print("Example 1: Auto-detection (default)")
    print("-" * 70)
    
    output = BKCoreFunction.apply(he_diag, h0_super, h0_sub, z)
    print(f"Output shape: {output.shape}")
    print(f"Triton enabled: {get_triton_mode()}")
    print()
    
    # ========================================================================
    # Example 2: Explicitly enable Triton
    # ========================================================================
    print("Example 2: Explicitly enable Triton")
    print("-" * 70)
    
    set_triton_mode(True)
    print(f"Triton mode set to: {get_triton_mode()}")
    
    output_triton = BKCoreFunction.apply(he_diag, h0_super, h0_sub, z, True)
    print(f"Output shape: {output_triton.shape}")
    print()
    
    # ========================================================================
    # Example 3: Explicitly disable Triton (use PyTorch)
    # ========================================================================
    print("Example 3: Explicitly disable Triton (use PyTorch)")
    print("-" * 70)
    
    set_triton_mode(False)
    print(f"Triton mode set to: {get_triton_mode()}")
    
    output_pytorch = BKCoreFunction.apply(he_diag, h0_super, h0_sub, z, False)
    print(f"Output shape: {output_pytorch.shape}")
    print()
    
    # ========================================================================
    # Example 4: Verify numerical equivalence
    # ========================================================================
    print("Example 4: Verify numerical equivalence")
    print("-" * 70)
    
    diff = (output_triton - output_pytorch).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"Maximum difference: {max_diff:.2e}")
    print(f"Mean difference: {mean_diff:.2e}")
    
    if max_diff < 1e-4:
        print("✓ Outputs are numerically equivalent")
    else:
        print("✗ Outputs differ significantly")
    print()
    
    # ========================================================================
    # Example 5: Performance comparison
    # ========================================================================
    print("Example 5: Performance comparison")
    print("-" * 70)
    
    num_runs = 50
    warmup_runs = 5
    
    # Warmup
    for _ in range(warmup_runs):
        _ = BKCoreFunction.apply(he_diag, h0_super, h0_sub, z, False)
        if device == "cuda":
            torch.cuda.synchronize()
    
    # Benchmark PyTorch
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_runs):
        _ = BKCoreFunction.apply(he_diag, h0_super, h0_sub, z, False)
    if device == "cuda":
        torch.cuda.synchronize()
    pytorch_time = (time.perf_counter() - start) / num_runs * 1000
    
    # Check if Triton is available
    try:
        from src.kernels.bk_scan import is_triton_available
        triton_available = is_triton_available()
    except Exception:
        triton_available = False
    
    if triton_available:
        # Warmup
        for _ in range(warmup_runs):
            _ = BKCoreFunction.apply(he_diag, h0_super, h0_sub, z, True)
            if device == "cuda":
                torch.cuda.synchronize()
        
        # Benchmark Triton
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_runs):
            _ = BKCoreFunction.apply(he_diag, h0_super, h0_sub, z, True)
        if device == "cuda":
            torch.cuda.synchronize()
        triton_time = (time.perf_counter() - start) / num_runs * 1000
        
        speedup = pytorch_time / triton_time
        
        print(f"PyTorch time: {pytorch_time:.3f} ms")
        print(f"Triton time:  {triton_time:.3f} ms")
        print(f"Speedup:      {speedup:.2f}x")
        
        if speedup >= 1.5:
            print(f"✓ Triton is {speedup:.2f}x faster")
        else:
            print(f"⚠ Triton speedup is only {speedup:.2f}x")
    else:
        print("Triton not available, skipping performance comparison")
        print(f"PyTorch time: {pytorch_time:.3f} ms")
    
    print()


def demo_gradient_computation():
    """Demonstrate gradient computation with Triton."""
    print("=" * 70)
    print("Gradient Computation Demo")
    print("=" * 70)
    print()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Generate test data with gradients
    batch_size = 2
    seq_len = 512
    
    torch.manual_seed(42)
    he_diag = torch.randn(batch_size, seq_len, device=device, requires_grad=True)
    h0_super = torch.randn(batch_size, seq_len - 1, device=device)
    h0_sub = torch.randn(batch_size, seq_len - 1, device=device)
    z = torch.tensor(0.1 + 0.1j, dtype=torch.complex64, device=device)
    
    # Forward pass
    output = BKCoreFunction.apply(he_diag, h0_super, h0_sub, z)
    
    # Compute loss (sum of outputs)
    loss = output.sum()
    
    # Backward pass
    loss.backward()
    
    print(f"Input shape: {he_diag.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Gradient shape: {he_diag.grad.shape}")
    print(f"Gradient norm: {he_diag.grad.norm().item():.4f}")
    print()
    
    # Check for NaN/Inf in gradients
    has_nan = torch.isnan(he_diag.grad).any().item()
    has_inf = torch.isinf(he_diag.grad).any().item()
    
    if not has_nan and not has_inf:
        print("✓ Gradients are finite and well-behaved")
    else:
        print("✗ Gradients contain NaN or Inf")
    
    print()


def main():
    """Run all demos."""
    demo_basic_usage()
    demo_gradient_computation()
    
    print("=" * 70)
    print("Demo completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
