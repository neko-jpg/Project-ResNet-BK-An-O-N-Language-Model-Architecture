"""
Demonstration of Adaptive Rank Semiseparable Layer (AR-SSM)

This example shows how to use the AR-SSM layer for efficient sequence processing
with adaptive rank adjustment based on input complexity.

物理的直観 (Physical Intuition):
AR-SSMは、入力信号の複雑度に応じて計算資源を動的に調整します。
簡単なトークンには低ランク（少ない計算）、複雑な文脈には高ランク（多くの計算）を割り当てます。

Usage:
    python examples/demo_ar_ssm.py
"""

import torch
import matplotlib.pyplot as plt
import numpy as np

from src.models.phase1 import AdaptiveRankSemiseparableLayer, Phase1Config


def demo_basic_usage():
    """Demonstrate basic AR-SSM usage."""
    print("=" * 60)
    print("Demo 1: Basic AR-SSM Usage")
    print("=" * 60)
    
    # Create AR-SSM layer
    layer = AdaptiveRankSemiseparableLayer(
        d_model=128,
        max_rank=32,
        min_rank=4,
        l1_regularization=0.001,
        use_fused_scan=False,  # Use torch.cumsum for CPU
    )
    
    # Create input
    batch_size = 2
    seq_len = 256
    x = torch.randn(batch_size, seq_len, 128)
    
    print(f"Input shape: {x.shape}")
    print(f"Max rank: {layer.max_rank}")
    print(f"Min rank: {layer.min_rank}")
    
    # Forward pass
    y, diagnostics = layer(x)
    
    print(f"Output shape: {y.shape}")
    print(f"Effective rank: {diagnostics['effective_rank'].item():.2f}")
    print(f"Gate L1 loss: {diagnostics['gate_l1_loss'].item():.6f}")
    
    # Memory usage
    memory_info = layer.get_memory_usage(batch_size, seq_len)
    print(f"\nMemory Usage:")
    print(f"  AR-SSM: {memory_info['activation_memory_mb']:.2f} MB")
    print(f"  Attention (O(N²)): {memory_info['attention_memory_mb']:.2f} MB")
    print(f"  Reduction: {memory_info['memory_reduction_vs_attention']:.1%}")
    
    print("\n✓ Basic usage demo completed\n")


def demo_complexity_adaptation():
    """Demonstrate adaptive rank based on input complexity."""
    print("=" * 60)
    print("Demo 2: Complexity-Based Rank Adaptation")
    print("=" * 60)
    
    layer = AdaptiveRankSemiseparableLayer(
        d_model=64,
        max_rank=16,
        min_rank=4,
        use_fused_scan=False,
    )
    
    # Create inputs with different complexity levels
    seq_len = 128
    
    # Simple input (low complexity): constant values
    x_simple = torch.ones(1, seq_len, 64) * 0.5
    
    # Complex input (high complexity): random noise
    x_complex = torch.randn(1, seq_len, 64)
    
    # Medium input: sine wave
    t = torch.linspace(0, 4 * np.pi, seq_len).unsqueeze(0).unsqueeze(2)
    x_medium = torch.sin(t).expand(1, seq_len, 64)
    
    # Process each input
    inputs = [
        ("Simple (constant)", x_simple),
        ("Medium (sine wave)", x_medium),
        ("Complex (random)", x_complex),
    ]
    
    print(f"\nInput complexity vs Effective rank:")
    print("-" * 50)
    
    for name, x in inputs:
        _, diagnostics = layer(x)
        effective_rank = diagnostics['effective_rank'].item()
        gates = diagnostics['gates']
        gate_mean = gates.mean().item()
        gate_std = gates.std().item()
        
        print(f"{name:25s}: rank={effective_rank:5.2f}, "
              f"gate_mean={gate_mean:.3f}, gate_std={gate_std:.3f}")
    
    print("\n✓ Complexity adaptation demo completed\n")


def demo_rank_scheduling():
    """Demonstrate rank scheduling for curriculum learning."""
    print("=" * 60)
    print("Demo 3: Rank Scheduling (Curriculum Learning)")
    print("=" * 60)
    
    layer = AdaptiveRankSemiseparableLayer(
        d_model=64,
        max_rank=32,
        min_rank=4,
        use_fused_scan=False,
    )
    
    warmup_steps = 1000
    steps = [0, 250, 500, 750, 1000, 1500]
    
    print(f"\nRank schedule (warmup_steps={warmup_steps}):")
    print("-" * 50)
    
    for step in steps:
        layer.update_rank_schedule(step, warmup_steps)
        print(f"Step {step:4d}: current_max_rank = {layer.current_max_rank:2d}")
    
    print("\n✓ Rank scheduling demo completed\n")


def demo_memory_efficiency():
    """Demonstrate memory efficiency vs standard attention."""
    print("=" * 60)
    print("Demo 4: Memory Efficiency Analysis")
    print("=" * 60)
    
    layer = AdaptiveRankSemiseparableLayer(
        d_model=128,
        max_rank=32,
        use_fused_scan=False,
    )
    
    sequence_lengths = [128, 256, 512, 1024, 2048]
    batch_size = 2
    
    print(f"\nMemory usage vs sequence length (batch_size={batch_size}):")
    print("-" * 70)
    print(f"{'Seq Len':>8s} | {'AR-SSM (MB)':>12s} | {'Attention (MB)':>15s} | {'Reduction':>10s}")
    print("-" * 70)
    
    for seq_len in sequence_lengths:
        memory_info = layer.get_memory_usage(batch_size, seq_len)
        ar_ssm_mb = memory_info['activation_memory_mb']
        attention_mb = memory_info['attention_memory_mb']
        reduction = memory_info['memory_reduction_vs_attention']
        
        print(f"{seq_len:8d} | {ar_ssm_mb:12.2f} | {attention_mb:15.2f} | {reduction:9.1%}")
    
    print("\n✓ Memory efficiency demo completed\n")


def demo_gradient_flow():
    """Demonstrate gradient flow through AR-SSM."""
    print("=" * 60)
    print("Demo 5: Gradient Flow Verification")
    print("=" * 60)
    
    layer = AdaptiveRankSemiseparableLayer(
        d_model=64,
        max_rank=16,
        use_fused_scan=False,
    )
    
    # Create input with gradient tracking
    x = torch.randn(2, 128, 64, requires_grad=True)
    
    # Forward pass
    y, diagnostics = layer(x)
    
    # Compute loss
    loss = y.sum() + diagnostics['gate_l1_loss']
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    print(f"\nGradient statistics:")
    print("-" * 50)
    print(f"Input gradient norm: {x.grad.norm().item():.4f}")
    
    for name, param in layer.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"{name:30s}: {grad_norm:.4f}")
    
    print("\n✓ Gradient flow demo completed\n")


def demo_from_config():
    """Demonstrate creating AR-SSM from Phase1Config."""
    print("=" * 60)
    print("Demo 6: Creating AR-SSM from Configuration")
    print("=" * 60)
    
    # Create configuration for 8GB VRAM
    config = Phase1Config.for_hardware(vram_gb=8.0)
    
    print(f"Configuration for 8GB VRAM:")
    print(f"  ar_ssm_max_rank: {config.ar_ssm_max_rank}")
    print(f"  ar_ssm_min_rank: {config.ar_ssm_min_rank}")
    print(f"  ar_ssm_l1_regularization: {config.ar_ssm_l1_regularization}")
    print(f"  ar_ssm_use_fused_scan: {config.ar_ssm_use_fused_scan}")
    
    # Create layer from config
    layer = AdaptiveRankSemiseparableLayer.from_config(
        config=config,
        d_model=128,
    )
    
    # Test forward pass
    x = torch.randn(2, 256, 128)
    y, diagnostics = layer(x)
    
    print(f"\nLayer created successfully:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y.shape}")
    print(f"  Effective rank: {diagnostics['effective_rank'].item():.2f}")
    
    print("\n✓ Configuration demo completed\n")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 60)
    print("AR-SSM Layer Demonstration")
    print("Adaptive Rank Semiseparable Layer for Efficient Sequence Processing")
    print("=" * 60 + "\n")
    
    # Run demos
    demo_basic_usage()
    demo_complexity_adaptation()
    demo_rank_scheduling()
    demo_memory_efficiency()
    demo_gradient_flow()
    demo_from_config()
    
    print("=" * 60)
    print("All demonstrations completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
