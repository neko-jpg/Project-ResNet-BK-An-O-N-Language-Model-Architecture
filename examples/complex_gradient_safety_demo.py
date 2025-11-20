"""
Complex Gradient Safety Demo for Phase 2

Demonstrates:
1. NonHermitian potential with complex gradients
2. Gradient safety mechanisms (clipping, NaN/Inf handling)
3. Training loop with gradient monitoring

Requirements: Phase 2 Task 2 (Complex Gradient Safety)
"""

import torch
import torch.nn as nn
import torch.optim as optim

from src.models.phase2 import (
    NonHermitianPotential,
    DissipativeBKLayer,
    GradientSafetyModule,
    safe_complex_backward,
)


def demo_nonhermitian_potential():
    """Demo 1: NonHermitian potential with complex gradients."""
    print("=" * 60)
    print("Demo 1: NonHermitian Potential")
    print("=" * 60)
    
    # Configuration
    d_model = 128
    n_seq = 64
    batch_size = 4
    
    # Create NonHermitian potential
    potential = NonHermitianPotential(
        d_model=d_model,
        n_seq=n_seq,
        base_decay=0.01,
        adaptive_decay=True,
    )
    
    print(f"Created NonHermitian potential:")
    print(f"  d_model: {d_model}")
    print(f"  n_seq: {n_seq}")
    print(f"  base_decay: {potential.base_decay}")
    print(f"  adaptive_decay: {potential.adaptive_decay}")
    
    # Generate input
    x = torch.randn(batch_size, n_seq, d_model, requires_grad=True)
    
    # Forward pass
    V_complex = potential(x)
    
    print(f"\nComplex potential shape: {V_complex.shape}")
    print(f"  Real part (V) range: [{V_complex.real.min():.4f}, {V_complex.real.max():.4f}]")
    print(f"  Imag part (-Γ) range: [{V_complex.imag.min():.4f}, {V_complex.imag.max():.4f}]")
    
    # Extract decay rate
    gamma = -V_complex.imag
    print(f"  Decay rate (Γ) range: [{gamma.min():.4f}, {gamma.max():.4f}]")
    print(f"  All Γ > 0: {(gamma > 0).all().item()}")
    
    # Backward pass
    loss = V_complex.real.sum() + gamma.sum()
    loss.backward()
    
    print(f"\nGradients computed:")
    print(f"  Input gradient norm: {torch.norm(x.grad):.4f}")
    print(f"  v_proj gradient norm: {torch.norm(potential.v_proj.weight.grad):.4f}")
    print(f"  gamma_proj gradient norm: {torch.norm(potential.gamma_proj.weight.grad):.4f}")
    
    # Get stability statistics
    stats = potential.get_statistics()
    print(f"\nStability statistics:")
    print(f"  Mean Γ: {stats['mean_gamma']:.4f}")
    print(f"  Std Γ: {stats['std_gamma']:.4f}")
    print(f"  Mean Γ/|V| ratio: {stats['mean_energy_ratio']:.4f}")
    print(f"  Max Γ/|V| ratio: {stats['max_energy_ratio']:.4f}")


def demo_dissipative_bk_layer():
    """Demo 2: DissipativeBKLayer with gradient flow."""
    print("\n" + "=" * 60)
    print("Demo 2: DissipativeBKLayer")
    print("=" * 60)
    
    # Configuration
    d_model = 128
    n_seq = 64
    batch_size = 4
    
    # Create layer
    layer = DissipativeBKLayer(
        d_model=d_model,
        n_seq=n_seq,
        use_triton=False,  # Use PyTorch for demo
        base_decay=0.01,
        adaptive_decay=True,
    )
    
    print(f"Created DissipativeBKLayer:")
    print(f"  d_model: {d_model}")
    print(f"  n_seq: {n_seq}")
    print(f"  use_triton: False")
    
    # Generate input
    x = torch.randn(batch_size, n_seq, d_model)
    
    # Forward pass
    features, V_complex = layer(x, return_potential=True)
    
    print(f"\nOutput features shape: {features.shape}")
    print(f"  Real part range: [{features[..., 0].min():.4f}, {features[..., 0].max():.4f}]")
    print(f"  Imag part range: [{features[..., 1].min():.4f}, {features[..., 1].max():.4f}]")
    
    # Backward pass
    loss = features.sum()
    loss.backward()
    
    print(f"\nGradients computed:")
    print(f"  v_proj gradient norm: {torch.norm(layer.potential.v_proj.weight.grad):.4f}")
    print(f"  gamma_proj gradient norm: {torch.norm(layer.potential.gamma_proj.weight.grad):.4f}")


def demo_gradient_safety():
    """Demo 3: Gradient safety mechanisms."""
    print("\n" + "=" * 60)
    print("Demo 3: Gradient Safety Mechanisms")
    print("=" * 60)
    
    # Create safety module
    safety = GradientSafetyModule(
        max_grad_norm=100.0,
        replace_nan_with_zero=True,
        monitor_stats=True,
    )
    
    print(f"Created GradientSafetyModule:")
    print(f"  max_grad_norm: {safety.max_grad_norm}")
    print(f"  replace_nan_with_zero: {safety.replace_nan_with_zero}")
    print(f"  monitor_stats: {safety.monitor_stats}")
    
    # Test 1: Gradient clipping
    print("\nTest 1: Gradient clipping")
    large_grad = torch.randn(100) * 1000.0
    norm_before = torch.norm(large_grad).item()
    
    safe_grad = safety.apply_safety(large_grad, param_name="test_param")
    norm_after = torch.norm(safe_grad).item()
    
    print(f"  Gradient norm before: {norm_before:.2f}")
    print(f"  Gradient norm after: {norm_after:.2f}")
    print(f"  Clipped: {norm_after < norm_before}")
    
    # Test 2: NaN/Inf replacement
    print("\nTest 2: NaN/Inf replacement")
    bad_grad = torch.randn(100)
    bad_grad[10:20] = float('nan')
    bad_grad[30:40] = float('inf')
    
    nan_count = torch.isnan(bad_grad).sum().item()
    inf_count = torch.isinf(bad_grad).sum().item()
    
    safe_grad = safety.apply_safety(bad_grad, param_name="test_param")
    
    print(f"  NaN values before: {nan_count}")
    print(f"  Inf values before: {inf_count}")
    print(f"  All finite after: {torch.isfinite(safe_grad).all().item()}")
    
    # Get statistics
    stats = safety.get_statistics()
    print(f"\nGradient statistics:")
    print(f"  Mean gradient norm: {stats['mean_grad_norm']:.4f}")
    print(f"  Max gradient norm: {stats['max_grad_norm']:.4f}")
    print(f"  Clip rate: {stats['clip_rate']:.2%}")
    print(f"  NaN rate: {stats['nan_rate']:.2%}")


def demo_training_with_safety():
    """Demo 4: Training loop with gradient safety."""
    print("\n" + "=" * 60)
    print("Demo 4: Training Loop with Gradient Safety")
    print("=" * 60)
    
    # Configuration
    d_model = 64
    n_seq = 32
    batch_size = 4
    num_steps = 20
    
    # Create model
    model = DissipativeBKLayer(
        d_model=d_model,
        n_seq=n_seq,
        use_triton=False,
        base_decay=0.01,
        adaptive_decay=True,
    )
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Create safety module
    safety = GradientSafetyModule(
        max_grad_norm=100.0,
        replace_nan_with_zero=True,
        monitor_stats=True,
    )
    
    print(f"Training configuration:")
    print(f"  Model: DissipativeBKLayer")
    print(f"  Optimizer: Adam (lr=0.001)")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {n_seq}")
    print(f"  Training steps: {num_steps}")
    
    # Training loop
    losses = []
    for step in range(num_steps):
        # Generate random data
        x = torch.randn(batch_size, n_seq, d_model)
        target = torch.randn(batch_size, n_seq, 2)
        
        # Forward pass
        features, _ = model(x, return_potential=False)
        loss = ((features - target) ** 2).mean()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Apply gradient safety
        safe_complex_backward(model, max_grad_norm=100.0, replace_nan=True)
        
        # Optimizer step
        optimizer.step()
        
        losses.append(loss.item())
        
        if (step + 1) % 5 == 0:
            print(f"  Step {step + 1:2d}: Loss = {loss.item():.6f}")
    
    # Get gradient statistics
    stats = model.potential.get_statistics()
    print(f"\nFinal statistics:")
    print(f"  Initial loss: {losses[0]:.6f}")
    print(f"  Final loss: {losses[-1]:.6f}")
    print(f"  Loss change: {losses[-1] - losses[0]:.6f}")
    print(f"  Mean Γ: {stats['mean_gamma']:.4f}")
    print(f"  Mean Γ/|V| ratio: {stats['mean_energy_ratio']:.4f}")


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("Complex Gradient Safety Demo - Phase 2")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    # Run demos
    demo_nonhermitian_potential()
    demo_dissipative_bk_layer()
    demo_gradient_safety()
    demo_training_with_safety()
    
    print("\n" + "=" * 60)
    print("All demos completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
