"""
Non-Hermitian Forgetting Mechanism Demo

This demo showcases the Non-Hermitian potential layer that enables
natural forgetting through energy dissipation, inspired by open quantum systems.

Physical Interpretation:
- V (real part): Semantic potential energy
- Γ (imaginary part): Dissipation rate (information decay)
- Time evolution: ||ψ(t)||² = exp(-2Γt) ||ψ(0)||²

Key Features:
1. Adaptive decay: Γ depends on input features
2. Stability monitoring: Detects overdamping (Γ >> |V|)
3. BK-Core integration: O(N) complexity maintained
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from src.models.phase2.non_hermitian import NonHermitianPotential, DissipativeBKLayer


def demo_basic_potential():
    """Demonstrate basic NonHermitianPotential functionality."""
    print("=" * 60)
    print("Demo 1: Basic Non-Hermitian Potential")
    print("=" * 60)
    
    d_model = 128
    n_seq = 256
    batch_size = 4
    
    # Create potential module
    potential = NonHermitianPotential(
        d_model=d_model,
        n_seq=n_seq,
        base_decay=0.01,
        adaptive_decay=True
    )
    
    # Create sample input
    x = torch.randn(batch_size, n_seq, d_model)
    
    # Generate complex potential
    V_complex = potential(x)
    
    # Extract components
    V_real = V_complex.real  # Semantic potential
    gamma = -V_complex.imag  # Decay rate (positive)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Complex potential shape: {V_complex.shape}")
    print(f"\nReal part (V) statistics:")
    print(f"  Mean: {V_real.mean():.4f}")
    print(f"  Std:  {V_real.std():.4f}")
    print(f"  Range: [{V_real.min():.4f}, {V_real.max():.4f}]")
    print(f"\nDecay rate (Γ) statistics:")
    print(f"  Mean: {gamma.mean():.4f}")
    print(f"  Std:  {gamma.std():.4f}")
    print(f"  Range: [{gamma.min():.4f}, {gamma.max():.4f}]")
    print(f"  Min >= base_decay (0.01): {gamma.min() >= 0.01}")
    
    # Visualize potential distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot V distribution
    axes[0].hist(V_real.flatten().detach().numpy(), bins=50, alpha=0.7, color='blue')
    axes[0].set_xlabel('Real Potential (V)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Semantic Potential')
    axes[0].grid(True, alpha=0.3)
    
    # Plot Γ distribution
    axes[1].hist(gamma.flatten().detach().numpy(), bins=50, alpha=0.7, color='red')
    axes[1].axvline(0.01, color='black', linestyle='--', label='base_decay')
    axes[1].set_xlabel('Decay Rate (Γ)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of Dissipation Rate')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/visualizations/non_hermitian_potential_distribution.png', dpi=150)
    print(f"\n✓ Saved visualization to results/visualizations/non_hermitian_potential_distribution.png")
    plt.close()


def demo_time_evolution():
    """Demonstrate time evolution with dissipation."""
    print("\n" + "=" * 60)
    print("Demo 2: Time Evolution with Dissipation")
    print("=" * 60)
    
    # Simulate information decay over time
    gamma_values = [0.01, 0.05, 0.1, 0.5]  # Different decay rates
    time_steps = np.linspace(0, 10, 100)
    
    plt.figure(figsize=(10, 6))
    
    for gamma in gamma_values:
        # Time evolution: ||ψ(t)||² = exp(-2Γt)
        amplitude_squared = np.exp(-2 * gamma * time_steps)
        plt.plot(time_steps, amplitude_squared, label=f'Γ = {gamma}', linewidth=2)
    
    plt.xlabel('Time (arbitrary units)', fontsize=12)
    plt.ylabel('Information Amplitude ||ψ(t)||²', fontsize=12)
    plt.title('Information Decay through Non-Hermitian Dissipation', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1.1])
    
    plt.tight_layout()
    plt.savefig('results/visualizations/non_hermitian_time_evolution.png', dpi=150)
    print(f"\n✓ Saved visualization to results/visualizations/non_hermitian_time_evolution.png")
    plt.close()
    
    print("\nKey observations:")
    print("  - Higher Γ → Faster information decay")
    print("  - Γ = 0.01: Slow forgetting (long-term memory)")
    print("  - Γ = 0.5:  Fast forgetting (short-term memory)")


def demo_dissipative_bk_layer():
    """Demonstrate DissipativeBKLayer integration."""
    print("\n" + "=" * 60)
    print("Demo 3: Dissipative BK-Core Integration")
    print("=" * 60)
    
    d_model = 128
    n_seq = 256
    batch_size = 2
    
    # Create dissipative BK layer
    layer = DissipativeBKLayer(
        d_model=d_model,
        n_seq=n_seq,
        use_triton=False,  # Use PyTorch for demo
        base_decay=0.01,
        adaptive_decay=True
    )
    
    # Create input
    x = torch.randn(batch_size, n_seq, d_model)
    
    # Forward pass
    features, V_complex = layer(x, return_potential=True)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Output features shape: {features.shape}")
    print(f"Complex potential shape: {V_complex.shape}")
    
    # Extract BK-Core features
    G_real = features[:, :, 0]  # Real part of Green's function diagonal
    G_imag = features[:, :, 1]  # Imaginary part
    
    print(f"\nBK-Core features (Green's function diagonal):")
    print(f"  Real part range: [{G_real.min():.4f}, {G_real.max():.4f}]")
    print(f"  Imag part range: [{G_imag.min():.4f}, {G_imag.max():.4f}]")
    
    # Get decay rate
    gamma = layer.get_gamma(x)
    print(f"\nDecay rate (Γ):")
    print(f"  Mean: {gamma.mean():.4f}")
    print(f"  Std:  {gamma.std():.4f}")
    
    # Test gradient flow
    x_grad = torch.randn(batch_size, n_seq, d_model, requires_grad=True)
    features_grad, _ = layer(x_grad, return_potential=False)
    loss = features_grad.sum()
    loss.backward()
    
    print(f"\nGradient flow:")
    print(f"  Input gradient exists: {x_grad.grad is not None}")
    print(f"  Gradient contains NaN: {torch.isnan(x_grad.grad).any().item()}")
    print(f"  Gradient contains Inf: {torch.isinf(x_grad.grad).any().item()}")
    print(f"  Gradient norm: {x_grad.grad.norm():.4f}")


def demo_stability_monitoring():
    """Demonstrate stability monitoring."""
    print("\n" + "=" * 60)
    print("Demo 4: Stability Monitoring")
    print("=" * 60)
    
    d_model = 128
    n_seq = 256
    batch_size = 4
    
    # Create potential with monitoring
    potential = NonHermitianPotential(
        d_model=d_model,
        n_seq=n_seq,
        base_decay=0.01,
        adaptive_decay=True
    )
    
    # Set to training mode (enables monitoring)
    potential.train()
    
    # Run multiple forward passes to accumulate statistics
    print("\nRunning 50 forward passes to accumulate statistics...")
    for i in range(50):
        x = torch.randn(batch_size, n_seq, d_model)
        V_complex = potential(x)
        
        if (i + 1) % 10 == 0:
            stats = potential.get_statistics()
            print(f"  Step {i+1:2d}: mean_gamma={stats['mean_gamma']:.4f}, "
                  f"energy_ratio={stats['mean_energy_ratio']:.4f}")
    
    # Get final statistics
    stats = potential.get_statistics()
    print(f"\nFinal statistics:")
    print(f"  Mean Γ: {stats['mean_gamma']:.4f} ± {stats['std_gamma']:.4f}")
    print(f"  Mean energy ratio (Γ/|V|): {stats['mean_energy_ratio']:.4f}")
    print(f"  Max energy ratio: {stats['max_energy_ratio']:.4f}")
    
    if stats['max_energy_ratio'] > 10.0:
        print(f"\n⚠️  Warning: Overdamping detected (ratio > 10.0)")
    else:
        print(f"\n✓ System is stable (ratio < 10.0)")


def demo_adaptive_vs_fixed():
    """Compare adaptive vs fixed decay rates."""
    print("\n" + "=" * 60)
    print("Demo 5: Adaptive vs Fixed Decay")
    print("=" * 60)
    
    d_model = 128
    n_seq = 256
    batch_size = 4
    
    # Create two potentials
    potential_adaptive = NonHermitianPotential(
        d_model=d_model,
        n_seq=n_seq,
        base_decay=0.01,
        adaptive_decay=True
    )
    
    potential_fixed = NonHermitianPotential(
        d_model=d_model,
        n_seq=n_seq,
        base_decay=0.05,
        adaptive_decay=False
    )
    
    # Create input
    x = torch.randn(batch_size, n_seq, d_model)
    
    # Forward passes
    V_adaptive = potential_adaptive(x)
    V_fixed = potential_fixed(x)
    
    gamma_adaptive = -V_adaptive.imag
    gamma_fixed = -V_fixed.imag
    
    print(f"\nAdaptive decay:")
    print(f"  Mean: {gamma_adaptive.mean():.4f}")
    print(f"  Std:  {gamma_adaptive.std():.4f}")
    print(f"  Range: [{gamma_adaptive.min():.4f}, {gamma_adaptive.max():.4f}]")
    
    print(f"\nFixed decay:")
    print(f"  Mean: {gamma_fixed.mean():.4f}")
    print(f"  Std:  {gamma_fixed.std():.4f}")
    print(f"  Range: [{gamma_fixed.min():.4f}, {gamma_fixed.max():.4f}]")
    
    # Visualize comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].hist(gamma_adaptive.flatten().detach().numpy(), bins=50, alpha=0.7, color='blue')
    axes[0].set_xlabel('Decay Rate (Γ)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Adaptive Decay (Input-Dependent)')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(gamma_fixed.flatten().detach().numpy(), bins=50, alpha=0.7, color='red')
    axes[1].set_xlabel('Decay Rate (Γ)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Fixed Decay (Constant)')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/visualizations/non_hermitian_adaptive_vs_fixed.png', dpi=150)
    print(f"\n✓ Saved visualization to results/visualizations/non_hermitian_adaptive_vs_fixed.png")
    plt.close()


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("Non-Hermitian Forgetting Mechanism Demo")
    print("Phase 2: Breath of Life")
    print("=" * 60)
    
    # Create output directory
    import os
    os.makedirs('results/visualizations', exist_ok=True)
    
    # Run demos
    demo_basic_potential()
    demo_time_evolution()
    demo_dissipative_bk_layer()
    demo_stability_monitoring()
    demo_adaptive_vs_fixed()
    
    print("\n" + "=" * 60)
    print("✅ All demos completed successfully!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("  1. Non-Hermitian potential enables natural forgetting")
    print("  2. Γ (decay rate) is always positive and >= base_decay")
    print("  3. Adaptive decay allows input-dependent forgetting")
    print("  4. Stability monitoring prevents overdamping")
    print("  5. BK-Core integration maintains O(N) complexity")
    print("  6. Gradient flow is stable and well-behaved")


if __name__ == "__main__":
    main()
