"""
Sparse BK-Core Demo

Demonstrates learned sparsity in BK-Core computation with importance prediction
and interpolation for masked positions.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from src.models.sparse_bk_core import SparseBKCore, SparseMoEResNetBKLayer


def demo_sparse_bk_core():
    """
    Demonstrate SparseBKCore with visualization of learned sparsity patterns.
    """
    print("=" * 80)
    print("Sparse BK-Core Demo")
    print("=" * 80)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Model parameters
    d_model = 64
    n_seq = 128
    target_sparsity = 0.5
    batch_size = 4
    
    print(f"\nModel Configuration:")
    print(f"  d_model: {d_model}")
    print(f"  n_seq: {n_seq}")
    print(f"  target_sparsity: {target_sparsity}")
    print(f"  batch_size: {batch_size}")
    
    # Create sparse BK-Core
    sparse_bk = SparseBKCore(d_model, n_seq, target_sparsity).to(device)
    
    print(f"\nModel Parameters: {sum(p.numel() for p in sparse_bk.parameters()):,}")
    
    # Generate sample data
    x = torch.randn(batch_size, n_seq, d_model, device=device)
    v = torch.randn(batch_size, n_seq, device=device)
    
    print("\n" + "-" * 80)
    print("Forward Pass")
    print("-" * 80)
    
    # Forward pass
    with torch.no_grad():
        features, mask, sparsity_ratio = sparse_bk(x, v)
    
    print(f"\nOutput shapes:")
    print(f"  features: {features.shape}")
    print(f"  mask: {mask.shape}")
    
    print(f"\nSparsity Statistics:")
    print(f"  Target sparsity: {target_sparsity:.2%}")
    print(f"  Actual sparsity: {sparsity_ratio.item():.2%}")
    print(f"  Positions computed: {mask.sum(dim=-1).mean().item():.1f} / {n_seq}")
    print(f"  Positions masked: {(n_seq - mask.sum(dim=-1).mean().item()):.1f} / {n_seq}")
    
    # Visualize mask pattern
    print("\n" + "-" * 80)
    print("Visualizing Sparsity Pattern")
    print("-" * 80)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Learned Sparsity Patterns in BK-Core', fontsize=14, fontweight='bold')
    
    for idx in range(min(4, batch_size)):
        ax = axes[idx // 2, idx % 2]
        
        # Plot mask
        mask_np = mask[idx].cpu().numpy()
        ax.imshow(mask_np.reshape(1, -1), cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        ax.set_title(f'Batch {idx}: {mask_np.sum():.0f}/{n_seq} computed')
        ax.set_xlabel('Sequence Position')
        ax.set_yticks([])
        
        # Add colorbar
        cbar = plt.colorbar(ax.images[0], ax=ax, orientation='horizontal', pad=0.1)
        cbar.set_label('Computed (1) / Masked (0)')
    
    plt.tight_layout()
    plt.savefig('sparse_bk_core_patterns.png', dpi=150, bbox_inches='tight')
    print("\nSaved sparsity pattern visualization to: sparse_bk_core_patterns.png")
    
    # Training demo
    print("\n" + "-" * 80)
    print("Training Demo: Learning Sparsity")
    print("-" * 80)
    
    # Create optimizer
    optimizer = optim.Adam(sparse_bk.parameters(), lr=1e-3)
    
    # Training loop
    num_steps = 50
    losses = []
    sparsities = []
    
    print(f"\nTraining for {num_steps} steps...")
    
    for step in range(num_steps):
        # Generate random data
        x = torch.randn(batch_size, n_seq, d_model, device=device)
        v = torch.randn(batch_size, n_seq, device=device)
        
        # Forward pass
        features, mask, sparsity_ratio = sparse_bk(x, v)
        
        # Compute losses
        # Task loss: minimize feature magnitude (dummy task)
        task_loss = features.pow(2).mean()
        
        # Sparsity loss: encourage target sparsity
        sparsity_loss = sparse_bk.sparsity_loss(mask)
        
        # Total loss
        total_loss = task_loss + 0.1 * sparsity_loss
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Record metrics
        losses.append(total_loss.item())
        sparsities.append(sparsity_ratio.item())
        
        if (step + 1) % 10 == 0:
            print(f"  Step {step+1:3d}: Loss={total_loss.item():.4f}, "
                  f"Sparsity={sparsity_ratio.item():.2%}, "
                  f"Target={target_sparsity:.2%}")
    
    # Plot training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('Training Dynamics', fontsize=14, fontweight='bold')
    
    # Loss curve
    ax1.plot(losses, linewidth=2)
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Total Loss')
    ax1.set_title('Loss Curve')
    ax1.grid(True, alpha=0.3)
    
    # Sparsity curve
    ax2.plot(sparsities, linewidth=2, label='Actual Sparsity')
    ax2.axhline(y=target_sparsity, color='r', linestyle='--', linewidth=2, label='Target Sparsity')
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Sparsity Ratio')
    ax2.set_title('Sparsity Convergence')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig('sparse_bk_core_training.png', dpi=150, bbox_inches='tight')
    print("\nSaved training curves to: sparse_bk_core_training.png")
    
    print("\n" + "=" * 80)
    print("Demo Complete!")
    print("=" * 80)


def demo_sparse_moe_layer():
    """
    Demonstrate SparseMoEResNetBKLayer in a language modeling context.
    """
    print("\n" + "=" * 80)
    print("Sparse MoE-ResNet-BK Layer Demo")
    print("=" * 80)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model parameters
    d_model = 64
    n_seq = 128
    batch_size = 2
    
    print(f"\nConfiguration:")
    print(f"  d_model: {d_model}")
    print(f"  n_seq: {n_seq}")
    print(f"  num_experts: 4")
    print(f"  target_sparsity: 0.5")
    
    # Create sparse layer
    sparse_layer = SparseMoEResNetBKLayer(
        d_model=d_model,
        n_seq=n_seq,
        num_experts=4,
        target_sparsity=0.5,
        sparsity_loss_weight=0.01
    ).to(device)
    
    print(f"\nLayer Parameters: {sum(p.numel() for p in sparse_layer.parameters()):,}")
    
    # Generate sample input
    x = torch.randn(batch_size, n_seq, d_model, device=device)
    
    print("\n" + "-" * 80)
    print("Forward Pass")
    print("-" * 80)
    
    # Forward pass
    with torch.no_grad():
        output = sparse_layer(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Get sparsity statistics
    stats = sparse_layer.get_sparsity_stats()
    
    print(f"\nSparsity Statistics:")
    print(f"  Target sparsity: {stats['target_sparsity']:.2%}")
    print(f"  Actual sparsity: {stats['sparsity_ratio']:.2%}")
    print(f"  Positions computed: {stats['num_computed']:.1f} / {n_seq}")
    
    # Get sparsity loss
    sparsity_loss = sparse_layer.get_sparsity_loss()
    print(f"  Sparsity loss: {sparsity_loss.item():.6f}")
    
    # Backward pass test
    print("\n" + "-" * 80)
    print("Backward Pass Test")
    print("-" * 80)
    
    x_grad = torch.randn(batch_size, n_seq, d_model, device=device, requires_grad=True)
    output = sparse_layer(x_grad)
    sparsity_loss = sparse_layer.get_sparsity_loss()
    
    # Compute loss
    loss = output.sum() + sparsity_loss
    loss.backward()
    
    print(f"\nGradient statistics:")
    print(f"  Gradient norm: {x_grad.grad.norm().item():.4f}")
    print(f"  Gradient mean: {x_grad.grad.mean().item():.6f}")
    print(f"  Gradient std: {x_grad.grad.std().item():.6f}")
    print(f"  Contains NaN: {torch.isnan(x_grad.grad).any().item()}")
    print(f"  Contains Inf: {torch.isinf(x_grad.grad).any().item()}")
    
    print("\n" + "=" * 80)
    print("Demo Complete!")
    print("=" * 80)


def compare_sparse_vs_dense():
    """
    Compare sparse BK-Core vs dense BK-Core in terms of computation.
    """
    print("\n" + "=" * 80)
    print("Sparse vs Dense BK-Core Comparison")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    d_model = 64
    n_seq = 128
    batch_size = 4
    
    # Create models
    sparse_bk = SparseBKCore(d_model, n_seq, target_sparsity=0.5).to(device)
    
    # Generate data
    x = torch.randn(batch_size, n_seq, d_model, device=device)
    v = torch.randn(batch_size, n_seq, device=device)
    
    # Sparse forward
    with torch.no_grad():
        features_sparse, mask, sparsity_ratio = sparse_bk(x, v)
    
    print(f"\nSparsity Analysis:")
    print(f"  Target sparsity: 50%")
    print(f"  Actual sparsity: {sparsity_ratio.item():.2%}")
    print(f"  Positions computed: {mask.sum().item():.0f} / {batch_size * n_seq}")
    print(f"  Theoretical speedup: {1 / (1 - sparsity_ratio.item()):.2f}×")
    
    # Note: Actual speedup requires optimized sparse computation
    print(f"\nNote: Current implementation computes full BK-Core then masks.")
    print(f"      Optimized implementation would skip masked positions for true speedup.")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    # Run demos
    demo_sparse_bk_core()
    demo_sparse_moe_layer()
    compare_sparse_vs_dense()
    
    print("\n✓ All demos completed successfully!")
