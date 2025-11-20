"""
Phase2Block Demo

This script demonstrates the usage of Phase2Block, which integrates:
1. Non-Hermitian Forgetting (散逸的忘却)
2. Dissipative Hebbian Dynamics (散逸的Hebbian動力学)
3. SNR-based Memory Selection (SNRベースの記憶選択)
4. Memory Resonance (記憶共鳴)

Requirements: 6.1, 6.2, 6.3
"""

import torch
import torch.nn as nn
from src.models.phase2 import Phase2Block


def main():
    print("=" * 80)
    print("Phase2Block Demo - Breath of Life")
    print("=" * 80)
    
    # Configuration
    config = {
        'd_model': 256,
        'n_seq': 128,
        'num_heads': 8,
        'head_dim': 32,
        'use_triton': False,  # Use PyTorch for demo
        'ffn_dim': 1024,
        'dropout': 0.1,
    }
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Create Phase2Block
    print("\n1. Creating Phase2Block...")
    block = Phase2Block(**config).to(device)
    print(f"   - Model dimension: {config['d_model']}")
    print(f"   - Sequence length: {config['n_seq']}")
    print(f"   - Number of heads: {config['num_heads']}")
    print(f"   - Head dimension: {config['head_dim']}")
    
    # Count parameters
    total_params = sum(p.numel() for p in block.parameters())
    trainable_params = sum(p.numel() for p in block.parameters() if p.requires_grad)
    print(f"   - Total parameters: {total_params:,}")
    print(f"   - Trainable parameters: {trainable_params:,}")
    
    # Create sample input
    print("\n2. Creating sample input...")
    batch_size = 4
    x = torch.randn(batch_size, config['n_seq'], config['d_model'], device=device)
    print(f"   - Input shape: {x.shape}")
    
    # Forward pass without diagnostics
    print("\n3. Forward pass (without diagnostics)...")
    with torch.no_grad():
        output = block(x, return_diagnostics=False)
    print(f"   - Output shape: {output.shape}")
    print(f"   - Output mean: {output.mean().item():.6f}")
    print(f"   - Output std: {output.std().item():.6f}")
    
    # Forward pass with diagnostics
    print("\n4. Forward pass (with diagnostics)...")
    with torch.no_grad():
        output, diagnostics = block(x, return_diagnostics=True)
    
    print(f"   - Output shape: {output.shape}")
    print("\n   Diagnostics:")
    print(f"   - Gamma (decay rate):")
    print(f"     * Mean: {diagnostics['gamma'].mean().item():.6f}")
    print(f"     * Std: {diagnostics['gamma'].std().item():.6f}")
    print(f"     * Min: {diagnostics['gamma'].min().item():.6f}")
    print(f"     * Max: {diagnostics['gamma'].max().item():.6f}")
    
    print(f"\n   - Fast Weight Energy: {diagnostics['fast_weight_energy']:.6f}")
    
    if diagnostics['snr_stats']:
        print(f"\n   - SNR Statistics:")
        for key, value in diagnostics['snr_stats'].items():
            print(f"     * {key}: {value:.6f}")
    
    if diagnostics['resonance_info']:
        print(f"\n   - Resonance Information:")
        for key, value in diagnostics['resonance_info'].items():
            if isinstance(value, (int, float)):
                print(f"     * {key}: {value:.6f}")
    
    if diagnostics['stability']:
        print(f"\n   - Stability Metrics:")
        for key, value in diagnostics['stability'].items():
            if isinstance(value, (int, float, bool)):
                print(f"     * {key}: {value}")
    
    # Test gradient flow
    print("\n5. Testing gradient flow...")
    x_grad = torch.randn(batch_size, config['n_seq'], config['d_model'], device=device, requires_grad=True)
    output = block(x_grad)
    loss = output.sum()
    loss.backward()
    
    print(f"   - Input gradient norm: {x_grad.grad.norm().item():.6f}")
    
    # Count parameters with gradients
    params_with_grad = sum(1 for p in block.parameters() if p.grad is not None)
    total_params_count = sum(1 for p in block.parameters())
    print(f"   - Parameters with gradients: {params_with_grad}/{total_params_count}")
    
    # Get statistics
    print("\n6. Block statistics...")
    stats = block.get_statistics()
    
    print("   - Hebbian Layer:")
    for key, value in stats['hebbian'].items():
        print(f"     * {key}: {value:.6f}")
    
    print("\n   - SNR Filter:")
    for key, value in stats['snr'].items():
        print(f"     * {key}: {value:.6f}")
    
    print("\n   - Non-Hermitian Potential:")
    for key, value in stats['non_hermitian'].items():
        print(f"     * {key}: {value:.6f}")
    
    # Test state management
    print("\n7. Testing state management...")
    print(f"   - Initial state: {block.fast_weight_state is not None}")
    
    # Reset state
    block.reset_state()
    print(f"   - After reset: {block.fast_weight_state is None}")
    
    # Forward pass to create state
    with torch.no_grad():
        _ = block(x)
    print(f"   - After forward: {block.fast_weight_state is not None}")
    if block.fast_weight_state is not None:
        print(f"   - State shape: {block.fast_weight_state.shape}")
    
    # Multiple forward passes
    print("\n8. Testing multiple forward passes...")
    block.reset_state()
    
    for i in range(3):
        with torch.no_grad():
            x_i = torch.randn(batch_size, config['n_seq'], config['d_model'], device=device)
            output_i = block(x_i)
            
            if block.fast_weight_state is not None:
                state_norm = torch.norm(block.fast_weight_state).item()
                print(f"   - Pass {i+1}: State norm = {state_norm:.6f}")
    
    print("\n" + "=" * 80)
    print("Demo completed successfully!")
    print("=" * 80)


if __name__ == '__main__':
    main()
