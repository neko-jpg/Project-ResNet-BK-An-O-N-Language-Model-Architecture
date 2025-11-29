#!/usr/bin/env python3
"""
Property Test Runner for Adaptive Computation Savings

**Feature: phase8-hyperbolic-transcendence, Property 22: Adaptive Computation Savings**
**Validates: Requirements 80.4**

This script runs the property-based test and outputs results to JSON.
"""
import torch
import json
import random
from datetime import datetime
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.phase8.adaptive import AdaptiveComputation


def generate_mixed_distribution(batch_size: int, seq_len: int, d_model: int, origin_ratio: float) -> torch.Tensor:
    """Generate a mixed distribution of tokens."""
    x = torch.randn(batch_size, seq_len, d_model)
    num_origin_tokens = int(seq_len * origin_ratio)
    
    for b in range(batch_size):
        # Tokens near origin (small norm)
        for i in range(num_origin_tokens):
            x[b, i] = x[b, i] * 0.05
        # Tokens near boundary (large norm)
        for i in range(num_origin_tokens, seq_len):
            x[b, i] = x[b, i] / x[b, i].norm() * 0.9
    
    # Shuffle
    perm = torch.randperm(seq_len)
    x = x[:, perm, :]
    return x


def compute_savings(adaptive: AdaptiveComputation, x: torch.Tensor, total_layers: int) -> float:
    """Compute the compute savings from adaptive computation."""
    batch_size, seq_len, _ = x.shape
    total_tokens = batch_size * seq_len
    exited_tokens = 0
    remaining_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    
    for layer_idx in range(total_layers):
        if not remaining_mask.any():
            break
        should_exit, _ = adaptive(x, layer_idx, total_layers)
        newly_exited = (should_exit & remaining_mask).sum().item()
        layers_saved = total_layers - layer_idx - 1
        exited_tokens += newly_exited * layers_saved
        remaining_mask = remaining_mask & ~should_exit
    
    max_compute = total_tokens * total_layers
    return exited_tokens / max_compute if max_compute > 0 else 0.0


def main():
    print("="*70)
    print("Property Test: Adaptive Computation Savings")
    print("**Feature: phase8-hyperbolic-transcendence, Property 22**")
    print("**Validates: Requirements 80.4**")
    print("="*70)
    
    # Configuration
    d_model = 64
    num_iterations = 100
    total_layers = 12
    exit_threshold = 0.5
    min_savings_threshold = 0.30
    
    adaptive = AdaptiveComputation(d_model, exit_threshold=exit_threshold)
    
    # Run property test
    print(f"\nRunning {num_iterations} iterations...")
    savings_list = []
    
    for i in range(num_iterations):
        batch_size = random.randint(1, 4)
        seq_len = random.randint(16, 128)
        origin_ratio = random.uniform(0.5, 0.9)
        x = generate_mixed_distribution(batch_size, seq_len, d_model, origin_ratio)
        savings = compute_savings(adaptive, x, total_layers)
        savings_list.append(savings)
        
        if (i + 1) % 20 == 0:
            print(f"  Completed {i+1}/{num_iterations} iterations")
    
    avg_savings = sum(savings_list) / len(savings_list)
    min_savings = min(savings_list)
    max_savings = max(savings_list)
    
    print(f"\nResults:")
    print(f"  Average savings: {avg_savings*100:.2f}%")
    print(f"  Min savings: {min_savings*100:.2f}%")
    print(f"  Max savings: {max_savings*100:.2f}%")
    
    # Test by origin ratio
    print("\nSavings by origin ratio:")
    origin_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
    savings_by_ratio = {}
    
    for ratio in origin_ratios:
        ratio_savings = []
        for _ in range(10):
            x = generate_mixed_distribution(2, 64, d_model, ratio)
            savings = compute_savings(adaptive, x, total_layers)
            ratio_savings.append(savings)
        avg = sum(ratio_savings) / len(ratio_savings)
        savings_by_ratio[str(ratio)] = avg
        print(f"  Origin ratio {ratio:.1f}: {avg*100:.2f}%")
    
    # Build results
    property_verified = avg_savings >= min_savings_threshold
    
    results = {
        'test_name': 'Property 22: Adaptive Computation Savings',
        'validates': 'Requirements 80.4',
        'timestamp': datetime.now().isoformat(),
        'configuration': {
            'd_model': d_model,
            'num_iterations': num_iterations,
            'total_layers': total_layers,
            'exit_threshold': exit_threshold,
            'min_savings_threshold': min_savings_threshold
        },
        'results': {
            'average_savings': avg_savings,
            'min_savings': min_savings,
            'max_savings': max_savings,
            'savings_by_origin_ratio': savings_by_ratio
        },
        'property_verified': property_verified,
        'conclusion': f'Average compute savings {avg_savings*100:.2f}% meets 30% threshold' if property_verified else f'FAILED: Average savings {avg_savings*100:.2f}% below 30%'
    }
    
    # Save to JSON
    output_path = 'results/benchmarks/phase8_adaptive_computation_property_test.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    if property_verified:
        print(f"\n✓ Property VERIFIED: Average savings {avg_savings*100:.2f}% >= {min_savings_threshold*100:.2f}%")
    else:
        print(f"\n✗ Property FAILED: Average savings {avg_savings*100:.2f}% < {min_savings_threshold*100:.2f}%")
    
    return 0 if property_verified else 1


if __name__ == '__main__':
    sys.exit(main())
