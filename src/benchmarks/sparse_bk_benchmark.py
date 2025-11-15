"""
Benchmark for Sparse BK-Core Computation

Measures the speedup achieved by skipping theta/phi recursions for masked positions.
"""

import torch
import time
from ..models.sparse_bk_core import SparseBKCore, optimized_sparse_bk_core
from ..models.bk_core import BKCoreFunction


def benchmark_sparse_vs_full(
    d_model=64,
    n_seq=128,
    batch_size=32,
    target_sparsity=0.5,
    num_iterations=100,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Benchmark sparse BK-Core vs full BK-Core computation.
    
    Args:
        d_model: hidden dimension
        n_seq: sequence length
        batch_size: batch size
        target_sparsity: target sparsity ratio
        num_iterations: number of iterations for timing
        device: device to run on
    
    Returns:
        dict with timing results and speedup
    """
    print(f"\n{'='*60}")
    print(f"Sparse BK-Core Benchmark")
    print(f"{'='*60}")
    print(f"Configuration:")
    print(f"  d_model: {d_model}")
    print(f"  n_seq: {n_seq}")
    print(f"  batch_size: {batch_size}")
    print(f"  target_sparsity: {target_sparsity}")
    print(f"  device: {device}")
    print(f"  iterations: {num_iterations}")
    print(f"{'='*60}\n")
    
    # Create sparse BK-Core
    sparse_bk = SparseBKCore(d_model, n_seq, target_sparsity).to(device)
    sparse_bk.eval()
    
    # Create dummy input
    x = torch.randn(batch_size, n_seq, d_model, device=device)
    v = torch.randn(batch_size, n_seq, device=device) * 0.5
    
    # Create a fixed mask for fair comparison
    with torch.no_grad():
        importance_scores = sparse_bk.importance_predictor(x).squeeze(-1)
        mask = (torch.sigmoid(importance_scores) > 0.5).float()
        actual_sparsity = 1.0 - mask.mean().item()
    
    print(f"Actual sparsity: {actual_sparsity:.2%}")
    print(f"Positions computed: {mask.sum().item():.0f} / {batch_size * n_seq}")
    print()
    
    # Prepare inputs for direct comparison
    h0_diag = sparse_bk.h0_diag_base.expand(batch_size, -1)
    h0_sub = sparse_bk.h0_sub_base.expand(batch_size, -1)
    h0_super = sparse_bk.h0_super_base.expand(batch_size, -1)
    he_diag = h0_diag + v
    z = sparse_bk.z
    
    # Warm-up
    print("Warming up...")
    for _ in range(10):
        _ = optimized_sparse_bk_core(he_diag, h0_super, h0_sub, z, mask)
        _ = BKCoreFunction.apply(he_diag, h0_super, h0_sub, z)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark sparse computation
    print("Benchmarking sparse computation...")
    start_time = time.time()
    for _ in range(num_iterations):
        features_sparse = optimized_sparse_bk_core(he_diag, h0_super, h0_sub, z, mask)
    if device == 'cuda':
        torch.cuda.synchronize()
    sparse_time = time.time() - start_time
    
    # Benchmark full computation
    print("Benchmarking full computation...")
    start_time = time.time()
    for _ in range(num_iterations):
        features_full = BKCoreFunction.apply(he_diag, h0_super, h0_sub, z)
    if device == 'cuda':
        torch.cuda.synchronize()
    full_time = time.time() - start_time
    
    # Calculate speedup
    speedup = full_time / sparse_time
    
    # Verify correctness (features should be similar for computed positions)
    with torch.no_grad():
        features_sparse_check = optimized_sparse_bk_core(he_diag, h0_super, h0_sub, z, mask)
        features_full_check = BKCoreFunction.apply(he_diag, h0_super, h0_sub, z)
        
        # Compare only at computed positions
        mask_expanded = mask.unsqueeze(-1).expand_as(features_sparse_check)
        diff = torch.abs(features_sparse_check - features_full_check)
        masked_diff = diff * mask_expanded
        max_diff = masked_diff.max().item()
        mean_diff = masked_diff.sum().item() / mask.sum().item() / 2  # Divide by 2 for real/imag
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"{'='*60}")
    print(f"Sparse computation time: {sparse_time:.4f}s ({sparse_time/num_iterations*1000:.2f}ms per iteration)")
    print(f"Full computation time:   {full_time:.4f}s ({full_time/num_iterations*1000:.2f}ms per iteration)")
    print(f"Speedup:                 {speedup:.2f}x")
    print(f"\nAccuracy (at computed positions):")
    print(f"Max difference:          {max_diff:.6f}")
    print(f"Mean difference:         {mean_diff:.6f}")
    print(f"{'='*60}\n")
    
    # Theoretical speedup estimate
    theoretical_speedup = 1.0 / (1.0 - actual_sparsity * 0.5)  # Rough estimate
    print(f"Theoretical speedup (rough): {theoretical_speedup:.2f}x")
    print(f"Efficiency: {speedup/theoretical_speedup*100:.1f}%")
    print()
    
    return {
        'sparse_time': sparse_time,
        'full_time': full_time,
        'speedup': speedup,
        'actual_sparsity': actual_sparsity,
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'theoretical_speedup': theoretical_speedup,
        'efficiency': speedup / theoretical_speedup
    }


def benchmark_sparsity_levels(
    d_model=64,
    n_seq=128,
    batch_size=32,
    sparsity_levels=[0.0, 0.25, 0.5, 0.75],
    num_iterations=50,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Benchmark sparse BK-Core at different sparsity levels.
    
    Args:
        d_model: hidden dimension
        n_seq: sequence length
        batch_size: batch size
        sparsity_levels: list of sparsity ratios to test
        num_iterations: number of iterations for timing
        device: device to run on
    
    Returns:
        list of dicts with results for each sparsity level
    """
    print(f"\n{'='*60}")
    print(f"Sparsity Level Sweep")
    print(f"{'='*60}\n")
    
    results = []
    
    for sparsity in sparsity_levels:
        print(f"\nTesting sparsity: {sparsity:.0%}")
        print("-" * 60)
        
        result = benchmark_sparse_vs_full(
            d_model=d_model,
            n_seq=n_seq,
            batch_size=batch_size,
            target_sparsity=sparsity,
            num_iterations=num_iterations,
            device=device
        )
        
        results.append({
            'target_sparsity': sparsity,
            **result
        })
    
    # Summary table
    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    print(f"{'Sparsity':<12} {'Speedup':<10} {'Efficiency':<12} {'Max Diff':<12}")
    print("-" * 60)
    for r in results:
        print(f"{r['target_sparsity']:<12.0%} {r['speedup']:<10.2f} {r['efficiency']*100:<11.1f}% {r['max_diff']:<12.6f}")
    print(f"{'='*60}\n")
    
    return results


if __name__ == '__main__':
    # Run benchmarks
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Single benchmark at 50% sparsity
    print("\n" + "="*60)
    print("BENCHMARK 1: Single Run at 50% Sparsity")
    print("="*60)
    benchmark_sparse_vs_full(
        d_model=64,
        n_seq=128,
        batch_size=32,
        target_sparsity=0.5,
        num_iterations=100,
        device=device
    )
    
    # Sparsity sweep
    print("\n" + "="*60)
    print("BENCHMARK 2: Sparsity Level Sweep")
    print("="*60)
    benchmark_sparsity_levels(
        d_model=64,
        n_seq=128,
        batch_size=32,
        sparsity_levels=[0.0, 0.25, 0.5, 0.75],
        num_iterations=50,
        device=device
    )
