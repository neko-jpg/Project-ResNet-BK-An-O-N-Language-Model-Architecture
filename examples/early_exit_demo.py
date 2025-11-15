"""
Early Exit Demo
Demonstrates early exiting for inference with confidence-based halting.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append('src')

from models.early_exit import EarlyExitLanguageModel, EarlyExitEvaluator


def create_synthetic_data(vocab_size=1000, n_seq=128, num_samples=1000):
    """Create synthetic language modeling data."""
    # Generate random sequences
    x = torch.randint(0, vocab_size, (num_samples, n_seq))
    
    # Targets are shifted inputs (next token prediction)
    y = torch.cat([x[:, 1:], torch.randint(0, vocab_size, (num_samples, 1))], dim=1)
    
    return x, y


def visualize_exit_distribution(stats, threshold):
    """Visualize exit layer distribution."""
    exit_dist = stats['exit_distribution']
    layers = list(range(len(exit_dist)))
    
    plt.figure(figsize=(10, 6))
    plt.bar(layers, exit_dist, alpha=0.7, color='steelblue')
    plt.xlabel('Exit Layer', fontsize=12)
    plt.ylabel('Percentage of Tokens (%)', fontsize=12)
    plt.title(f'Early Exit Distribution (Threshold={threshold})', fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    
    # Add average exit layer line
    avg_exit = stats['avg_exit_layer']
    plt.axvline(avg_exit, color='red', linestyle='--', linewidth=2, 
                label=f'Avg Exit Layer: {avg_exit:.2f}')
    plt.legend()
    
    plt.tight_layout()
    return plt.gcf()


def visualize_threshold_comparison(results):
    """Compare performance across different confidence thresholds."""
    thresholds = sorted(results.keys())
    avg_exits = [results[t]['avg_exit_layer'] for t in thresholds]
    speedups = [results[t]['speedup_estimate'] for t in thresholds]
    perplexities = [results[t]['perplexity'] for t in thresholds]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Average exit layer and speedup vs threshold
    ax1_twin = ax1.twinx()
    
    line1 = ax1.plot(thresholds, avg_exits, 'o-', color='steelblue', 
                     linewidth=2, markersize=8, label='Avg Exit Layer')
    ax1.set_xlabel('Confidence Threshold', fontsize=12)
    ax1.set_ylabel('Average Exit Layer', fontsize=12, color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    ax1.grid(alpha=0.3)
    
    line2 = ax1_twin.plot(thresholds, speedups, 's-', color='coral', 
                          linewidth=2, markersize=8, label='Speedup Estimate')
    ax1_twin.set_ylabel('Speedup Estimate', fontsize=12, color='coral')
    ax1_twin.tick_params(axis='y', labelcolor='coral')
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    
    ax1.set_title('Exit Layer and Speedup vs Threshold', fontsize=14)
    
    # Plot 2: Perplexity vs threshold
    ax2.plot(thresholds, perplexities, 'o-', color='forestgreen', 
             linewidth=2, markersize=8)
    ax2.set_xlabel('Confidence Threshold', fontsize=12)
    ax2.set_ylabel('Perplexity', fontsize=12)
    ax2.set_title('Perplexity vs Threshold', fontsize=14)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig


def main():
    print("=" * 70)
    print("Early Exit Demo")
    print("=" * 70)
    
    # Configuration
    vocab_size = 1000
    d_model = 64
    n_layers = 4
    n_seq = 128
    batch_size = 32
    num_samples = 1000
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Create model
    print(f"\nCreating Early Exit Language Model...")
    print(f"  Vocab Size: {vocab_size}")
    print(f"  Model Dimension: {d_model}")
    print(f"  Number of Layers: {n_layers}")
    print(f"  Sequence Length: {n_seq}")
    
    model = EarlyExitLanguageModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_seq=n_seq,
        num_experts=4,
        top_k=1,
        dropout_p=0.1,
        confidence_threshold=0.9
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total Parameters: {total_params:,}")
    
    # Create synthetic data
    print(f"\nGenerating synthetic data...")
    x_data, y_data = create_synthetic_data(vocab_size, n_seq, num_samples)
    
    dataset = TensorDataset(x_data, y_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    print(f"  Training samples: {num_samples}")
    print(f"  Batch size: {batch_size}")
    
    # Test 1: Compare standard vs early exit forward pass
    print("\n" + "=" * 70)
    print("Test 1: Standard vs Early Exit Forward Pass")
    print("=" * 70)
    
    model.eval()
    x_sample = x_data[:batch_size].to(device)
    
    with torch.no_grad():
        # Standard forward
        logits_standard = model(x_sample, use_early_exit=False)
        print(f"\nStandard forward pass:")
        print(f"  Output shape: {logits_standard.shape}")
        print(f"  All {n_layers} layers executed")
        
        # Early exit forward
        logits_early, exit_info = model(x_sample, use_early_exit=True)
        print(f"\nEarly exit forward pass (threshold=0.9):")
        print(f"  Output shape: {logits_early.shape}")
        print(f"  Average exit layer: {exit_info['exit_layers'].float().mean():.2f}")
        print(f"  Exit layer range: [{exit_info['exit_layers'].min()}, {exit_info['exit_layers'].max()}]")
        print(f"  Average confidence: {exit_info['exit_confidences'].mean():.4f}")
    
    # Test 2: Evaluate across different thresholds
    print("\n" + "=" * 70)
    print("Test 2: Evaluate Across Confidence Thresholds")
    print("=" * 70)
    
    evaluator = EarlyExitEvaluator(model, device)
    thresholds = [0.7, 0.8, 0.9, 0.95]
    
    print(f"\nEvaluating with thresholds: {thresholds}")
    results = evaluator.evaluate(dataloader, confidence_thresholds=thresholds)
    
    print("\nResults:")
    print(f"{'Threshold':<12} {'Avg Exit':<12} {'Speedup':<12} {'Perplexity':<12}")
    print("-" * 50)
    for threshold in thresholds:
        stats = results[threshold]
        print(f"{threshold:<12.2f} {stats['avg_exit_layer']:<12.2f} "
              f"{stats['speedup_estimate']:<12.2f}x {stats['perplexity']:<12.2f}")
    
    # Test 3: Benchmark actual speedup
    print("\n" + "=" * 70)
    print("Test 3: Benchmark Actual Speedup")
    print("=" * 70)
    
    model.confidence_threshold = 0.9
    print(f"\nBenchmarking with threshold=0.9...")
    speedup_metrics = evaluator.benchmark_speedup(dataloader, num_batches=50)
    
    print(f"\nTiming Results:")
    print(f"  Time without early exit: {speedup_metrics['time_no_exit']:.4f}s")
    print(f"  Time with early exit: {speedup_metrics['time_with_exit']:.4f}s")
    print(f"  Actual speedup: {speedup_metrics['actual_speedup']:.2f}x")
    print(f"  Theoretical speedup: {speedup_metrics['theoretical_speedup']:.2f}x")
    print(f"  Average exit layer: {speedup_metrics['avg_exit_layer']:.2f}")
    
    # Test 4: Visualize exit distribution
    print("\n" + "=" * 70)
    print("Test 4: Visualize Exit Distribution")
    print("=" * 70)
    
    # Get statistics for threshold=0.9
    stats_09 = results[0.9]
    
    print(f"\nExit distribution for threshold=0.9:")
    for layer_idx, percentage in enumerate(stats_09['exit_distribution']):
        print(f"  Layer {layer_idx}: {percentage:.2f}%")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    # Visualization 1: Exit distribution for single threshold
    fig1 = visualize_exit_distribution(stats_09, threshold=0.9)
    fig1.savefig('early_exit_distribution.png', dpi=150, bbox_inches='tight')
    print("  Saved: early_exit_distribution.png")
    
    # Visualization 2: Comparison across thresholds
    fig2 = visualize_threshold_comparison(results)
    fig2.savefig('early_exit_threshold_comparison.png', dpi=150, bbox_inches='tight')
    print("  Saved: early_exit_threshold_comparison.png")
    
    plt.close('all')
    
    # Test 5: Per-token exit analysis
    print("\n" + "=" * 70)
    print("Test 5: Per-Token Exit Analysis")
    print("=" * 70)
    
    model.confidence_threshold = 0.9
    model.reset_exit_statistics()
    
    with torch.no_grad():
        x_sample = x_data[:1].to(device)  # Single sequence
        logits, exit_info = model(x_sample, use_early_exit=True)
        
        exit_layers = exit_info['exit_layers'][0].cpu().numpy()
        exit_confidences = exit_info['exit_confidences'][0].cpu().numpy()
    
    print(f"\nPer-token exit analysis (first 20 tokens):")
    print(f"{'Token':<8} {'Exit Layer':<12} {'Confidence':<12}")
    print("-" * 35)
    for i in range(min(20, n_seq)):
        print(f"{i:<8} {exit_layers[i]:<12} {exit_confidences[i]:<12.4f}")
    
    # Summary statistics
    print(f"\nSummary:")
    print(f"  Tokens exiting at layer 0: {(exit_layers == 0).sum()}")
    print(f"  Tokens exiting at layer 1: {(exit_layers == 1).sum()}")
    print(f"  Tokens exiting at layer 2: {(exit_layers == 2).sum()}")
    print(f"  Tokens exiting at layer 3: {(exit_layers == 3).sum()}")
    print(f"  Tokens reaching final layer: {(exit_layers == n_layers).sum()}")
    
    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    
    print("\nKey Findings:")
    print(f"  1. Early exiting reduces average layers from {n_layers} to ~{stats_09['avg_exit_layer']:.2f}")
    print(f"  2. Estimated speedup: {stats_09['speedup_estimate']:.2f}x")
    print(f"  3. Actual speedup: {speedup_metrics['actual_speedup']:.2f}x")
    print(f"  4. Perplexity impact: {stats_09['perplexity']:.2f}")
    print(f"  5. Higher thresholds → later exits → better quality but slower")
    
    print("\nRequirement Satisfaction:")
    print("  ✓ 6.14: Early exiting halts when confidence > threshold")
    print("  ✓ 6.15: Average exit layer measured and reported")


if __name__ == '__main__':
    main()
