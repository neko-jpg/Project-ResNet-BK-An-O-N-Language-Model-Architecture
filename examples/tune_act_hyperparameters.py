"""
Script to tune ACT hyperparameters using grid search.

Usage:
    python examples/tune_act_hyperparameters.py
    
    # With custom parameters:
    python examples/tune_act_hyperparameters.py --epochs 5 --batch-size 64
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import torch
from pathlib import Path

from src.training.act_hyperparameter_tuner import ACTHyperparameterTuner
from src.utils.data_utils import get_wikitext2_dataloaders


def parse_args():
    parser = argparse.ArgumentParser(description='Tune ACT hyperparameters')
    
    # Data parameters
    parser.add_argument('--data-path', type=str, default='data',
                       help='Path to data directory')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    
    # Model parameters
    parser.add_argument('--d-model', type=int, default=64,
                       help='Hidden dimension')
    parser.add_argument('--n-layers', type=int, default=4,
                       help='Number of layers')
    parser.add_argument('--n-seq', type=int, default=128,
                       help='Sequence length')
    parser.add_argument('--num-experts', type=int, default=4,
                       help='Number of MoE experts')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of epochs per configuration')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    
    # Hyperparameter search space
    parser.add_argument('--thresholds', type=float, nargs='+',
                       default=[0.5, 0.8, 0.9, 0.95, 0.99],
                       help='Halting threshold values to search')
    parser.add_argument('--lambdas', type=float, nargs='+',
                       default=[0.001, 0.005, 0.01, 0.05, 0.1],
                       help='Lambda (ponder cost weight) values to search')
    parser.add_argument('--score-metric', type=str, default='balanced',
                       choices=['perplexity', 'layers', 'balanced'],
                       help='Metric to optimize')
    
    # Output parameters
    parser.add_argument('--output-dir', type=str, default='results/act_tuning',
                       help='Directory to save results')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 70)
    print("ACT Hyperparameter Tuning")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Model: d_model={args.d_model}, n_layers={args.n_layers}, n_seq={args.n_seq}")
    print(f"  Training: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")
    print(f"  Search space:")
    print(f"    Thresholds: {args.thresholds}")
    print(f"    Lambdas: {args.lambdas}")
    print(f"  Score metric: {args.score_metric}")
    print(f"  Output directory: {args.output_dir}")
    print("=" * 70)
    
    # Load data
    print("\nLoading WikiText-2 dataset...")
    train_loader, val_loader, test_loader, vocab_size = get_wikitext2_dataloaders(
        data_path=args.data_path,
        batch_size=args.batch_size,
        n_seq=args.n_seq
    )
    print(f"Vocabulary size: {vocab_size}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Create tuner
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    tuner = ACTHyperparameterTuner(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_seq=args.n_seq,
        num_experts=args.num_experts,
        device=device
    )
    
    # Run grid search
    print("\nStarting grid search...")
    results = tuner.grid_search(
        train_loader=train_loader,
        val_loader=val_loader,
        threshold_values=args.thresholds,
        lambda_values=args.lambdas,
        num_epochs=args.epochs,
        lr=args.lr,
        score_metric=args.score_metric
    )
    
    # Save results
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    tuner.save_results(f'{args.output_dir}/act_tuning_results.json')
    
    # Plot results
    try:
        tuner.plot_results(save_path=f'{args.output_dir}/act_tuning_heatmap.png')
    except Exception as e:
        print(f"Could not generate plot: {e}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("Tuning Complete!")
    print("=" * 70)
    
    best_result = results['best_result']
    print(f"\nBest Configuration:")
    print(f"  act_threshold = {results['best_config']['threshold']}")
    print(f"  act_lambda = {results['best_config']['lambda']}")
    print(f"\nPerformance:")
    print(f"  Validation Perplexity: {best_result['final_val_perplexity']:.2f}")
    print(f"  Avg Layers Executed: {best_result['avg_layers_executed']:.2f} / {args.n_layers}")
    print(f"  Speedup Potential: {args.n_layers / best_result['avg_layers_executed']:.2f}x")
    print(f"  Computational Savings: {(1 - best_result['avg_layers_executed']/args.n_layers)*100:.1f}%")
    
    # Compare to baseline (all layers)
    print(f"\nComparison to Baseline (all {args.n_layers} layers):")
    print(f"  Layers reduction: {args.n_layers - best_result['avg_layers_executed']:.2f} layers")
    print(f"  Expected inference speedup: {args.n_layers / best_result['avg_layers_executed']:.2f}x")
    
    print(f"\nResults saved to: {args.output_dir}/")
    print("  - act_tuning_results.json (detailed results)")
    print("  - act_tuning_heatmap.png (visualization)")
    
    # Print top 5 configurations
    print("\n" + "=" * 70)
    print("Top 5 Configurations:")
    print("=" * 70)
    sorted_results = sorted(results['all_results'], key=lambda x: x['score'])
    for i, result in enumerate(sorted_results[:5], 1):
        print(f"\n{i}. threshold={result['threshold']}, lambda={result['lambda']}")
        print(f"   Perplexity: {result['final_val_perplexity']:.2f}")
        print(f"   Avg Layers: {result['avg_layers_executed']:.2f}")
        print(f"   Score: {result['score']:.4f}")


if __name__ == '__main__':
    main()
