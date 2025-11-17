"""
Run Penn Treebank Benchmark

This script runs the comprehensive Penn Treebank benchmark to evaluate
ResNet-BK on a different domain (financial news) compared to WikiText.

Usage:
    python run_penn_treebank_benchmark.py [--device cuda] [--epochs 5]
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.benchmarks.penn_treebank_benchmark import (
    PennTreebankBenchmark,
    BenchmarkConfig
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run Penn Treebank benchmark for ResNet-BK'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use for training (default: cuda)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=5,
        help='Number of training epochs (default: 5)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size (default: 32)'
    )
    
    parser.add_argument(
        '--d-model',
        type=int,
        default=64,
        help='Model dimension (default: 64)'
    )
    
    parser.add_argument(
        '--n-layers',
        type=int,
        default=4,
        help='Number of layers (default: 4)'
    )
    
    parser.add_argument(
        '--n-seq',
        type=int,
        default=128,
        help='Sequence length (default: 128)'
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3,
        help='Learning rate (default: 1e-3)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='benchmark_results/penn_treebank',
        help='Output directory for results (default: benchmark_results/penn_treebank)'
    )
    
    parser.add_argument(
        '--skip-transformer',
        action='store_true',
        help='Skip Transformer baseline benchmark'
    )
    
    parser.add_argument(
        '--skip-baseline',
        action='store_true',
        help='Skip ResNet-BK baseline benchmark'
    )
    
    parser.add_argument(
        '--data-limit',
        type=int,
        default=None,
        help='Limit number of tokens for testing (default: None = use all)'
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    print("=" * 80)
    print("Penn Treebank Benchmark")
    print("=" * 80)
    print(f"Device: {args.device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Model dimension: {args.d_model}")
    print(f"Number of layers: {args.n_layers}")
    print(f"Sequence length: {args.n_seq}")
    print(f"Learning rate: {args.lr}")
    print(f"Output directory: {args.output_dir}")
    if args.data_limit:
        print(f"Data limit: {args.data_limit:,} tokens")
    print("=" * 80 + "\n")
    
    # Create benchmark
    benchmark = PennTreebankBenchmark(output_dir=args.output_dir)
    
    # Common configuration
    common_config = {
        'd_model': args.d_model,
        'n_layers': args.n_layers,
        'n_seq': args.n_seq,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'lr': args.lr,
        'weight_decay': 0.01,
        'grad_clip': 0.5,
        'device': args.device,
        'seed': 42,
        'data_limit': args.data_limit,
    }
    
    benchmarks_to_run = []
    
    # 1. Transformer Baseline
    if not args.skip_transformer:
        benchmarks_to_run.append(('Transformer Baseline', BenchmarkConfig(
            model_name='transformer_baseline',
            **common_config,
            use_analytic_gradient=False,
        )))
    
    # 2. ResNet-BK Baseline
    if not args.skip_baseline:
        benchmarks_to_run.append(('ResNet-BK Baseline', BenchmarkConfig(
            model_name='resnet_bk_baseline',
            **common_config,
            use_analytic_gradient=False,
        )))
    
    # 3. ResNet-BK with All Optimizations
    benchmarks_to_run.append(('ResNet-BK Full', BenchmarkConfig(
        model_name='resnet_bk_full',
        **common_config,
        use_analytic_gradient=True,
        use_mixed_precision=True,
        use_act=True,
        use_multi_scale=True,
        use_sparse_bk=True,
    )))
    
    # Run benchmarks
    for i, (name, config) in enumerate(benchmarks_to_run, 1):
        print(f"\n[{i}/{len(benchmarks_to_run)}] Running {name}...")
        try:
            benchmark.run_benchmark(config)
        except Exception as e:
            print(f"Error running {name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Compare to other datasets if available
    print("\n" + "=" * 80)
    print("CROSS-DATASET COMPARISON")
    print("=" * 80)
    
    wt2_path = "benchmark_results/wikitext2/resnet_bk_full_results.json"
    wt103_path = "benchmark_results/wikitext103/resnet_bk_full_wikitext103_results.json"
    
    benchmark.compare_to_other_datasets(
        wikitext2_results_path=wt2_path if Path(wt2_path).exists() else None,
        wikitext103_results_path=wt103_path if Path(wt103_path).exists() else None
    )
    
    # Plot training curves
    try:
        benchmark.plot_training_curves()
    except Exception as e:
        print(f"Warning: Could not generate plots: {e}")
    
    # Final summary
    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {args.output_dir}")
    print("\nKey Findings:")
    for name, results in benchmark.results.items():
        print(f"  {name}:")
        print(f"    - Final Perplexity: {results.final_perplexity:.2f}")
        print(f"    - Best Perplexity: {results.best_perplexity:.2f}")
        print(f"    - Training Time: {results.training_time/60:.1f} minutes")
        print(f"    - Total Tokens: {results.total_tokens:,}")
        print(f"    - Vocab Size: {results.vocab_size:,}")
    
    print("\nDomain Analysis:")
    print("  Penn Treebank: Financial news (Wall Street Journal)")
    print("  WikiText: Wikipedia articles (general knowledge)")
    print("  Performance comparison shows model's domain generalization capability")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
