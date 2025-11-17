"""
Run The Pile benchmark.

Usage:
    python run_pile_benchmark.py [--device cuda] [--data-limit 1000000000] [--epochs 2]
"""

import argparse
from pathlib import Path
from src.benchmarks.pile_benchmark import PileBenchmark, BenchmarkConfig


def main():
    parser = argparse.ArgumentParser(description='Run The Pile benchmark')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='Device to use (default: cuda)')
    parser.add_argument('--d-model', type=int, default=64,
                        help='Model dimension (default: 64)')
    parser.add_argument('--n-layers', type=int, default=4,
                        help='Number of layers (default: 4)')
    parser.add_argument('--n-seq', type=int, default=128,
                        help='Sequence length (default: 128)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--epochs', type=int, default=2,
                        help='Number of epochs (default: 2)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--data-limit', type=int, default=1_000_000_000,
                        help='Maximum number of tokens (default: 1B)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--output-dir', type=str, default='benchmark_results/pile',
                        help='Output directory (default: benchmark_results/pile)')
    parser.add_argument('--baseline-only', action='store_true',
                        help='Run only baseline model')
    parser.add_argument('--optimized-only', action='store_true',
                        help='Run only optimized model')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("The Pile Comprehensive Benchmark")
    print("=" * 80)
    print(f"Device: {args.device}")
    print(f"Model: d_model={args.d_model}, n_layers={args.n_layers}, n_seq={args.n_seq}")
    print(f"Training: batch_size={args.batch_size}, epochs={args.epochs}, lr={args.lr}")
    print(f"Data limit: {args.data_limit:,} tokens")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80 + "\n")
    
    # Create benchmark
    benchmark = PileBenchmark(output_dir=args.output_dir)
    
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
        'seed': args.seed,
        'data_limit': args.data_limit,
    }
    
    # Run benchmarks
    if not args.optimized_only:
        print("\n[1/2] Running ResNet-BK Baseline on The Pile...")
        baseline_config = BenchmarkConfig(
            model_name='resnet_bk_baseline',
            **common_config,
            use_analytic_gradient=False,
        )
        benchmark.run_benchmark(baseline_config)
    
    if not args.baseline_only:
        print("\n[2/2] Running ResNet-BK with All Optimizations on The Pile...")
        optimized_config = BenchmarkConfig(
            model_name='resnet_bk_full',
            **common_config,
            use_analytic_gradient=True,
            use_mixed_precision=True,
            use_act=True,
            use_multi_scale=True,
            use_sparse_bk=True,
        )
        benchmark.run_benchmark(optimized_config)
    
    # Compare to other datasets if available
    print("\n" + "=" * 80)
    print("CROSS-DATASET COMPARISON")
    print("=" * 80)
    
    wt2_path = "benchmark_results/wikitext2/resnet_bk_full_results.json"
    wt103_path = "benchmark_results/wikitext103/resnet_bk_full_wikitext103_results.json"
    ptb_path = "benchmark_results/penn_treebank/resnet_bk_full_penn_treebank_results.json"
    c4_path = "benchmark_results/c4/resnet_bk_full_c4_results.json"
    
    benchmark.compare_to_other_datasets(
        wikitext2_results_path=wt2_path if Path(wt2_path).exists() else None,
        wikitext103_results_path=wt103_path if Path(wt103_path).exists() else None,
        penn_treebank_results_path=ptb_path if Path(ptb_path).exists() else None,
        c4_results_path=c4_path if Path(c4_path).exists() else None
    )
    
    # Plot training curves
    benchmark.plot_training_curves()
    
    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {benchmark.output_dir}")
    print("\nKey Findings:")
    for name, results in benchmark.results.items():
        print(f"  {name}:")
        print(f"    - Final Perplexity: {results.final_perplexity:.2f}")
        print(f"    - Training Time: {results.training_time/60:.1f} minutes")
        print(f"    - Total Tokens: {results.total_tokens:,}")
        print(f"    - Domain Perplexities:")
        for domain, ppl in sorted(results.domain_perplexities.items(), key=lambda x: x[1])[:5]:
            print(f"      - {domain}: {ppl:.2f}")
        if len(results.domain_perplexities) > 5:
            print(f"      - ... and {len(results.domain_perplexities) - 5} more domains")


if __name__ == '__main__':
    main()
