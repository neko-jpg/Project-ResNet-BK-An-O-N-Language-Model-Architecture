"""
Run WikiText-103 benchmark with limited data for testing
"""
from src.benchmarks.wikitext103_benchmark import WikiText103Benchmark, BenchmarkConfig

# Create benchmark
benchmark = WikiText103Benchmark(output_dir="benchmark_results/wikitext103")

# Configuration with limited data for CPU testing
config = BenchmarkConfig(
    model_name='resnet_bk_baseline',
    d_model=64,
    n_layers=4,
    n_seq=128,
    batch_size=16,  # Smaller batch for CPU
    epochs=2,  # Fewer epochs for testing
    lr=1e-3,
    weight_decay=0.01,
    grad_clip=0.5,
    device='cuda',  # Use GPU (RTX 3080)
    seed=42,
    data_limit=10_000_000,  # Limit to 10M tokens (10% of full dataset)
    use_analytic_gradient=False,
)

print("Running WikiText-103 benchmark with limited data (10M tokens)...")
print("Using GPU: RTX 3080 Laptop GPU")
print("This is ~10% of the full WikiText-103 dataset")
print()

results = benchmark.run_benchmark(config)

print("\n" + "="*80)
print("BENCHMARK COMPLETE")
print("="*80)
print(f"Final Perplexity: {results.final_perplexity:.2f}")
print(f"Training Time: {results.training_time:.1f}s ({results.training_time/60:.1f}min)")
print(f"Total Tokens: {results.total_tokens:,}")
print(f"Results saved to: benchmark_results/wikitext103/")
