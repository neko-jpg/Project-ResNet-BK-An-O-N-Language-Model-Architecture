"""
Run WikiText-103 benchmark with full optimizations
"""
from src.benchmarks.wikitext103_benchmark import WikiText103Benchmark, BenchmarkConfig

# Create benchmark
benchmark = WikiText103Benchmark(output_dir="benchmark_results/wikitext103")

# Configuration with all optimizations
config = BenchmarkConfig(
    model_name='resnet_bk_full',
    d_model=64,
    n_layers=4,
    n_seq=128,
    batch_size=16,
    epochs=2,
    lr=1e-3,
    weight_decay=0.01,
    grad_clip=0.5,
    device='cuda',  # Use GPU (RTX 3080)
    seed=42,
    data_limit=10_000_000,  # Limit to 10M tokens (10% of full dataset)
    
    # Enable all optimizations
    use_analytic_gradient=True,
    use_mixed_precision=True,
    use_act=True,
    use_multi_scale=True,
    use_sparse_bk=True,
)

print("Running WikiText-103 benchmark with ALL OPTIMIZATIONS (10M tokens)...")
print("Using GPU: RTX 3080 Laptop GPU")
print("Optimizations: Analytic Gradient + Mixed Precision + ACT + Multi-Scale + Sparse BK")
print()

results = benchmark.run_benchmark(config)

print("\n" + "="*80)
print("BENCHMARK COMPLETE")
print("="*80)
print(f"Final Perplexity: {results.final_perplexity:.2f}")
print(f"Training Time: {results.training_time:.1f}s ({results.training_time/60:.1f}min)")
print(f"Total Tokens: {results.total_tokens:,}")
print(f"Results saved to: benchmark_results/wikitext103/")

# Compare to baseline if available
import json
from pathlib import Path

baseline_path = Path("benchmark_results/wikitext103/resnet_bk_baseline_wikitext103_results.json")
if baseline_path.exists():
    with open(baseline_path, 'r') as f:
        baseline = json.load(f)
    
    print("\n" + "="*80)
    print("COMPARISON: Baseline vs Full Optimizations")
    print("="*80)
    print(f"\nPerplexity:")
    print(f"  Baseline: {baseline['final_perplexity']:.2f}")
    print(f"  Full:     {results.final_perplexity:.2f}")
    print(f"  Change:   {((results.final_perplexity - baseline['final_perplexity']) / baseline['final_perplexity'] * 100):+.1f}%")
    
    print(f"\nTraining Time:")
    print(f"  Baseline: {baseline['training_time']:.1f}s ({baseline['training_time']/60:.1f}min)")
    print(f"  Full:     {results.training_time:.1f}s ({results.training_time/60:.1f}min)")
    speedup = baseline['training_time'] / results.training_time
    print(f"  Speedup:  {speedup:.2f}×")
    
    print(f"\nTotal FLOPs:")
    print(f"  Baseline: {baseline['total_training_flops']/1e12:.2f} TFLOPs")
    print(f"  Full:     {results.total_training_flops/1e12:.2f} TFLOPs")
