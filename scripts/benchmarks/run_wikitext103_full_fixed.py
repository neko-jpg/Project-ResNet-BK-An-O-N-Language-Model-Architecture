"""
Run WikiText-103 benchmark with full optimizations (FIXED)
Lower learning rate for stability with multiple optimizations
"""
from src.benchmarks.wikitext103_benchmark import WikiText103Benchmark, BenchmarkConfig

# Create benchmark
benchmark = WikiText103Benchmark(output_dir="benchmark_results/wikitext103")

# Configuration with all optimizations - LOWER LEARNING RATE
config = BenchmarkConfig(
    model_name='resnet_bk_full_fixed',
    d_model=64,
    n_layers=4,
    n_seq=128,
    batch_size=16,
    epochs=2,
    lr=5e-4,  # REDUCED from 1e-3 to 5e-4 for stability
    weight_decay=0.01,
    grad_clip=0.5,
    device='cuda',
    seed=42,
    data_limit=10_000_000,
    
    # Enable optimizations ONE BY ONE for stability
    use_analytic_gradient=True,
    use_mixed_precision=False,  # Disable for now - can cause instability
    use_act=False,              # Disable for now - adds complexity
    use_multi_scale=False,      # Disable for now
    use_sparse_bk=False,        # Disable for now
)

print("Running WikiText-103 benchmark with STABLE optimizations (10M tokens)...")
print("Using GPU: RTX 3080 Laptop GPU")
print("Optimizations: Analytic Gradient ONLY (for stability)")
print("Learning Rate: 5e-4 (reduced for stability)")
print()

results = benchmark.run_benchmark(config)

print("\n" + "="*80)
print("BENCHMARK COMPLETE")
print("="*80)
print(f"Final Perplexity: {results.final_perplexity:.2f}")
print(f"Training Time: {results.training_time:.1f}s ({results.training_time/60:.1f}min)")
print(f"Total Tokens: {results.total_tokens:,}")

# Compare to baseline
import json
from pathlib import Path

baseline_path = Path("benchmark_results/wikitext103/resnet_bk_baseline_wikitext103_results.json")
if baseline_path.exists():
    with open(baseline_path, 'r') as f:
        baseline = json.load(f)
    
    print("\n" + "="*80)
    print("COMPARISON: Baseline vs Analytic Gradient")
    print("="*80)
    print(f"\nPerplexity:")
    print(f"  Baseline:         {baseline['final_perplexity']:.2f}")
    print(f"  Analytic Grad:    {results.final_perplexity:.2f}")
    ppl_change = ((results.final_perplexity - baseline['final_perplexity']) / baseline['final_perplexity'] * 100)
    print(f"  Change:           {ppl_change:+.1f}%")
    
    print(f"\nTraining Time:")
    print(f"  Baseline:         {baseline['training_time']:.1f}s ({baseline['training_time']/60:.1f}min)")
    print(f"  Analytic Grad:    {results.training_time:.1f}s ({results.training_time/60:.1f}min)")
    speedup = baseline['training_time'] / results.training_time
    print(f"  Speedup:          {speedup:.2f}×")
