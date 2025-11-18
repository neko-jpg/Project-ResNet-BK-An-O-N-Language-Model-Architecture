# WikiText-2 Comprehensive Benchmark

This document describes the WikiText-2 benchmarking infrastructure for evaluating ResNet-BK models with all optimizations enabled.

## Overview

The WikiText-2 benchmark (task 9.2) provides comprehensive evaluation of:
- **Training with all optimizations enabled**: Tests the full ResNet-BK model with all Step 2-7 optimizations
- **Final perplexity measurement**: Evaluates language modeling quality
- **Comparison to Transformer baseline**: Quantifies improvements over standard architecture
- **FLOPs tracking**: Measures computational cost reduction
- **Memory profiling**: Tracks GPU memory usage
- **Wall-clock timing**: Measures real-world training speed

## Quick Start

### Running the Benchmark

```bash
# Run full benchmark (3 models: Transformer, ResNet-BK baseline, ResNet-BK full)
python src/benchmarks/wikitext2_benchmark.py

# Results will be saved to: benchmark_results/wikitext2/
```

### Benchmark Configuration

The benchmark runs three models:

1. **Transformer Baseline**: Standard O(N²) Transformer with MultiheadAttention
2. **ResNet-BK Baseline**: O(N) ResNet-BK without optimizations
3. **ResNet-BK Full**: O(N) ResNet-BK with all optimizations enabled

Default configuration:
- `d_model`: 64
- `n_layers`: 4
- `n_seq`: 128
- `batch_size`: 32
- `epochs`: 5
- `lr`: 1e-3
- `device`: 'cuda' (falls back to 'cpu')

## Architecture

### BenchmarkConfig

Dataclass defining benchmark parameters:

```python
@dataclass
class BenchmarkConfig:
    model_name: str
    d_model: int
    n_layers: int
    n_seq: int
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    grad_clip: float
    device: str
    seed: int
    
    # Optimization flags
    use_analytic_gradient: bool = True
    use_koopman: bool = False
    use_physics_informed: bool = False
    use_quantization: bool = False
    use_pruning: bool = False
    use_mixed_precision: bool = False
    use_act: bool = False
    use_multi_scale: bool = False
    use_sparse_bk: bool = False
    use_early_exit: bool = False
    use_curriculum: bool = False
    use_active_learning: bool = False
```

### BenchmarkResults

Dataclass storing benchmark results:

```python
@dataclass
class BenchmarkResults:
    model_name: str
    config: Dict
    
    # Training metrics
    final_loss: float
    final_perplexity: float
    best_perplexity: float
    training_time: float  # seconds
    
    # FLOPs metrics
    forward_flops: int
    backward_flops: int
    optimizer_flops: int
    total_flops_per_step: int
    total_training_flops: int
    
    # Memory metrics
    peak_memory_mb: float
    model_size_mb: float
    
    # Per-epoch metrics
    epoch_losses: List[float]
    epoch_perplexities: List[float]
    epoch_times: List[float]
```

### WikiText2Benchmark Class

Main benchmark orchestrator:

```python
class WikiText2Benchmark:
    def __init__(self, output_dir: str = "benchmark_results")
    
    def run_benchmark(self, config: BenchmarkConfig) -> BenchmarkResults
    
    def compare_results(self, model1_name: str, model2_name: str)
    
    def plot_training_curves(self, model_names: Optional[List[str]] = None)
```

## Usage Examples

### Example 1: Run Single Model Benchmark

```python
from src.benchmarks.wikitext2_benchmark import WikiText2Benchmark, BenchmarkConfig

# Create benchmark
benchmark = WikiText2Benchmark(output_dir="my_results")

# Configure model
config = BenchmarkConfig(
    model_name='resnet_bk_test',
    d_model=64,
    n_layers=4,
    n_seq=128,
    batch_size=32,
    epochs=5,
    lr=1e-3,
    weight_decay=0.01,
    grad_clip=0.5,
    device='cuda',
    seed=42,
    use_analytic_gradient=True,
    use_mixed_precision=True,
)

# Run benchmark
results = benchmark.run_benchmark(config)

print(f"Final Perplexity: {results.final_perplexity:.2f}")
print(f"Training Time: {results.training_time:.1f}s")
print(f"Total FLOPs: {results.total_training_flops/1e12:.2f} TFLOPs")
```

### Example 2: Compare Multiple Models

```python
from src.benchmarks.wikitext2_benchmark import WikiText2Benchmark, BenchmarkConfig

benchmark = WikiText2Benchmark()

# Run baseline
baseline_config = BenchmarkConfig(
    model_name='baseline',
    d_model=64,
    n_layers=4,
    n_seq=128,
    batch_size=32,
    epochs=5,
    lr=1e-3,
    weight_decay=0.01,
    grad_clip=0.5,
    device='cuda',
    seed=42,
    use_analytic_gradient=False,
)
benchmark.run_benchmark(baseline_config)

# Run optimized
optimized_config = BenchmarkConfig(
    model_name='optimized',
    d_model=64,
    n_layers=4,
    n_seq=128,
    batch_size=32,
    epochs=5,
    lr=1e-3,
    weight_decay=0.01,
    grad_clip=0.5,
    device='cuda',
    seed=42,
    use_analytic_gradient=True,
    use_mixed_precision=True,
    use_act=True,
)
benchmark.run_benchmark(optimized_config)

# Compare
benchmark.compare_results('optimized', 'baseline')

# Plot training curves
benchmark.plot_training_curves()
```

### Example 3: Custom Configuration

```python
from src.benchmarks.wikitext2_benchmark import WikiText2Benchmark, BenchmarkConfig

benchmark = WikiText2Benchmark(output_dir="custom_benchmark")

# Larger model
config = BenchmarkConfig(
    model_name='large_model',
    d_model=128,  # Larger hidden dimension
    n_layers=8,   # More layers
    n_seq=256,    # Longer sequences
    batch_size=16,  # Smaller batch (memory constraint)
    epochs=10,
    lr=5e-4,
    weight_decay=0.01,
    grad_clip=0.5,
    device='cuda',
    seed=42,
    use_analytic_gradient=True,
    use_mixed_precision=True,
    use_act=True,
    use_multi_scale=True,
    use_sparse_bk=True,
)

results = benchmark.run_benchmark(config)
```

## Output Files

The benchmark generates the following files in the output directory:

### Per-Model Results

- `{model_name}_results.json`: Complete results in JSON format
  - Configuration
  - Training metrics (loss, perplexity, time)
  - FLOPs breakdown
  - Memory usage
  - Per-epoch data

### Comparisons

- `comparison_{model1}_vs_{model2}.json`: Detailed comparison
  - Perplexity improvement
  - FLOPs speedup
  - Time speedup
  - Memory reduction
  - Model size reduction

### Visualizations

- `training_curves.png`: Multi-panel plot showing:
  - Training loss curves
  - Perplexity curves
  - Time per epoch
  - Total training FLOPs (bar chart)

## Metrics Explained

### Perplexity

Perplexity = exp(cross_entropy_loss)

- Lower is better
- Measures how well the model predicts the next token
- Target: within 30% of Transformer baseline

### FLOPs

- **Forward FLOPs**: Computational cost of forward pass
- **Backward FLOPs**: Computational cost of backward pass (gradient computation)
- **Optimizer FLOPs**: Computational cost of parameter updates
- **Total FLOPs per Step**: Sum of forward + backward + optimizer
- **Total Training FLOPs**: Total FLOPs × number of training steps

### Speedup

Speedup = Baseline FLOPs / Optimized FLOPs

- Measures computational efficiency improvement
- Target: 10× for architecture (Step 1), 100× for gradients (Step 2), etc.

### Memory

- **Peak Memory**: Maximum GPU memory used during training
- **Model Size**: Size of model parameters in memory

## Optimization Flags

The benchmark supports enabling/disabling individual optimizations:

| Flag | Description | Expected Speedup |
|------|-------------|------------------|
| `use_analytic_gradient` | Analytic gradient computation (Step 2 Phase 1) | 50× backward pass |
| `use_koopman` | Koopman operator learning (Step 2 Phase 2) | 100× gradient cost |
| `use_physics_informed` | Physics-informed learning (Step 2 Phase 3) | 10× training steps |
| `use_quantization` | INT8/INT4 quantization (Step 4) | 4-8× model size |
| `use_pruning` | Structured pruning (Step 4) | 4× model size |
| `use_mixed_precision` | FP16 training (Step 5) | 2× speed, 50% memory |
| `use_act` | Adaptive Computation Time (Step 6) | 30% fewer layers |
| `use_multi_scale` | Multi-scale processing (Step 6) | 2× middle layers |
| `use_sparse_bk` | Learned sparsity in BK-Core (Step 6) | 1.8× at 50% sparsity |
| `use_early_exit` | Early exiting for inference (Step 6) | Variable |
| `use_curriculum` | Curriculum learning (Step 7) | 30% fewer steps |
| `use_active_learning` | Active learning (Step 7) | 50% less data |

## Troubleshooting

### Out of Memory (OOM)

If you encounter OOM errors:

1. Reduce `batch_size` (e.g., 32 → 16 → 8)
2. Reduce `n_seq` (e.g., 128 → 64)
3. Reduce `d_model` (e.g., 64 → 32)
4. Enable `use_mixed_precision=True`
5. Use CPU: `device='cpu'` (slower but no memory limit)

### Slow Training

If training is too slow:

1. Enable `use_mixed_precision=True`
2. Increase `batch_size` (if memory allows)
3. Reduce `epochs` for quick testing
4. Use GPU: `device='cuda'`

### NaN/Inf Loss

If loss becomes NaN or Inf:

1. Reduce `lr` (e.g., 1e-3 → 5e-4 → 1e-4)
2. Increase `grad_clip` (e.g., 0.5 → 1.0)
3. Check numerical stability settings in model
4. Disable problematic optimizations

## Integration with FLOPs Counter

The benchmark uses the `FLOPsCounter` class (task 9.1) to measure computational cost:

```python
from src.benchmarks.flops_counter import FLOPsCounter

# Count FLOPs for a model
counter = FLOPsCounter(model, batch_size=32, seq_len=128)
flops = counter.count_total_flops()

print(f"Forward: {flops.forward:,} FLOPs")
print(f"Backward: {flops.backward:,} FLOPs")
print(f"Optimizer: {flops.optimizer:,} FLOPs")
print(f"Total: {flops.total:,} FLOPs")
```

## Expected Results

Based on the design specifications, expected results for ResNet-BK Full vs Transformer Baseline:

| Metric | Transformer | ResNet-BK Full | Improvement |
|--------|-------------|----------------|-------------|
| Perplexity | ~30 | ~35-40 | Within 30% |
| Forward FLOPs | O(N²) | O(N) | 10× at N=2048 |
| Backward FLOPs | Standard BP | Analytic | 50-100× |
| Training Time | Baseline | Optimized | 5-10× |
| Memory Usage | Baseline | Reduced | 30-50% |
| Model Size | Baseline | Compressed | 4-100× |

## References

- Task 9.1: FLOPs Counter Implementation
- Task 9.2: WikiText-2 Benchmark (this document)
- Requirements 8.15, 9.1: Benchmark specifications
- Design Document: Step-by-step optimization details

## See Also

- `src/benchmarks/flops_counter.py`: FLOPs counting infrastructure
- `src/models/configurable_resnet_bk.py`: Configurable ResNet-BK model
- `src/utils/data_utils.py`: Data loading utilities
- `tests/test_wikitext2_benchmark.py`: Unit tests
