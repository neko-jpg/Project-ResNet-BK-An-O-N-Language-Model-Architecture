# C4 Benchmark Documentation

## Overview

The C4 (Colossal Clean Crawled Corpus) benchmark evaluates ResNet-BK on a large-scale, diverse web-crawled dataset. This benchmark implements **Task 9.5** from the implementation plan.

## Dataset

**C4 (Colossal Clean Crawled Corpus)**:
- **Source**: Web-crawled text from Common Crawl
- **Size**: 100M tokens (configurable)
- **Domains**: Diverse (news, blogs, forums, technical documentation, etc.)
- **Vocabulary**: ~32K tokens
- **Characteristics**: 
  - Most diverse and challenging dataset
  - Multiple domains and writing styles
  - Real-world web text with varying quality
  - Tests model's generalization capability

## Benchmark Configuration

### Default Settings

```python
{
    'd_model': 64,
    'n_layers': 4,
    'n_seq': 128,
    'batch_size': 32,
    'epochs': 2,  # Fewer epochs due to large dataset
    'lr': 1e-3,
    'weight_decay': 0.01,
    'grad_clip': 0.5,
    'device': 'cuda',
    'seed': 42,
    'data_limit': 100_000_000,  # 100M tokens
}
```

### Models Tested

1. **ResNet-BK Baseline**: No optimizations
2. **ResNet-BK Full**: All optimizations enabled
   - Analytic gradient
   - Mixed precision
   - Adaptive computation (ACT)
   - Multi-scale processing
   - Learned sparsity

## Running the Benchmark

### Basic Usage

```bash
python run_c4_benchmark.py
```

### Programmatic Usage

```python
from src.benchmarks.c4_benchmark import C4Benchmark, BenchmarkConfig

# Create benchmark
benchmark = C4Benchmark(output_dir="benchmark_results/c4")

# Configure
config = BenchmarkConfig(
    model_name='resnet_bk_full',
    d_model=64,
    n_layers=4,
    n_seq=128,
    batch_size=32,
    epochs=2,
    lr=1e-3,
    weight_decay=0.01,
    grad_clip=0.5,
    device='cuda',
    seed=42,
    data_limit=100_000_000,
    use_analytic_gradient=True,
    use_mixed_precision=True,
    use_act=True,
    use_multi_scale=True,
    use_sparse_bk=True,
)

# Run benchmark
results = benchmark.run_benchmark(config)

# Compare to other datasets
benchmark.compare_to_other_datasets(
    wikitext2_results_path="benchmark_results/wikitext2/resnet_bk_full_results.json",
    wikitext103_results_path="benchmark_results/wikitext103/resnet_bk_full_wikitext103_results.json",
    penn_treebank_results_path="benchmark_results/penn_treebank/resnet_bk_full_penn_treebank_results.json"
)

# Plot results
benchmark.plot_training_curves()
```

## Metrics Tracked

### Training Metrics
- **Final Loss**: Cross-entropy loss after training
- **Final Perplexity**: exp(final_loss)
- **Best Perplexity**: Lowest perplexity achieved
- **Training Time**: Total wall-clock time (seconds)

### Domain-Specific Metrics
- **General Domain**: First 1000 validation examples
- **Technical Domain**: Next 1000 validation examples
- **Diverse Domain**: Another 1000 validation examples

### Computational Metrics
- **Forward FLOPs**: FLOPs per forward pass
- **Backward FLOPs**: FLOPs per backward pass
- **Optimizer FLOPs**: FLOPs per optimizer step
- **Total Training FLOPs**: Cumulative FLOPs

### Memory Metrics
- **Peak Memory**: Maximum GPU memory usage (MB)
- **Model Size**: Model parameter memory (MB)

## Output Files

### Results JSON
```
benchmark_results/c4/
├── resnet_bk_baseline_c4_results.json
├── resnet_bk_full_c4_results.json
├── c4_cross_dataset_comparison.json
└── c4_training_curves.png
```

### Results Structure
```json
{
  "model_name": "resnet_bk_full",
  "dataset_name": "c4",
  "final_perplexity": 45.2,
  "best_perplexity": 43.8,
  "training_time": 3600.0,
  "total_tokens": 100000000,
  "vocab_size": 32000,
  "domain_perplexities": {
    "general": 44.5,
    "technical": 46.8,
    "diverse": 45.1
  },
  "total_training_flops": 5.2e14,
  "peak_memory_mb": 2048.0,
  "epoch_losses": [3.9, 3.8],
  "epoch_perplexities": [49.4, 45.2]
}
```

## Cross-Dataset Comparison

The benchmark automatically compares C4 results to other datasets:

### Dataset Characteristics

| Dataset | Tokens | Vocab | Domain | Difficulty |
|---------|--------|-------|--------|------------|
| WikiText-2 | ~2M | ~30K | Wikipedia | Easy |
| WikiText-103 | ~100M | ~30K | Wikipedia | Medium |
| Penn Treebank | ~1M | ~10K | Financial News | Easy |
| **C4** | **100M** | **32K** | **Web (Diverse)** | **Hard** |

### Expected Performance

- **C4 Perplexity**: Higher than WikiText (more diverse)
- **Domain Variation**: Significant perplexity differences across domains
- **Generalization**: Tests model's ability to handle diverse text

## Domain Analysis

### General Domain
- News articles
- Blog posts
- General web content
- Expected: Moderate perplexity

### Technical Domain
- Technical documentation
- Scientific articles
- Specialized content
- Expected: Higher perplexity (specialized vocabulary)

### Diverse Domain
- Mixed content types
- Forums, discussions
- Varied writing styles
- Expected: Variable perplexity

## Performance Expectations

### ResNet-BK Baseline
- **Perplexity**: ~50-60
- **Training Time**: ~60-90 minutes (GPU)
- **Memory**: ~2-3 GB

### ResNet-BK Full (Optimized)
- **Perplexity**: ~45-55 (within 30% of baseline)
- **Training Time**: ~30-45 minutes (2× faster)
- **Memory**: ~1-2 GB (50% reduction)
- **FLOPs**: 10× reduction vs Transformer

## Troubleshooting

### Dataset Download Issues
```python
# If C4 download fails, try:
from datasets import load_dataset
dataset = load_dataset("c4", "en", split="train", streaming=True)
```

### Memory Issues
```python
# Reduce batch size or data limit
config.batch_size = 16  # Instead of 32
config.data_limit = 50_000_000  # Instead of 100M
```

### Slow Training
```python
# Enable mixed precision
config.use_mixed_precision = True

# Reduce epochs
config.epochs = 1
```

## Testing

Run tests with:
```bash
pytest tests/test_c4_benchmark.py -v
```

Skip slow tests:
```bash
pytest tests/test_c4_benchmark.py -v -m "not slow"
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- datasets library
- Stable internet connection (for dataset download)
- GPU recommended (CPU training will be very slow)

## References

- **C4 Dataset**: [Raffel et al., 2020](https://arxiv.org/abs/1910.10683)
- **Task 9.5**: Train on 100M tokens, measure perplexity across domains
- **Requirement 9.3**: Evaluate on C4, train on 100M tokens, measure perplexity

## Notes

- C4 is the most challenging benchmark due to domain diversity
- Higher perplexity on C4 vs WikiText is expected and normal
- Domain-specific perplexities provide insights into model strengths/weaknesses
- Training on 100M tokens tests scalability and efficiency
- Results demonstrate real-world performance on diverse web text
