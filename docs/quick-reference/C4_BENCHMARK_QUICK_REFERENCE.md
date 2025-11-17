# C4 Benchmark Quick Reference

## Quick Start

```bash
# Run full benchmark
python run_c4_benchmark.py

# Run tests
pytest tests/test_c4_benchmark.py -v
```

## Key Commands

```python
from src.benchmarks.c4_benchmark import C4Benchmark, BenchmarkConfig

# Create benchmark
benchmark = C4Benchmark(output_dir="benchmark_results/c4")

# Run with default config
config = BenchmarkConfig(
    model_name='resnet_bk_full',
    d_model=64, n_layers=4, n_seq=128,
    batch_size=32, epochs=2,
    lr=1e-3, device='cuda', seed=42,
    data_limit=100_000_000,  # 100M tokens
    use_analytic_gradient=True,
    use_mixed_precision=True,
)

results = benchmark.run_benchmark(config)
```

## Dataset Info

- **Name**: C4 (Colossal Clean Crawled Corpus)
- **Size**: 100M tokens (configurable)
- **Domains**: Web-crawled (news, blogs, forums, technical)
- **Vocab**: ~32K tokens
- **Difficulty**: Highest (most diverse)

## Expected Results

| Model | Perplexity | Time (min) | Memory (GB) |
|-------|------------|------------|-------------|
| Baseline | ~50-60 | 60-90 | 2-3 |
| Optimized | ~45-55 | 30-45 | 1-2 |

## Domain Perplexities

- **General**: ~44-46 (news, blogs)
- **Technical**: ~46-48 (documentation, scientific)
- **Diverse**: ~45-47 (mixed content)

## Output Files

```
benchmark_results/c4/
├── resnet_bk_baseline_c4_results.json
├── resnet_bk_full_c4_results.json
├── c4_cross_dataset_comparison.json
└── c4_training_curves.png
```

## Comparison to Other Datasets

| Dataset | Tokens | Difficulty | Expected PPL |
|---------|--------|------------|--------------|
| WikiText-2 | 2M | Easy | ~30-40 |
| WikiText-103 | 100M | Medium | ~35-45 |
| Penn Treebank | 1M | Easy | ~25-35 |
| **C4** | **100M** | **Hard** | **45-55** |

## Troubleshooting

**Dataset download fails:**
```bash
pip install datasets --upgrade
```

**Out of memory:**
```python
config.batch_size = 16  # Reduce from 32
config.data_limit = 50_000_000  # Reduce from 100M
```

**Slow training:**
```python
config.use_mixed_precision = True
config.epochs = 1  # Reduce from 2
```

## Task 9.5 Requirements

✅ Train on 100M tokens from C4  
✅ Measure perplexity across domains  
✅ Compare to WikiText-2, WikiText-103, Penn Treebank  
✅ Track FLOPs, time, memory  
✅ Generate plots and reports  

## Key Metrics

- **Final Perplexity**: Overall model performance
- **Domain Perplexities**: Performance across content types
- **Training Time**: Wall-clock time in minutes
- **Total FLOPs**: Computational cost
- **Peak Memory**: GPU memory usage

## Documentation

- Full docs: `docs/C4_BENCHMARK.md`
- Tests: `tests/test_c4_benchmark.py`
- Implementation: `src/benchmarks/c4_benchmark.py`
- Runner: `run_c4_benchmark.py`
