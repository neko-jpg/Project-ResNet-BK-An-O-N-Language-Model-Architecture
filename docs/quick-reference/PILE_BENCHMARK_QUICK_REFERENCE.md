# The Pile Benchmark - Quick Reference

## Quick Start

```bash
# Run full benchmark (1B tokens, ~2-4 hours on GPU)
python run_pile_benchmark.py

# Quick test (10M tokens, ~5-10 minutes)
python run_pile_benchmark.py --data-limit 10000000 --epochs 1

# Baseline only
python run_pile_benchmark.py --baseline-only

# Optimized only
python run_pile_benchmark.py --optimized-only
```

## The Pile Dataset

**Size**: 825 GiB (1B token subset for benchmark)
**Domains**: 22 diverse domains
**Characteristics**: Most comprehensive and challenging dataset

### 22 Domains
1. Pile-CC (web)
2. PubMed Central (medical)
3. Books3 (books)
4. OpenWebText2 (Reddit)
5. ArXiv (science)
6. Github (code)
7. FreeLaw (legal)
8. StackExchange (Q&A)
9. USPTO (patents)
10. PubMed Abstracts (medical)
11. Gutenberg (books)
12. OpenSubtitles (subtitles)
13. Wikipedia (encyclopedia)
14. DM Mathematics (math)
15. Ubuntu IRC (chat)
16. BookCorpus2 (books)
17. EuroParl (government)
18. HackerNews (tech news)
19. YoutubeSubtitles (video)
20. PhilPapers (philosophy)
21. NIH ExPorter (research)
22. Enron Emails (email)

## Key Commands

```bash
# Custom configuration
python run_pile_benchmark.py \
    --device cuda \
    --data-limit 1000000000 \
    --epochs 2 \
    --batch-size 32 \
    --d-model 64 \
    --n-layers 4

# CPU mode (slower but works without GPU)
python run_pile_benchmark.py --device cpu --data-limit 10000000

# Smaller model for testing
python run_pile_benchmark.py \
    --d-model 32 \
    --n-layers 2 \
    --batch-size 16 \
    --data-limit 10000000
```

## Expected Results

### Perplexity
- **Baseline**: ~150-200 (higher than other datasets)
- **Optimized**: ~120-150
- **Domain Range**: 2-3× variation across domains

### Training Time (1B tokens, GPU)
- **Baseline**: ~3-4 hours
- **Optimized**: ~2-3 hours

### Domain Difficulty (typical)
- **Easiest**: Wikipedia, Books, PubMed (~80-100 PPL)
- **Medium**: Pile-CC, OpenWebText2 (~120-150 PPL)
- **Hardest**: Github, IRC, Emails (~180-250 PPL)

## Output Files

```
benchmark_results/pile/
├── resnet_bk_baseline_pile_results.json
├── resnet_bk_full_pile_results.json
├── pile_cross_dataset_comparison.json
└── pile_training_curves.png
```

## Comparison with Other Datasets

| Dataset | Tokens | Domains | Perplexity (typical) |
|---------|--------|---------|---------------------|
| Penn Treebank | 1M | 1 (finance) | 80-100 |
| WikiText-2 | 2M | 1 (Wikipedia) | 100-120 |
| WikiText-103 | 100M | 1 (Wikipedia) | 90-110 |
| C4 | 100M | Multiple (web) | 120-150 |
| **The Pile** | **1B** | **22 (diverse)** | **150-200** |

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size
python run_pile_benchmark.py --batch-size 16

# Reduce sequence length
python run_pile_benchmark.py --n-seq 64

# Use smaller data limit
python run_pile_benchmark.py --data-limit 100000000
```

### Dataset Loading Failed
```bash
# Check internet connection
# Install datasets: pip install datasets
# Try smaller limit first
python run_pile_benchmark.py --data-limit 10000000
```

### Too Slow
```bash
# Use GPU
python run_pile_benchmark.py --device cuda

# Reduce data
python run_pile_benchmark.py --data-limit 100000000 --epochs 1

# Smaller model
python run_pile_benchmark.py --d-model 32 --n-layers 2
```

## Testing

```bash
# Run tests
pytest tests/test_pile_benchmark.py -v

# Run specific test
pytest tests/test_pile_benchmark.py::TestPileBenchmark::test_benchmark_config_creation -v
```

## Key Metrics

### Training Metrics
- Final perplexity
- Best perplexity
- Training time
- Tokens/second

### Domain Metrics
- Per-domain perplexity
- Domain difficulty ranking
- Performance variation

### Computational Metrics
- Forward/backward FLOPs
- Peak memory usage
- Model size

## Significance

**Why The Pile?**
- Most comprehensive dataset (22 domains)
- Tests true generalization capability
- Validates scalability (1B tokens)
- Real-world relevance (diverse text types)

**What It Shows**:
- Model's ability to handle diverse text
- Domain transfer capabilities
- Scaling behavior with data diversity
- Strengths and weaknesses across domains

## Next Steps

After running The Pile benchmark:

1. **Analyze domain-specific results**: Which domains are hardest?
2. **Compare to other datasets**: How does diversity affect performance?
3. **Identify improvements**: Which optimizations help most?
4. **Scale up**: Try larger models or more tokens
5. **Domain adaptation**: Fine-tune on specific domains

## References

- Paper: https://arxiv.org/abs/2101.00027
- Dataset: https://pile.eleuther.ai/
- Documentation: docs/PILE_BENCHMARK.md
