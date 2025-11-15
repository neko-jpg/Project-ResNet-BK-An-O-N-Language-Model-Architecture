# Penn Treebank Benchmark

## Overview

The Penn Treebank benchmark evaluates ResNet-BK on a different domain (financial news from Wall Street Journal) compared to WikiText (Wikipedia articles). This tests the model's domain generalization capability and validates performance across diverse text types.

**Task**: 9.4 - Benchmark on Penn Treebank  
**Requirement**: 9.2 - Evaluate on different domain

## Dataset Characteristics

### Penn Treebank
- **Domain**: Financial news (Wall Street Journal)
- **Size**: ~1 million tokens
- **Vocabulary**: ~10,000 words (smaller than WikiText)
- **Style**: Formal, structured financial reporting
- **Sentences**: Well-formed, grammatically correct

### Comparison to WikiText
| Dataset | Domain | Tokens | Vocab | Style |
|---------|--------|--------|-------|-------|
| Penn Treebank | Financial news | ~1M | ~10K | Formal, structured |
| WikiText-2 | Wikipedia | ~2M | ~30K | Encyclopedic, varied |
| WikiText-103 | Wikipedia | ~100M | ~260K | Encyclopedic, varied |

## Implementation

### Files
- `src/benchmarks/penn_treebank_benchmark.py` - Main benchmark implementation
- `tests/test_penn_treebank_benchmark.py` - Unit tests
- `run_penn_treebank_benchmark.py` - Standalone script
- `docs/PENN_TREEBANK_BENCHMARK.md` - This documentation

### Key Components

#### 1. Data Loading
```python
def load_penn_treebank_data(batch_size, n_seq, data_limit=None, vocab_size_limit=10000):
    """
    Load Penn Treebank dataset from Hugging Face.
    
    Returns:
        train_data: (seq_len, batch_size) LongTensor
        vocab: dict with stoi, itos, vocab_size
        get_batch: function to get training batches
    """
```

Features:
- Loads from `ptb_text_only` dataset
- Builds vocabulary with special tokens (`<unk>`, `<eos>`)
- Adds EOS token after each sentence
- Supports vocabulary size limiting
- Supports data limiting for testing

#### 2. Benchmark Configuration
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
    data_limit: Optional[int] = None
    
    # Optimization flags
    use_analytic_gradient: bool = True
    use_mixed_precision: bool = False
    use_act: bool = False
    use_multi_scale: bool = False
    use_sparse_bk: bool = False
    # ... more flags
```

#### 3. Benchmark Results
```python
@dataclass
class BenchmarkResults:
    model_name: str
    dataset_name: str
    final_loss: float
    final_perplexity: float
    best_perplexity: float
    training_time: float
    total_tokens: int
    vocab_size: int
    forward_flops: int
    backward_flops: int
    total_training_flops: int
    peak_memory_mb: float
    model_size_mb: float
    epoch_losses: List[float]
    epoch_perplexities: List[float]
    epoch_times: List[float]
```

#### 4. Cross-Dataset Comparison
```python
def compare_to_other_datasets(
    self,
    wikitext2_results_path: str = None,
    wikitext103_results_path: str = None
):
    """
    Compare Penn Treebank results to WikiText-2 and WikiText-103.
    
    Analyzes:
    - Dataset size differences
    - Vocabulary size differences
    - Perplexity differences (domain shift)
    - Training time scaling
    """
```

## Usage

### Basic Usage
```bash
# Run full benchmark (all models)
python run_penn_treebank_benchmark.py

# Run on CPU
python run_penn_treebank_benchmark.py --device cpu

# Run with fewer epochs (faster)
python run_penn_treebank_benchmark.py --epochs 3

# Skip baselines (only run optimized model)
python run_penn_treebank_benchmark.py --skip-transformer --skip-baseline
```

### Advanced Usage
```bash
# Custom configuration
python run_penn_treebank_benchmark.py \
    --device cuda \
    --epochs 10 \
    --batch-size 64 \
    --d-model 128 \
    --n-layers 6 \
    --n-seq 256 \
    --lr 5e-4 \
    --output-dir my_results

# Test with limited data
python run_penn_treebank_benchmark.py --data-limit 10000
```

### Programmatic Usage
```python
from src.benchmarks.penn_treebank_benchmark import (
    PennTreebankBenchmark,
    BenchmarkConfig
)

# Create benchmark
benchmark = PennTreebankBenchmark(output_dir="results")

# Configure model
config = BenchmarkConfig(
    model_name='resnet_bk_full',
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
    use_multi_scale=True,
    use_sparse_bk=True,
)

# Run benchmark
results = benchmark.run_benchmark(config)

# Compare to other datasets
benchmark.compare_to_other_datasets(
    wikitext2_results_path="results/wikitext2/resnet_bk_full_results.json",
    wikitext103_results_path="results/wikitext103/resnet_bk_full_results.json"
)

# Plot training curves
benchmark.plot_training_curves()
```

## Expected Results

### Perplexity Targets
Based on requirement 9.2, ResNet-BK should achieve perplexity within 30% of Transformer baseline:

| Model | Expected PPL | Target |
|-------|-------------|--------|
| Transformer Baseline | ~100-150 | Baseline |
| ResNet-BK Baseline | ~120-180 | Within 30% |
| ResNet-BK Full | ~110-160 | Within 30% |

### Domain Comparison
Penn Treebank typically shows:
- **Lower perplexity** than WikiText (more structured, formal language)
- **Smaller vocabulary** (specialized financial domain)
- **Faster training** (smaller dataset)

### Performance Metrics
- **Training Time**: ~5-10 minutes for 5 epochs (GPU)
- **Memory Usage**: ~500-1000 MB peak (depends on batch size)
- **FLOPs**: ~1-5 TFLOPs total training

## Output Files

### Results Directory Structure
```
benchmark_results/penn_treebank/
├── transformer_baseline_penn_treebank_results.json
├── resnet_bk_baseline_penn_treebank_results.json
├── resnet_bk_full_penn_treebank_results.json
├── cross_dataset_comparison.json
└── penn_treebank_training_curves.png
```

### JSON Results Format
```json
{
  "model_name": "resnet_bk_full",
  "dataset_name": "penn-treebank",
  "final_loss": 4.5,
  "final_perplexity": 90.0,
  "best_perplexity": 85.0,
  "training_time": 300.0,
  "total_tokens": 929589,
  "vocab_size": 10000,
  "forward_flops": 1500000,
  "backward_flops": 3000000,
  "total_training_flops": 2250000000,
  "peak_memory_mb": 800.0,
  "model_size_mb": 12.5,
  "epoch_losses": [5.2, 4.8, 4.6, 4.5, 4.5],
  "epoch_perplexities": [181.3, 121.5, 99.5, 90.0, 90.0],
  "epoch_times": [60.0, 60.0, 60.0, 60.0, 60.0]
}
```

## Domain Analysis

### Why Penn Treebank?
1. **Different Domain**: Financial news vs. Wikipedia
2. **Structured Text**: Well-formed sentences, formal language
3. **Smaller Vocabulary**: Specialized domain terminology
4. **Standard Benchmark**: Widely used in NLP research

### Expected Domain Effects
- **Lower Perplexity**: More predictable, structured language
- **Better Generalization**: Tests if model overfits to Wikipedia style
- **Vocabulary Mismatch**: Different word distributions
- **Syntax Patterns**: More formal grammatical structures

### Interpretation Guidelines
- **Lower PPL on PTB**: Model handles structured text well
- **Higher PPL on PTB**: Model may be overfitting to Wikipedia
- **Similar PPL**: Good domain generalization
- **Large PPL difference**: Domain-specific adaptation needed

## Testing

### Run Unit Tests
```bash
# Run all tests
pytest tests/test_penn_treebank_benchmark.py -v

# Run specific test
pytest tests/test_penn_treebank_benchmark.py::TestPennTreebankDataLoader -v

# Run with coverage
pytest tests/test_penn_treebank_benchmark.py --cov=src.benchmarks.penn_treebank_benchmark
```

### Test Coverage
- Data loading and preprocessing
- Vocabulary building
- Model creation (Transformer and ResNet-BK)
- Benchmark configuration
- Results serialization
- Cross-dataset comparison

## Troubleshooting

### Dataset Not Available
```python
# Error: Failed to load Penn Treebank
# Solution: Check internet connection, try alternative dataset name
dataset = load_dataset("ptb-text-only/ptb_text_only")
```

### Out of Memory
```bash
# Reduce batch size
python run_penn_treebank_benchmark.py --batch-size 16

# Reduce sequence length
python run_penn_treebank_benchmark.py --n-seq 64

# Use CPU
python run_penn_treebank_benchmark.py --device cpu
```

### Slow Training
```bash
# Reduce epochs
python run_penn_treebank_benchmark.py --epochs 3

# Limit data
python run_penn_treebank_benchmark.py --data-limit 100000

# Skip baselines
python run_penn_treebank_benchmark.py --skip-transformer --skip-baseline
```

## References

- Penn Treebank: Marcus et al. (1993) "Building a Large Annotated Corpus of English"
- Dataset: https://huggingface.co/datasets/ptb_text_only
- Requirement 9.2: Evaluate on different domain
- Task 9.4: Benchmark on Penn Treebank

## Next Steps

After completing Penn Treebank benchmark:
1. **Task 9.5**: Benchmark on C4 (100M tokens)
2. **Task 9.6**: Benchmark on The Pile (1B tokens)
3. **Task 9.7**: Scale model size experiments
4. **Task 9.8**: Scale sequence length experiments

## Summary

The Penn Treebank benchmark validates ResNet-BK's domain generalization by testing on financial news (different from Wikipedia training). Key metrics:
- Perplexity within 30% of baseline
- Cross-dataset comparison
- Domain-specific performance analysis
- Training efficiency on smaller, structured dataset
