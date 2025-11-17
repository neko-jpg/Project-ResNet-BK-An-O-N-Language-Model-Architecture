# Benchmarking Guide for ResNet-BK

Complete guide to benchmarking ResNet-BK against Mamba and other baselines.

---

## Table of Contents

1. [Overview](#overview)
2. [Benchmark Suite](#benchmark-suite)
3. [Running Benchmarks](#running-benchmarks)
4. [Fair Comparison Guidelines](#fair-comparison-guidelines)
5. [Interpreting Results](#interpreting-results)
6. [Generating Graphs](#generating-graphs)
7. [Statistical Validation](#statistical-validation)

---

## Overview

ResNet-BK benchmarking focuses on three key dimensions:

1. **Long-Context Stability**: Training stability on 128k-1M token sequences
2. **Quantization Robustness**: Performance after INT8/INT4 quantization
3. **Dynamic Efficiency**: FLOPs required to achieve target perplexity

### Benchmark Philosophy

- **Fair Comparison**: Identical hyperparameters, data, and hardware
- **Reproducible**: Fixed random seeds, documented environment
- **Statistical Rigor**: Multiple runs, p-values, confidence intervals
- **Comprehensive**: Multiple datasets, sequence lengths, bit widths

---

## Benchmark Suite

### 1. Long-Context Stability Benchmark

**Purpose**: Compare training stability on ultra-long sequences

**Metrics**:
- Loss curves over training steps
- Gradient norm statistics
- NaN/Inf spike counts
- Maximum stable sequence length

**Sequence Lengths**: 8k, 32k, 128k, 512k, 1M tokens

**Script**: `scripts/mamba_vs_bk_benchmark.py --benchmark longcontext`

### 2. Quantization Robustness Benchmark

**Purpose**: Compare performance after quantization

**Metrics**:
- Perplexity at different bit widths
- Model size reduction
- Inference speed
- Accuracy degradation

**Bit Widths**: FP32, FP16, INT8, INT4, INT2

**Script**: `scripts/mamba_vs_bk_benchmark.py --benchmark quantization`

### 3. Dynamic Efficiency Benchmark

**Purpose**: Compare FLOPs required for target perplexity

**Metrics**:
- Perplexity vs FLOPs curves
- Average FLOPs per token
- Pareto frontier analysis
- Efficiency ratio

**FLOPs Budgets**: 1B, 2B, 5B, 10B FLOPs

**Script**: `scripts/mamba_vs_bk_benchmark.py --benchmark efficiency`

### 4. Multi-Dataset Evaluation

**Purpose**: Verify generalization across datasets

**Datasets**:
- WikiText-2 (small, 2M tokens)
- WikiText-103 (medium, 103M tokens)
- Penn Treebank (small, 1M tokens)
- C4 (large, 365M tokens)
- The Pile (very large, 825GB)

**Script**: `scripts/mamba_vs_bk_benchmark.py --benchmark multidataset`

---

## Running Benchmarks

### Quick Start

Run all benchmarks with default settings:

```bash
python scripts/mamba_vs_bk_benchmark.py --all
```

### Individual Benchmarks

#### Long-Context Stability

```bash
python scripts/mamba_vs_bk_benchmark.py \
  --benchmark longcontext \
  --seq_lengths 8192 32768 131072 \
  --num_seeds 5 \
  --output results/longcontext_results.json
```

**Expected runtime**: ~6 hours on 4× T4 GPUs

**Expected results**:
- ResNet-BK: Stable at all sequence lengths
- Mamba: Diverges at 32k tokens

#### Quantization Robustness

```bash
python scripts/mamba_vs_bk_benchmark.py \
  --benchmark quantization \
  --bit_widths 32 16 8 4 2 \
  --num_seeds 5 \
  --output results/quantization_results.json
```

**Expected runtime**: ~4 hours on 4× T4 GPUs

**Expected results**:
- ResNet-BK INT4: PPL ~45
- Mamba INT4: PPL ~180

#### Dynamic Efficiency

```bash
python scripts/mamba_vs_bk_benchmark.py \
  --benchmark efficiency \
  --flops_budgets 1e9 2e9 5e9 1e10 \
  --num_seeds 5 \
  --output results/efficiency_results.json
```

**Expected runtime**: ~8 hours on 4× T4 GPUs

**Expected results**:
- ResNet-BK: 2× fewer FLOPs at equal PPL
- ResNet-BK: 30% lower PPL at equal FLOPs

### Advanced Options

```bash
python scripts/mamba_vs_bk_benchmark.py \
  --benchmark longcontext \
  --model_a resnetbk \
  --model_b mamba \
  --seq_lengths 8192 32768 131072 524288 1048576 \
  --batch_size 1 \
  --num_epochs 3 \
  --num_seeds 5 \
  --device cuda \
  --mixed_precision true \
  --gradient_checkpointing true \
  --output results/longcontext_full.json \
  --save_checkpoints true \
  --checkpoint_dir checkpoints/benchmark/
```

---

## Fair Comparison Guidelines

### Hyperparameter Matching

**Critical**: Use identical hyperparameters for both models

```yaml
# Shared hyperparameters
learning_rate: 1e-3
batch_size: 8
optimizer: AdamW
warmup_steps: 100
weight_decay: 0.01
gradient_clip_norm: 1.0

# Model size matching
d_model: 256
n_layers: 6
total_params: ~4M  # Match within 10%
```

### Data Preprocessing

**Critical**: Use identical tokenization and preprocessing

```python
# Use same tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Same vocabulary size
vocab_size = len(tokenizer)

# Same sequence length
seq_len = 512

# Same random seed for data shuffling
torch.manual_seed(42)
np.random.seed(42)
```

### Hardware Consistency

**Critical**: Run on same hardware

```python
# Check GPU
assert torch.cuda.is_available()
assert torch.cuda.get_device_name(0) == "Tesla T4"

# Set deterministic mode
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

### FLOPs Counting

**Critical**: Count all operations fairly

```python
from src.benchmarks import FairFLOPsCounter

# Count all operations
counter = FairFLOPsCounter(
    count_activations=True,
    count_normalization=True,
    count_routing=True,
    count_state_updates=True  # For Mamba
)

flops = counter.count(model, input_ids)
```

### Memory Measurement

**Critical**: Include all memory components

```python
from src.benchmarks import FairMemoryMeasurement

# Measure all memory
memory = FairMemoryMeasurement(
    include_activations=True,
    include_gradients=True,
    include_optimizer_states=True,
    include_buffers=True
)

total_memory = memory.measure(model, optimizer, batch)
```

---

## Interpreting Results

### Long-Context Stability Results

**Example output**:
```json
{
  "seq_len_8192": {
    "resnetbk": {
      "final_loss": 7.02,
      "gradient_norm_mean": 0.45,
      "gradient_norm_std": 0.12,
      "nan_count": 0,
      "diverged": false
    },
    "mamba": {
      "final_loss": 7.15,
      "gradient_norm_mean": 0.52,
      "gradient_norm_std": 0.18,
      "nan_count": 0,
      "diverged": false
    }
  },
  "seq_len_32768": {
    "resnetbk": {
      "final_loss": 7.08,
      "gradient_norm_mean": 0.48,
      "gradient_norm_std": 0.14,
      "nan_count": 0,
      "diverged": false
    },
    "mamba": {
      "final_loss": null,
      "gradient_norm_mean": null,
      "gradient_norm_std": null,
      "nan_count": 47,
      "diverged": true
    }
  }
}
```

**Interpretation**:
- ResNet-BK remains stable at 32k tokens
- Mamba diverges (47 NaN spikes)
- ResNet-BK has lower gradient variance

### Quantization Robustness Results

**Example output**:
```json
{
  "fp32": {
    "resnetbk": {"ppl": 30.5, "size_mb": 16.6},
    "mamba": {"ppl": 31.2, "size_mb": 16.8}
  },
  "int8": {
    "resnetbk": {"ppl": 32.1, "size_mb": 4.2},
    "mamba": {"ppl": 38.7, "size_mb": 4.2}
  },
  "int4": {
    "resnetbk": {"ppl": 45.3, "size_mb": 2.1},
    "mamba": {"ppl": 182.4, "size_mb": 2.1}
  }
}
```

**Interpretation**:
- ResNet-BK: 5.3% degradation at INT8
- Mamba: 24.0% degradation at INT8
- ResNet-BK: 4× better at INT4

### Dynamic Efficiency Results

**Example output**:
```json
{
  "flops_1e9": {
    "resnetbk": {"ppl": 35.2, "avg_flops_per_token": 9.8e8},
    "mamba": {"ppl": 42.1, "avg_flops_per_token": 9.9e8}
  },
  "flops_2e9": {
    "resnetbk": {"ppl": 30.5, "avg_flops_per_token": 1.95e9},
    "mamba": {"ppl": 35.8, "avg_flops_per_token": 1.98e9}
  }
}
```

**Interpretation**:
- At 1B FLOPs: ResNet-BK 16% lower PPL
- At 2B FLOPs: ResNet-BK 15% lower PPL
- ResNet-BK dominates Pareto frontier

---

## Generating Graphs

### Long-Context Stability Graph

```python
from scripts import generate_stability_graph

generate_stability_graph.main(
    results_file="results/longcontext_results.json",
    output="results/stability_graph.pdf",
    seq_lengths=[8192, 32768, 131072, 524288, 1048576],
    annotate_divergence=True,
    show_confidence_intervals=True
)
```

**Output**: Publication-quality graph showing:
- Loss curves for each sequence length
- Mamba divergence points
- ResNet-BK stable regions
- Confidence intervals (±std over 5 runs)

### Quantization Robustness Graph

```python
from scripts import generate_quantization_graph

generate_quantization_graph.main(
    results_file="results/quantization_results.json",
    output="results/quantization_graph.pdf",
    bit_widths=[32, 16, 8, 4, 2],
    annotate_threshold=True,  # PPL < 100 threshold
    show_confidence_intervals=True
)
```

**Output**: Graph showing:
- PPL vs bit width for both models
- Practical deployment threshold (PPL < 100)
- ResNet-BK maintaining low PPL at INT4
- Mamba degrading significantly

### Dynamic Efficiency Graph

```python
from scripts import generate_efficiency_graph

generate_efficiency_graph.main(
    results_file="results/efficiency_results.json",
    output="results/efficiency_graph.pdf",
    flops_budgets=[1e9, 2e9, 5e9, 1e10],
    annotate_pareto=True,
    show_confidence_intervals=True
)
```

**Output**: Graph showing:
- PPL vs FLOPs for both models
- Pareto frontier
- ResNet-BK achieving lower PPL at every FLOPs budget
- 2× efficiency advantage

---

## Statistical Validation

### Paired T-Test

```python
from scipy import stats

# Compare perplexities across 5 runs
resnetbk_ppls = [30.2, 30.5, 30.1, 30.8, 30.3]
mamba_ppls = [35.1, 35.8, 35.2, 36.0, 35.5]

# Perform paired t-test
t_stat, p_value = stats.ttest_rel(resnetbk_ppls, mamba_ppls)

print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.6f}")

if p_value < 0.001:
    print("Difference is statistically significant (p < 0.001)")
```

### Confidence Intervals

```python
import numpy as np

# Compute 95% confidence interval
mean = np.mean(resnetbk_ppls)
std = np.std(resnetbk_ppls)
n = len(resnetbk_ppls)

ci_lower = mean - 1.96 * std / np.sqrt(n)
ci_upper = mean + 1.96 * std / np.sqrt(n)

print(f"Mean: {mean:.2f}")
print(f"95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]")
```

### Bonferroni Correction

```python
# Multiple comparisons correction
num_comparisons = 15  # Number of metrics compared
alpha = 0.05
corrected_alpha = alpha / num_comparisons

print(f"Corrected significance level: {corrected_alpha:.6f}")

if p_value < corrected_alpha:
    print("Significant after Bonferroni correction")
```

### Effect Size (Cohen's d)

```python
# Compute effect size
mean_diff = np.mean(resnetbk_ppls) - np.mean(mamba_ppls)
pooled_std = np.sqrt((np.var(resnetbk_ppls) + np.var(mamba_ppls)) / 2)
cohens_d = mean_diff / pooled_std

print(f"Cohen's d: {cohens_d:.4f}")

if abs(cohens_d) > 0.8:
    print("Large effect size")
elif abs(cohens_d) > 0.5:
    print("Medium effect size")
else:
    print("Small effect size")
```

---

## Benchmark Checklist

Before publishing benchmark results, verify:

- [ ] Identical hyperparameters for both models
- [ ] Same tokenizer and vocabulary
- [ ] Same random seeds
- [ ] Same hardware (GPU model, CUDA version)
- [ ] Fair FLOPs counting (all operations)
- [ ] Fair memory measurement (all components)
- [ ] Multiple runs (≥5 seeds)
- [ ] Statistical significance (p < 0.001)
- [ ] Confidence intervals reported
- [ ] Bonferroni correction applied
- [ ] Effect sizes computed
- [ ] Results reproducible (code + data + checkpoints)

---

## Troubleshooting

### Issue: Results not reproducible

**Solution**: Set all random seeds

```python
import torch
import numpy as np
import random

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
```

### Issue: Mamba runs out of memory

**Solution**: Use same memory optimizations

```python
# Enable same optimizations for both models
config = {
    'gradient_checkpointing': True,
    'mixed_precision': True,
    'cpu_offloading': True
}
```

### Issue: FLOPs counts don't match

**Solution**: Verify counter includes all operations

```python
# Check what's being counted
counter = FairFLOPsCounter(verbose=True)
flops_breakdown = counter.count_detailed(model, input_ids)

print("FLOPs breakdown:")
for op, flops in flops_breakdown.items():
    print(f"  {op}: {flops:.2e}")
```

---

## Best Practices

1. **Run Multiple Seeds**: Use ≥5 seeds for statistical validity
2. **Report Confidence Intervals**: Always include ±std
3. **Test Statistical Significance**: Use paired t-tests
4. **Apply Corrections**: Use Bonferroni for multiple comparisons
5. **Document Everything**: Hardware, software, hyperparameters
6. **Share Artifacts**: Code, data, checkpoints, results
7. **Visualize Results**: Publication-quality graphs
8. **Be Fair**: Identical conditions for all models

---

## Resources

- **Benchmark Scripts**: `scripts/mamba_vs_bk_benchmark.py`
- **Graph Generation**: `scripts/generate_*_graph.py`
- **Fair Comparison**: `src/benchmarks/fair_comparison.py`
- **Statistical Tests**: `src/benchmarks/statistical_validation.py`

For more information:
- [TUTORIAL.md](TUTORIAL.md) - Training guide
- [API_REFERENCE.md](API_REFERENCE.md) - API documentation
- [FAQ.md](FAQ.md) - Common questions
