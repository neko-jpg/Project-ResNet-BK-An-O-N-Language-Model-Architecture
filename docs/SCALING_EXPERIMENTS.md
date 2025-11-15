# Model Size Scaling Experiments

This document describes the model size scaling experiments for task 9.7, which trains ResNet-BK models with different configurations to measure scaling laws.

## Overview

The scaling experiments train models with various combinations of:
- **d_model**: Model dimension ∈ {64, 128, 256, 512}
- **n_layers**: Number of layers ∈ {4, 8, 12, 16}

This results in 16 different model configurations, ranging from ~1M to ~100M parameters.

## Purpose

The experiments aim to:
1. Measure how perplexity scales with model size
2. Fit power law relationships: `perplexity = a * (num_params)^b`
3. Analyze the relationship between d_model, n_layers, and performance
4. Validate that ResNet-BK follows similar scaling laws to Transformers

## Requirements

From requirements.md:
- **Requirement 9.5**: Scale model size (d_model ∈ {64, 128, 256, 512}, n_layers ∈ {4, 8, 12, 16})
- **Requirement 9.6**: Achieve at least 100M parameters when scaling to d_model=512, n_layers=16
- **Requirement 9.20**: Validate that ResNet-BK follows similar scaling laws to Transformers

## Usage

### Quick Test (4 configurations)

```bash
python run_scaling_experiments.py --quick --epochs 3
```

This runs a quick test with:
- d_model: [64, 128]
- n_layers: [4, 8]
- Total: 4 experiments
- Time: ~30-60 minutes on GPU

### Full Experiments (16 configurations)

```bash
python run_scaling_experiments.py --epochs 5
```

This runs the full experiment suite with:
- d_model: [64, 128, 256, 512]
- n_layers: [4, 8, 12, 16]
- Total: 16 experiments
- Time: ~4-8 hours on GPU

### Custom Configuration

```bash
python run_scaling_experiments.py \
    --epochs 10 \
    --batch-size 64 \
    --device cuda \
    --output-dir my_results
```

## Implementation

### ScalingConfig

Configuration for a single scaling experiment:

```python
@dataclass
class ScalingConfig:
    d_model: int          # Model dimension
    n_layers: int         # Number of layers
    n_seq: int = 128      # Sequence length
    batch_size: int = 32  # Batch size
    epochs: int = 5       # Training epochs
    device: str = 'cuda'  # Device
```

### ScalingResults

Results from a single experiment:

```python
@dataclass
class ScalingResults:
    d_model: int
    n_layers: int
    num_params: int
    final_perplexity: float
    best_perplexity: float
    training_time: float
    total_training_flops: int
    peak_memory_mb: float
    epoch_perplexities: List[float]
```

### ScalingExperiments

Main class for running experiments:

```python
experiments = ScalingExperiments(output_dir="benchmark_results/scaling")

# Run all experiments
experiments.run_all_experiments(
    d_model_values=[64, 128, 256, 512],
    n_layers_values=[4, 8, 12, 16],
    epochs=5
)

# Analyze scaling laws
experiments.analyze_scaling_laws()

# Generate plots
experiments.plot_scaling_laws()
```

## Output Files

The experiments generate the following files in `benchmark_results/scaling/`:

### Individual Results

- `d64_l4_results.json`: Results for d_model=64, n_layers=4
- `d128_l8_results.json`: Results for d_model=128, n_layers=8
- ... (one file per configuration)

### Aggregated Results

- `all_scaling_results.json`: All experiment results in a single file
- `scaling_law.json`: Fitted scaling law parameters
- `scaling_laws.png`: Visualization plots

## Scaling Law Analysis

The experiments fit a power law relationship:

```
perplexity = a * (num_params)^b
```

Where:
- `a`: Scaling coefficient
- `b`: Scaling exponent (typically negative, ~-0.1 to -0.2)

Example output:
```
Scaling Law Fit:
  perplexity = 1234.56 * (num_params)^-0.1234
  R² = 0.9876
```

## Visualization

The experiments generate 4 plots:

1. **Perplexity vs Model Size (log-log)**: Shows power law relationship
2. **Perplexity vs d_model**: For each n_layers value
3. **Perplexity vs n_layers**: For each d_model value
4. **Training Time vs Model Size**: Computational cost analysis

## Expected Results

### Model Sizes

| d_model | n_layers | Parameters | Memory (MB) |
|---------|----------|------------|-------------|
| 64      | 4        | ~1M        | ~10         |
| 128     | 8        | ~4M        | ~40         |
| 256     | 12       | ~16M       | ~160        |
| 512     | 16       | ~100M      | ~1000       |

### Perplexity Trends

- Larger models (more parameters) achieve lower perplexity
- Increasing d_model has stronger effect than increasing n_layers
- Diminishing returns at very large model sizes

### Scaling Law

Expected power law exponent: b ≈ -0.1 to -0.2

This means:
- 10× more parameters → ~1.3-1.6× lower perplexity
- 100× more parameters → ~1.6-2.5× lower perplexity

## Comparison to Transformers

The scaling experiments validate that ResNet-BK follows similar scaling laws to Transformers:

1. **Power Law Relationship**: Both follow `perplexity ∝ (num_params)^b`
2. **Scaling Exponent**: Similar b values (~-0.1 to -0.2)
3. **Efficiency**: ResNet-BK achieves comparable perplexity with O(N) complexity

## Testing

Run tests:

```bash
pytest tests/test_scaling_experiments.py -v
```

Tests cover:
- Configuration creation
- Model creation with different sizes
- Single experiment execution
- Results saving and loading
- Scaling law fitting
- Plot generation

## Integration with Other Tasks

The scaling experiments build on:
- **Task 9.1**: FLOPs counter for measuring computational cost
- **Task 9.2**: WikiText-2 benchmark infrastructure
- **Task 1.1**: Configurable ResNet-BK model

The results feed into:
- **Task 9.14**: Scaling law analysis
- **Task 10.9**: Training GPT-2 level model

## Troubleshooting

### Out of Memory

If you encounter OOM errors with large models:

```bash
# Reduce batch size
python run_scaling_experiments.py --batch-size 16

# Use CPU for very large models
python run_scaling_experiments.py --device cpu
```

### Slow Training

For faster experiments:

```bash
# Reduce epochs
python run_scaling_experiments.py --epochs 3

# Run quick test first
python run_scaling_experiments.py --quick
```

### Missing Dependencies

Install required packages:

```bash
pip install torch datasets matplotlib numpy scipy
```

## References

- **Requirements**: Section 9 (Comprehensive Benchmarking)
- **Design**: Scaling experiments section
- **Related Tasks**: 9.1, 9.2, 9.5, 9.6, 9.14, 9.20

## Example Output

```
Model Size vs Perplexity:
Parameters      d_model    n_layers   Perplexity  
--------------------------------------------------
1,234,567       64         4          123.45      
4,567,890       128        8          67.89       
16,789,012      256        12         34.56       
98,765,432      512        16         12.34       

Scaling Law Fit:
  perplexity = 1234.56 * (num_params)^-0.1234
  R² = 0.9876
```

## Notes

- All experiments use WikiText-2 dataset for consistency
- Models use baseline configuration (no advanced optimizations) for fair comparison
- Random seed is fixed (42) for reproducibility
- Results are saved incrementally (can resume if interrupted)
