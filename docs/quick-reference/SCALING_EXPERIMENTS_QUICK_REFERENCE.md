# Scaling Experiments Quick Reference

Quick reference for running model size scaling experiments (Task 9.7).

## Quick Start

```bash
# Quick test (4 configurations, ~30-60 min)
python run_scaling_experiments.py --quick --epochs 3

# Full experiments (16 configurations, ~4-8 hours)
python run_scaling_experiments.py --epochs 5
```

## What It Does

Trains ResNet-BK models with different sizes:
- **d_model**: 64, 128, 256, 512
- **n_layers**: 4, 8, 12, 16
- **Total**: 16 configurations (1M to 100M parameters)

Measures:
- Perplexity vs model size
- Training time vs model size
- Scaling law: `perplexity = a * (num_params)^b`

## Output Files

```
benchmark_results/scaling/
├── all_scaling_results.json    # All results
├── scaling_law.json             # Fitted power law
├── scaling_laws.png             # Plots
├── d64_l4_results.json          # Individual results
├── d128_l8_results.json
└── ...
```

## Command Options

```bash
python run_scaling_experiments.py \
    --quick              # Quick test (4 configs)
    --epochs 5           # Training epochs
    --batch-size 32      # Batch size
    --device cuda        # Device (cuda/cpu)
    --output-dir DIR     # Output directory
```

## Expected Results

| d_model | n_layers | Parameters | Perplexity (approx) |
|---------|----------|------------|---------------------|
| 64      | 4        | ~1M        | ~100-150            |
| 128     | 8        | ~4M        | ~50-80              |
| 256     | 12       | ~16M       | ~25-40              |
| 512     | 16       | ~100M      | ~10-20              |

## Scaling Law

Expected: `perplexity = a * (num_params)^b` where b ≈ -0.1 to -0.2

Interpretation:
- 10× more parameters → ~1.3-1.6× lower perplexity
- 100× more parameters → ~1.6-2.5× lower perplexity

## Testing

```bash
# Run tests
pytest tests/test_scaling_experiments.py -v

# Run specific test
pytest tests/test_scaling_experiments.py::TestScalingExperiments::test_run_single_experiment_cpu -v
```

## Troubleshooting

**Out of Memory:**
```bash
python run_scaling_experiments.py --batch-size 16 --device cpu
```

**Too Slow:**
```bash
python run_scaling_experiments.py --quick --epochs 3
```

## Python API

```python
from src.benchmarks.scaling_experiments import ScalingExperiments

# Create experiments
experiments = ScalingExperiments(output_dir="my_results")

# Run all
experiments.run_all_experiments(
    d_model_values=[64, 128, 256, 512],
    n_layers_values=[4, 8, 12, 16],
    epochs=5
)

# Analyze
experiments.analyze_scaling_laws()
experiments.plot_scaling_laws()
```

## Related Files

- **Implementation**: `src/benchmarks/scaling_experiments.py`
- **Runner**: `run_scaling_experiments.py`
- **Tests**: `tests/test_scaling_experiments.py`
- **Docs**: `docs/SCALING_EXPERIMENTS.md`

## Requirements

- Task 9.7: Scale model size experiments
- Requirement 9.5: Train with d_model ∈ {64, 128, 256, 512}, n_layers ∈ {4, 8, 12, 16}
- Requirement 9.6: Achieve 100M parameters at d_model=512, n_layers=16
- Requirement 9.20: Validate scaling laws

## Time Estimates

| Configuration | Time (GPU) | Time (CPU) |
|---------------|------------|------------|
| Quick (4)     | 30-60 min  | 2-4 hours  |
| Full (16)     | 4-8 hours  | 1-2 days   |

## Dependencies

```bash
pip install torch datasets matplotlib numpy scipy
```
