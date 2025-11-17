# ACT Hyperparameter Tuning - Quick Reference

## Quick Start

### Run Tuning (Command Line)
```bash
python examples/tune_act_hyperparameters.py
```

### Run Tuning (Python)
```python
from src.training.act_hyperparameter_tuner import run_act_hyperparameter_tuning

results = run_act_hyperparameter_tuning(
    data_path='data',
    batch_size=32,
    num_epochs=3,
    output_dir='results/act_tuning'
)

print(f"Best: threshold={results['best_config']['threshold']}, "
      f"lambda={results['best_config']['lambda']}")
```

## Recommended Configurations

### Production (Balanced)
```python
act_threshold = 0.95
act_lambda = 0.01
# ~3.5 layers, <5% perplexity increase, 1.14x speedup
```

### Edge Devices (High Speed)
```python
act_threshold = 0.8
act_lambda = 0.05
# ~2.5 layers, 10-15% perplexity increase, 1.6x speedup
```

### Research (High Accuracy)
```python
act_threshold = 0.99
act_lambda = 0.001
# ~3.9 layers, minimal perplexity increase
```

## Custom Search Space

```bash
python examples/tune_act_hyperparameters.py \
    --thresholds 0.5 0.7 0.9 0.95 0.99 \
    --lambdas 0.001 0.01 0.1 \
    --epochs 5 \
    --score-metric balanced
```

## Output Files

- `results/act_tuning_results.json` - Detailed results
- `results/act_tuning_heatmap.png` - Visual comparison

## Key Metrics

- **Validation Perplexity**: Lower is better (quality)
- **Avg Layers Executed**: Lower is better (speed)
- **Speedup Potential**: n_layers / avg_layers

## Files

- Implementation: `src/training/act_hyperparameter_tuner.py`
- CLI: `examples/tune_act_hyperparameters.py`
- Tests: `tests/test_act_hyperparameter_tuner.py`
- Docs: `docs/ACT_HYPERPARAMETER_TUNING.md`

## Next: Task 7.3 - Multi-Scale Processing
