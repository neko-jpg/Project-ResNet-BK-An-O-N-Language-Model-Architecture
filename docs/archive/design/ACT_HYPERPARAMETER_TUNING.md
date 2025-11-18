# ACT Hyperparameter Tuning Guide

This guide explains how to tune Adaptive Computation Time (ACT) hyperparameters to find the optimal balance between model performance and computational efficiency.

## Overview

ACT has two main hyperparameters:

1. **`act_threshold`**: Halting probability threshold (0.0 to 1.0)
   - Controls when tokens stop processing
   - Lower values → fewer layers executed → faster inference
   - Higher values → more layers executed → better accuracy

2. **`act_lambda`**: Ponder cost weight (typically 0.001 to 0.1)
   - Weight of computational cost in the loss function
   - Higher values → stronger penalty on computation → encourages early halting
   - Lower values → weaker penalty → allows more computation

## Quick Start

### Basic Usage

```python
from src.training.act_hyperparameter_tuner import run_act_hyperparameter_tuning

# Run tuning with default settings
results = run_act_hyperparameter_tuning(
    data_path='data',
    batch_size=32,
    num_epochs=3,
    output_dir='results/act_tuning'
)

print(f"Best threshold: {results['best_config']['threshold']}")
print(f"Best lambda: {results['best_config']['lambda']}")
```

### Command Line

```bash
# Basic tuning
python examples/tune_act_hyperparameters.py

# Custom search space
python examples/tune_act_hyperparameters.py \
    --thresholds 0.5 0.7 0.9 0.95 0.99 \
    --lambdas 0.001 0.01 0.1 \
    --epochs 5 \
    --batch-size 64

# Optimize for perplexity only
python examples/tune_act_hyperparameters.py --score-metric perplexity

# Optimize for computational efficiency
python examples/tune_act_hyperparameters.py --score-metric layers
```

## Hyperparameter Effects

### Threshold (`act_threshold`)

| Threshold | Avg Layers | Speedup | Perplexity Impact |
|-----------|------------|---------|-------------------|
| 0.5       | ~2.0       | 2.0x    | +15-20%          |
| 0.8       | ~2.5       | 1.6x    | +8-12%           |
| 0.9       | ~3.0       | 1.3x    | +5-8%            |
| 0.95      | ~3.5       | 1.14x   | +2-5%            |
| 0.99      | ~3.9       | 1.03x   | +0-2%            |

**Recommendation**: Start with 0.95 for good balance, adjust based on requirements.

### Lambda (`act_lambda`)

| Lambda | Effect on Training | Effect on Inference |
|--------|-------------------|---------------------|
| 0.001  | Minimal penalty, allows full computation | More layers used |
| 0.01   | Moderate penalty, balanced | Balanced layers |
| 0.1    | Strong penalty, encourages early halting | Fewer layers used |

**Recommendation**: Start with 0.01, increase if you need more speedup.

## Grid Search

The tuner performs exhaustive grid search over specified hyperparameter ranges:

```python
from src.training.act_hyperparameter_tuner import ACTHyperparameterTuner
from src.utils.data_utils import get_wikitext2_dataloaders

# Load data
train_loader, val_loader, _, vocab_size = get_wikitext2_dataloaders(
    data_path='data',
    batch_size=32,
    n_seq=128
)

# Create tuner
tuner = ACTHyperparameterTuner(
    vocab_size=vocab_size,
    d_model=64,
    n_layers=4,
    n_seq=128
)

# Run grid search
results = tuner.grid_search(
    train_loader=train_loader,
    val_loader=val_loader,
    threshold_values=[0.5, 0.8, 0.9, 0.95, 0.99],
    lambda_values=[0.001, 0.005, 0.01, 0.05, 0.1],
    num_epochs=3,
    score_metric='balanced'  # 'perplexity', 'layers', or 'balanced'
)

# Save results
tuner.save_results('results/act_tuning_results.json')
tuner.plot_results(save_path='results/act_tuning_heatmap.png')
```

## Score Metrics

The tuner supports three scoring strategies:

### 1. Perplexity (`score_metric='perplexity'`)
- Optimizes for lowest validation perplexity
- Best for accuracy-critical applications
- May result in higher computational cost

### 2. Layers (`score_metric='layers'`)
- Optimizes for fewest average layers executed
- Best for latency-critical applications
- May sacrifice some accuracy

### 3. Balanced (`score_metric='balanced'`)
- Balances perplexity and computational cost
- Score = perplexity + 10 × avg_layers
- **Recommended for most use cases**

## Interpreting Results

### Output Files

1. **`act_tuning_results.json`**: Detailed results for all configurations
   ```json
   {
     "best_config": {
       "threshold": 0.95,
       "lambda": 0.01
     },
     "best_score": 45.2,
     "all_results": [...]
   }
   ```

2. **`act_tuning_heatmap.png`**: Visual comparison of configurations
   - Left: Validation perplexity heatmap
   - Middle: Average layers executed heatmap
   - Right: Combined score heatmap
   - White star marks best configuration

### Key Metrics

For each configuration, the tuner reports:

- **Validation Perplexity**: Lower is better (quality metric)
- **Avg Layers Executed**: Lower is better (efficiency metric)
- **Training Time**: Time to train for specified epochs
- **Convergence Speed**: Perplexity improvement rate
- **Speedup Potential**: n_layers / avg_layers_executed

## Best Practices

### 1. Start with Coarse Search
```python
# Coarse search
results = tuner.grid_search(
    threshold_values=[0.5, 0.8, 0.95],
    lambda_values=[0.001, 0.01, 0.1],
    num_epochs=2
)
```

### 2. Refine Around Best Configuration
```python
# Fine-grained search around best config
best_threshold = results['best_config']['threshold']
best_lambda = results['best_config']['lambda']

results_fine = tuner.grid_search(
    threshold_values=[best_threshold - 0.05, best_threshold, best_threshold + 0.05],
    lambda_values=[best_lambda * 0.5, best_lambda, best_lambda * 2],
    num_epochs=5
)
```

### 3. Validate on Test Set
```python
# After finding best config, validate on test set
model = tuner.create_model(
    act_threshold=results['best_config']['threshold'],
    act_lambda=results['best_config']['lambda']
)

# Train on full dataset
# ... training code ...

# Evaluate on test set
test_perplexity = tuner.evaluate(model, test_loader)
```

## Common Configurations

### High Accuracy (Research)
```python
act_threshold = 0.99
act_lambda = 0.001
# Expected: ~3.9 layers, minimal perplexity increase
```

### Balanced (Production)
```python
act_threshold = 0.95
act_lambda = 0.01
# Expected: ~3.5 layers, <5% perplexity increase, 1.14x speedup
```

### High Speed (Edge Devices)
```python
act_threshold = 0.8
act_lambda = 0.05
# Expected: ~2.5 layers, 10-15% perplexity increase, 1.6x speedup
```

### Maximum Speed (Real-time)
```python
act_threshold = 0.5
act_lambda = 0.1
# Expected: ~2.0 layers, 15-20% perplexity increase, 2.0x speedup
```

## Troubleshooting

### Issue: All configurations have similar performance
**Solution**: Increase lambda range or decrease threshold range

### Issue: Training is too slow
**Solution**: 
- Reduce `num_epochs` (use 1-2 for initial search)
- Reduce search space size
- Use smaller model for tuning

### Issue: High variance in results
**Solution**:
- Run multiple seeds and average
- Increase `num_epochs` for more stable estimates
- Use more training/validation batches

### Issue: Best config uses all layers (no speedup)
**Solution**:
- Increase lambda values (try 0.05, 0.1, 0.2)
- Decrease threshold values (try 0.5, 0.7, 0.8)
- Check if model is undertrained (increase epochs)

## Advanced Usage

### Custom Scoring Function
```python
def custom_score(result):
    """Custom scoring: prioritize speedup over perplexity."""
    perplexity = result['final_val_perplexity']
    layers = result['avg_layers_executed']
    
    # Heavily weight computational savings
    return perplexity + 50.0 * layers

# Modify tuner to use custom scoring
# (requires modifying grid_search method)
```

### Multi-Objective Optimization
```python
# Find Pareto frontier of perplexity vs. layers
import matplotlib.pyplot as plt

perplexities = [r['final_val_perplexity'] for r in results['all_results']]
layers = [r['avg_layers_executed'] for r in results['all_results']]

plt.scatter(layers, perplexities)
plt.xlabel('Avg Layers Executed')
plt.ylabel('Validation Perplexity')
plt.title('Perplexity vs. Computational Cost')
plt.savefig('pareto_frontier.png')
```

### Transfer Tuning Results
```python
# Tune on small model, apply to large model
small_tuner = ACTHyperparameterTuner(vocab_size, d_model=64, n_layers=4)
small_results = small_tuner.grid_search(...)

# Use best config for large model
large_model = ACTLanguageModel(
    vocab_size=vocab_size,
    d_model=256,
    n_layers=12,
    act_threshold=small_results['best_config']['threshold'],
    act_lambda=small_results['best_config']['lambda']
)
```

## References

- Original ACT paper: [Adaptive Computation Time for Recurrent Neural Networks](https://arxiv.org/abs/1603.08983)
- Universal Transformers: [Universal Transformers](https://arxiv.org/abs/1807.03819)
- PonderNet: [PonderNet: Learning to Ponder](https://arxiv.org/abs/2107.05407)

## Next Steps

After tuning ACT hyperparameters:

1. **Implement multi-scale processing** (Task 7.3)
2. **Add learned sparsity** (Task 7.4)
3. **Combine with other optimizations** (Steps 4-5)
4. **Deploy optimized model** (Step 10)

See `docs/ACT_IMPLEMENTATION.md` for implementation details.
