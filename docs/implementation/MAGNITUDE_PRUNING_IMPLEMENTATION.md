# Magnitude-Based Pruning Implementation (Task 5.5)

## Overview

This document describes the implementation of Task 5.5: Magnitude-based pruning with iterative retraining for output_proj and fc layers.

## Implementation Details

### Core Components

#### 1. MagnitudePruner (`src/models/pruned_moe.py`)

Enhanced magnitude pruner with:
- **Threshold-based pruning**: Remove weights with |w| < threshold
- **Sparsity-based pruning**: Prune to achieve target sparsity level
- **Mask persistence**: Store and apply masks to maintain sparsity during training
- **Layer-wise sparsity tracking**: Monitor sparsity for each layer

Key methods:
```python
pruner = MagnitudePruner(threshold=0.01, target_sparsity=0.5)

# Prune single layer
pruner.prune_layer(layer, sparsity=0.5)

# Prune entire model
pruner.prune_model(model, layer_names=['output_proj', 'fc'], sparsity=0.5)

# Apply masks after training step
pruner.apply_masks(model)

# Get sparsity statistics
sparsity = pruner.get_model_sparsity(model)
```

#### 2. IterativeMagnitudePruner (`src/models/pruned_moe.py`)

Implements gradual sparsity increase over multiple prune-retrain cycles:
- **Exponential sparsity schedule**: Gradually increase from initial to final sparsity
- **Layer filtering**: Target specific layers (e.g., output_proj, fc)
- **History tracking**: Record pruning statistics for each iteration

Key methods:
```python
pruner = IterativeMagnitudePruner(
    initial_sparsity=0.2,
    final_sparsity=0.8,
    num_iterations=5,
    prune_layers=['output_proj', 'fc']
)

# Execute one pruning step
stats = pruner.prune_step(model, verbose=True)

# Apply masks during training
pruner.train_step_with_mask(model)

# Get summary
summary = pruner.get_pruning_summary()
```

#### 3. IterativePruningTrainer (`src/training/iterative_pruning_trainer.py`)

Complete training workflow for iterative pruning:
- **Prune-retrain cycles**: Automated iteration through pruning and retraining
- **Evaluation tracking**: Monitor perplexity before/after pruning and retraining
- **Checkpoint saving**: Save model at each iteration
- **Comprehensive metrics**: Track compression ratio, sparsity, accuracy

Key methods:
```python
trainer = create_iterative_pruning_trainer(
    model=model,
    initial_sparsity=0.2,
    final_sparsity=0.8,
    num_iterations=5,
    target_layers=['output_proj', 'fc'],
    device='cuda'
)

# Run complete workflow
results = trainer.run_iterative_pruning(
    train_loader=train_loader,
    val_loader=val_loader,
    retrain_epochs=3,
    learning_rate=1e-4,
    save_dir='checkpoints/pruning'
)
```

### Integration with Compression Pipeline

The iterative magnitude pruning is integrated into Stage 2 of the compression pipeline:

```python
from src.training.compression_pipeline import CompressionPipeline

pipeline = CompressionPipeline(model, target_compression=100.0)

results = pipeline.run_pipeline(
    train_loader=train_loader,
    val_loader=val_loader,
    qat_epochs=3,
    pruning_epochs=3,  # Now uses iterative pruning
    distillation_epochs=5
)
```

## Usage Examples

### Example 1: Basic Magnitude Pruning

```python
from src.models.pruned_moe import MagnitudePruner

# Create pruner
pruner = MagnitudePruner(threshold=0.01)

# Prune model to 50% sparsity
stats = pruner.prune_model(model, sparsity=0.5, verbose=True)

# Check sparsity
sparsity = pruner.get_model_sparsity(model)
print(f"Average sparsity: {sum(sparsity.values()) / len(sparsity):.2%}")
```

### Example 2: Iterative Pruning with Retraining

```python
from src.training.iterative_pruning_trainer import create_iterative_pruning_trainer

# Create trainer targeting output_proj and fc layers
trainer = create_iterative_pruning_trainer(
    model=model,
    initial_sparsity=0.2,
    final_sparsity=0.7,
    num_iterations=3,
    target_layers=['output_proj', 'fc'],
    device='cuda'
)

# Run iterative pruning
results = trainer.run_iterative_pruning(
    train_loader=train_loader,
    val_loader=val_loader,
    retrain_epochs=2,
    learning_rate=1e-4
)

print(f"Final sparsity: {results['achieved_sparsity']:.1%}")
print(f"Compression ratio: {results['compression_ratio']:.2f}×")
print(f"Perplexity degradation: {results['perplexity_degradation']:.2%}")
```

### Example 3: Layer-Specific Pruning

```python
from src.models.pruned_moe import IterativeMagnitudePruner

# Create pruner for specific layers
pruner = IterativeMagnitudePruner(
    initial_sparsity=0.3,
    final_sparsity=0.8,
    num_iterations=4,
    prune_layers=['output_proj', 'fc']  # Only prune these layers
)

# Execute pruning steps
for i in range(pruner.num_iterations):
    # Prune
    stats = pruner.prune_step(model, verbose=True)
    
    # Retrain (your training loop)
    for epoch in range(retrain_epochs):
        for batch in train_loader:
            # ... training code ...
            optimizer.step()
            
            # Apply masks to maintain sparsity
            pruner.train_step_with_mask(model)
    
    # Evaluate
    val_metrics = evaluate(model, val_loader)
    print(f"Iteration {i+1}: PPL = {val_metrics['perplexity']:.2f}")
```

## Key Features

### 1. Exponential Sparsity Schedule

The iterative pruner uses an exponential schedule to gradually increase sparsity:

```
s(t) = s_0 * (s_f / s_0)^(t / T)
```

Where:
- `s_0` = initial sparsity
- `s_f` = final sparsity
- `t` = current iteration
- `T` = total iterations

This allows the model to adapt gradually, maintaining better accuracy.

### 2. Mask Persistence

Pruning masks are stored and applied after each optimizer step:

```python
# During training
optimizer.step()
pruner.train_step_with_mask(model)  # Re-apply masks
```

This ensures pruned weights remain zero throughout training.

### 3. Layer Filtering

Target specific layers using pattern matching:

```python
prune_layers=['output_proj', 'fc']  # Prune layers containing these patterns
```

This focuses compression on specific components (as per requirements).

### 4. Comprehensive Metrics

Track detailed metrics at each iteration:
- Sparsity per layer
- Total weights pruned
- Perplexity before/after pruning
- Perplexity after retraining
- Compression ratio

## Testing

Run tests to verify implementation:

```bash
python -m pytest tests/test_magnitude_pruning.py -v
```

Tests cover:
- Basic magnitude pruning
- Target sparsity pruning
- Mask application
- Iterative pruning schedule
- Layer filtering
- Summary generation

## Demonstration

See `notebooks/magnitude_pruning_demo.ipynb` for:
- Basic pruning examples
- Weight distribution visualization
- Iterative pruning workflow
- Results analysis
- Layer-wise sparsity analysis

## Performance

Expected results on WikiText-2:
- **Sparsity**: 70-80% achievable
- **Perplexity degradation**: <15% with iterative retraining
- **Compression ratio**: 3-5× for targeted layers
- **Training overhead**: ~2× (due to retraining cycles)

## Requirements Satisfied

✅ **Requirement 4.8**: Prune weights with |w| < threshold in output_proj and fc layers
✅ **Iterative pruning**: Gradual sparsity increase over multiple cycles
✅ **Retraining**: Fine-tune after each pruning step to recover accuracy
✅ **Layer targeting**: Focus on output_proj and fc layers
✅ **Mask persistence**: Maintain sparsity during training

## Integration Points

1. **Compression Pipeline**: Integrated into Stage 2
2. **Model Training**: Compatible with all ResNet-BK variants
3. **Evaluation**: Works with standard evaluation metrics
4. **Checkpointing**: Saves pruned models and masks

## Future Enhancements

Potential improvements:
- Structured pruning (remove entire neurons/channels)
- Sensitivity-based pruning (prune based on loss impact)
- Dynamic pruning schedules (adaptive based on validation loss)
- Hardware-aware pruning (optimize for specific accelerators)

## References

- Han et al. (2015): "Learning both Weights and Connections for Efficient Neural Networks"
- Zhu & Gupta (2017): "To prune, or not to prune: exploring the efficacy of pruning for model compression"
- Frankle & Carbin (2019): "The Lottery Ticket Hypothesis"
