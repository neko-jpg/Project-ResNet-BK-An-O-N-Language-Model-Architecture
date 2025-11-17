# Step 6 - Adaptive Computation Time (ACT) Quick Reference

## Overview

Adaptive Computation Time (ACT) enables dynamic layer execution in ResNet-BK, allowing tokens to "exit early" when they've been processed sufficiently. This is the first component of Step 6: Algorithmic Innovations.

## Quick Start

### Basic Usage

```python
from src.models.adaptive_computation import ACTLanguageModel, ACTTrainer

# Create ACT model
model = ACTLanguageModel(
    vocab_size=10000,
    d_model=64,
    n_layers=4,
    n_seq=128,
    act_threshold=0.95,  # Halting threshold
    act_lambda=0.01      # Ponder cost weight
)

# Training
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
trainer = ACTTrainer(model, optimizer)

for x_batch, y_batch in dataloader:
    metrics = trainer.train_step(x_batch, y_batch)
    print(f"Loss: {metrics['total_loss']:.4f}, "
          f"Layers: {metrics['avg_layers_executed']:.2f}")

# Inference
model.eval()
with torch.no_grad():
    logits = model(x, return_ponder_cost=False)
```

## Key Parameters

| Parameter | Default | Description | Impact |
|-----------|---------|-------------|--------|
| `act_threshold` | 0.99 | Cumulative halting probability threshold | Lower → faster, less accurate |
| `act_lambda` | 0.01 | Ponder cost weight in loss | Higher → encourages early halting |
| `d_model` | 64 | Hidden dimension | Standard ResNet-BK parameter |
| `n_layers` | 4 | Number of layers | More layers → more potential savings |

## Hyperparameter Tuning

### Recommended Presets

```python
# Maximum Accuracy (minimal speedup)
model = ACTLanguageModel(..., act_threshold=0.99, act_lambda=0.001)
# Expected: 1.1-1.3× speedup, <2% accuracy loss

# Balanced (recommended)
model = ACTLanguageModel(..., act_threshold=0.95, act_lambda=0.01)
# Expected: 1.3-1.5× speedup, 2-5% accuracy loss

# Maximum Speed
model = ACTLanguageModel(..., act_threshold=0.80, act_lambda=0.1)
# Expected: 1.5-2.0× speedup, 5-10% accuracy loss
```

### Grid Search (Task 7.2)

```python
thresholds = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
lambdas = [0.001, 0.005, 0.01, 0.05, 0.1]

for threshold in thresholds:
    for lambda_val in lambdas:
        model = ACTLanguageModel(..., 
                                 act_threshold=threshold,
                                 act_lambda=lambda_val)
        # Train and evaluate
        # Track: avg_layers_executed, perplexity, speedup
```

## Monitoring

### Key Metrics

```python
# During training
metrics = trainer.train_step(x_batch, y_batch)

print(f"Total Loss: {metrics['total_loss']:.4f}")
print(f"CE Loss: {metrics['ce_loss']:.4f}")
print(f"Ponder Cost: {metrics['ponder_cost']:.4f}")
print(f"Avg Layers: {metrics['avg_layers_executed']:.2f}")

# Average over epoch
avg_metrics = trainer.get_average_metrics()
print(f"Avg CE Loss: {avg_metrics['avg_ce_loss']:.4f}")
print(f"Avg Ponder Cost: {avg_metrics['avg_ponder_cost']:.4f}")
```

### Speedup Calculation

```python
n_layers = 4
avg_layers = model.get_avg_layers_executed()
speedup = n_layers / avg_layers

print(f"Speedup: {speedup:.2f}×")
print(f"Computational Savings: {(1 - avg_layers/n_layers)*100:.1f}%")
```

## Architecture Details

### AdaptiveResNetBKBlock

```
Input (B, N, D)
  ↓
LayerNorm
  ↓
MoEResNetBKLayer
  ↓
Residual Connection
  ↓
Halting Unit (D → D/2 → 1 → Sigmoid)
  ↓
Compute p_halt, update cumsum
  ↓
Determine weights and still_running mask
  ↓
Output (B, N, D), cumsum, still_running, weight
```

### Loss Function

```
Total Loss = CE_loss + λ * ponder_cost

where:
  CE_loss = CrossEntropy(logits, targets)
  ponder_cost = sum(weights) / (batch_size * seq_length)
  λ = act_lambda hyperparameter
```

## Performance Characteristics

### Expected Results (after training)

| Metric | Value |
|--------|-------|
| Average Layers Executed | 2.5-3.5 (out of 4) |
| Speedup | 1.3-1.6× |
| Memory Overhead | <5% |
| Accuracy Impact | -2% to -5% perplexity |

### Computational Breakdown

```
Standard Model:
  - Layers executed: 4 (always)
  - FLOPs per token: 4 × layer_flops

ACT Model:
  - Layers executed: 2.5-3.5 (average)
  - FLOPs per token: 2.5-3.5 × layer_flops + halting_overhead
  - Halting overhead: ~2% of layer_flops
  - Net speedup: 1.3-1.6×
```

## Testing

```bash
# Run ACT tests
python -m pytest tests/test_adaptive_computation.py -v

# Run demo
python examples/act_demo.py

# Expected output:
# - 14 tests pass
# - Demo shows threshold/lambda effects
# - Comparison with standard model
```

## Common Issues

### Issue: All tokens halt at first layer

**Cause**: Halting unit learns to always output high probability

**Solution**:
- Increase `act_threshold` (e.g., 0.99)
- Decrease `act_lambda` (e.g., 0.001)
- Ensure sufficient training data

### Issue: No tokens halt (all use max layers)

**Cause**: Halting unit learns to always output low probability

**Solution**:
- Decrease `act_threshold` (e.g., 0.90)
- Increase `act_lambda` (e.g., 0.05)
- Check if ponder cost is being added to loss

### Issue: High variance in layers executed

**Cause**: Unstable halting unit training

**Solution**:
- Use gradient clipping (already implemented: max_norm=0.5)
- Reduce learning rate for halting unit
- Increase batch size for more stable gradients

## Integration with Other Optimizations

### With Multi-Scale Processing (Task 7.3)

```python
# Apply ACT at each scale level
# Expected combined speedup: 1.4× (ACT) × 2× (multi-scale) = 2.8×
```

### With Learned Sparsity (Task 7.4)

```python
# Combine early halting with sparse computation
# Expected combined speedup: 1.4× (ACT) × 1.8× (sparsity) = 2.5×
```

### With All Step 6 Optimizations

```python
# ACT + Multi-Scale + Learned Sparsity
# Expected combined speedup: 1.4 × 2 × 1.8 ≈ 5×
# Target: 10× (may need additional optimizations)
```

## Files Reference

| File | Purpose |
|------|---------|
| `src/models/adaptive_computation.py` | Core implementation |
| `tests/test_adaptive_computation.py` | Unit tests |
| `examples/act_demo.py` | Interactive demo |
| `docs/ACT_IMPLEMENTATION.md` | Detailed documentation |
| `TASK_7.1_ACT_COMPLETION.md` | Implementation summary |

## Next Steps

1. **Task 7.2**: Tune ACT hyperparameters
   - Grid search over threshold and λ
   - Find optimal balance for WikiText-2

2. **Task 7.3**: Implement multi-scale processing
   - Hierarchical sequence processing
   - Learned downsampling/upsampling

3. **Task 7.4**: Implement learned sparsity
   - Sparse BK-Core computation
   - Importance prediction

## Requirements Satisfied

✅ **Requirement 6.1**: Adaptive computation time implemented  
✅ **Requirement 6.2**: Cumulative halting probability tracking  
✅ **Requirement 6.3**: Ponder cost added to loss function

## Status

**Task 7.1**: ✅ Complete  
**Task 7.2**: ⏳ Next  
**Task 7.3**: ⏳ Pending  
**Task 7.4**: ⏳ Pending

---

*Last Updated: Task 7.1 completion*
