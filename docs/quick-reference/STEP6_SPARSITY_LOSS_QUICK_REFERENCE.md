# Sparsity Loss - Quick Reference

## Quick Start

```python
from src.models.sparse_bk_core import SparseMoEResNetBKLayer, AdaptiveSparsityScheduler

# Create layer with sparsity loss
layer = SparseMoEResNetBKLayer(
    d_model=64, n_seq=128,
    target_sparsity=0.5,           # 50% sparsity target
    sparsity_loss_weight=0.01,     # Loss weight
    sparsity_loss_type='adaptive'  # Loss type
)

# Create adaptive scheduler
scheduler = AdaptiveSparsityScheduler(
    initial_sparsity=0.2,
    final_sparsity=0.5,
    warmup_steps=1000,
    schedule_type='cosine'
)

# Training loop
for step in range(num_steps):
    # Update sparsity target
    state = scheduler.step()
    layer.sparse_bk_core.target_sparsity = state['sparsity_target']
    
    # Forward pass
    output = layer(x)
    
    # Get balanced loss
    total_loss, loss_dict = layer.get_balanced_loss(accuracy_loss)
    
    # Backward and optimize
    total_loss.backward()
    optimizer.step()
```

## Loss Types

| Type | Formula | Use Case |
|------|---------|----------|
| **adaptive** ✅ | Huber loss | **Recommended default** - Robust to outliers |
| l2 | (current - target)² | Smooth gradients, stable training |
| l1 | \|current - target\| | Faster convergence |
| kl | KL(target \|\| current) | Information-theoretic |

## Hyperparameters

### Sparsity Target
- **0.2-0.3**: Conservative, minimal accuracy loss
- **0.4-0.5**: Balanced ✅ **Recommended**
- **0.6-0.7**: Aggressive, maximum speedup

### Loss Weight
- **0.001-0.01**: Typical range ✅ **Recommended**
- **0.1+**: Strong enforcement (may hurt accuracy)
- **<0.001**: Weak enforcement

### Schedule Type
- **cosine**: Smooth transition ✅ **Recommended**
- **linear**: Simple, predictable
- **step**: Curriculum learning

## API Reference

### SparseBKCore

```python
# Sparsity loss
loss = sparse_bk_core.sparsity_loss(mask, loss_type='adaptive')

# Balanced loss
total_loss, loss_dict = sparse_bk_core.balanced_sparsity_loss(
    mask, accuracy_loss, 
    sparsity_weight=0.01, 
    accuracy_weight=1.0
)
```

### SparseMoEResNetBKLayer

```python
# Get sparsity loss
loss = layer.get_sparsity_loss(loss_type='adaptive')

# Get balanced loss
total_loss, loss_dict = layer.get_balanced_loss(
    accuracy_loss,
    sparsity_weight=0.01,
    accuracy_weight=1.0
)

# Get statistics
stats = layer.get_sparsity_stats()
# Returns: {'sparsity_ratio', 'num_computed', 'target_sparsity'}
```

### AdaptiveSparsityScheduler

```python
scheduler = AdaptiveSparsityScheduler(
    initial_sparsity=0.2,
    final_sparsity=0.5,
    initial_weight=0.001,
    final_weight=0.01,
    warmup_steps=1000,
    schedule_type='cosine',
    accuracy_threshold=3.0  # Optional
)

# Update scheduler
state = scheduler.step(current_accuracy=val_loss)
# Returns: {'sparsity_target', 'loss_weight', 'progress', 'step'}

# Reset scheduler
scheduler.reset()
```

## Loss Dictionary

The `balanced_sparsity_loss()` and `get_balanced_loss()` methods return a dictionary:

```python
loss_dict = {
    'total_loss': total_loss,           # Combined loss
    'accuracy_loss': accuracy_loss,     # Accuracy component
    'sparsity_loss': sparsity_loss,     # Sparsity component
    'sparsity_weight': sparsity_weight, # Weight used
    'accuracy_weight': accuracy_weight, # Weight used
    'current_sparsity': current,        # Actual sparsity
    'target_sparsity': target           # Target sparsity
}
```

## Monitoring

```python
# Get sparsity statistics
stats = layer.get_sparsity_stats()

print(f"Sparsity ratio: {stats['sparsity_ratio']:.3f}")
print(f"Positions computed: {stats['num_computed']:.1f} / {n_seq}")
print(f"Target sparsity: {stats['target_sparsity']:.3f}")

# Monitor loss components
total_loss, loss_dict = layer.get_balanced_loss(accuracy_loss)

print(f"Total loss: {loss_dict['total_loss'].item():.4f}")
print(f"Accuracy loss: {loss_dict['accuracy_loss'].item():.4f}")
print(f"Sparsity loss: {loss_dict['sparsity_loss'].item():.6f}")
print(f"Current sparsity: {loss_dict['current_sparsity']:.3f}")
```

## Examples

### Basic Usage

```python
# Create layer
layer = SparseMoEResNetBKLayer(
    d_model=64, n_seq=128,
    target_sparsity=0.5,
    sparsity_loss_weight=0.01
)

# Forward pass
output = layer(x)

# Get sparsity loss
sparsity_loss = layer.get_sparsity_loss()

# Total loss
total_loss = accuracy_loss + sparsity_loss
```

### With Balanced Loss

```python
# Forward pass
output = layer(x)

# Compute accuracy loss
accuracy_loss = criterion(output, target)

# Get balanced loss
total_loss, loss_dict = layer.get_balanced_loss(
    accuracy_loss,
    sparsity_weight=0.01,
    accuracy_weight=1.0
)

# Backward pass
total_loss.backward()
optimizer.step()
```

### With Adaptive Scheduling

```python
scheduler = AdaptiveSparsityScheduler(
    initial_sparsity=0.2,
    final_sparsity=0.5,
    warmup_steps=1000,
    schedule_type='cosine'
)

for step in range(num_steps):
    # Update sparsity target
    state = scheduler.step(current_accuracy=val_loss)
    layer.sparse_bk_core.target_sparsity = state['sparsity_target']
    layer.sparsity_loss_weight = state['loss_weight']
    
    # Training step
    output = layer(x)
    total_loss, loss_dict = layer.get_balanced_loss(accuracy_loss)
    total_loss.backward()
    optimizer.step()
    
    # Log progress
    if step % 100 == 0:
        print(f"Step {step}: "
              f"Sparsity target {state['sparsity_target']:.3f}, "
              f"Current {loss_dict['current_sparsity']:.3f}")
```

## Testing

```bash
# Test sparsity loss types
pytest tests/test_sparse_bk_core.py::TestSparseBKCore::test_sparsity_loss_types -v

# Test balanced loss
pytest tests/test_sparse_bk_core.py::TestSparseBKCore::test_balanced_sparsity_loss -v

# Test adaptive scheduler
pytest tests/test_sparse_bk_core.py::TestAdaptiveSparsityScheduler -v

# Run all sparse BK-Core tests
pytest tests/test_sparse_bk_core.py -v
```

## Demo

```bash
# Run comprehensive demo
python examples/sparsity_loss_demo.py
```

Demonstrates:
1. Different sparsity loss types
2. Balanced loss with various weights
3. Adaptive sparsity scheduling
4. Visualization of schedules

## Performance

- **Computational overhead**: <1% of forward pass time
- **Memory overhead**: Negligible (B*N floats for mask)
- **Training stability**: Robust with adaptive loss type

## Tips

1. **Start conservative**: Begin with low sparsity target (0.2-0.3)
2. **Use cosine schedule**: Provides smooth transitions
3. **Monitor accuracy**: Use adaptive adjustment if accuracy drops
4. **Tune weights**: Adjust based on application requirements
5. **Use adaptive loss**: Best balance between robustness and performance

## Documentation

- Full documentation: `docs/SPARSITY_LOSS.md`
- Completion summary: `TASK_7.6_SPARSITY_LOSS_COMPLETION.md`
- Sparse BK-Core docs: `docs/SPARSE_BK_CORE.md`

## Related

- Task 7.4: Sparse BK-Core implementation
- Task 7.5: Sparse computation optimization
- Task 7.7: Early exiting for inference (next)
- Requirement 6.14: Early exiting implementation
