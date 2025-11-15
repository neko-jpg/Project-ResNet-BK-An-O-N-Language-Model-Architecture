# Sparsity Loss Implementation

## Overview

This document describes the sparsity loss implementation for the Sparse BK-Core, which enables balancing between computational efficiency (sparsity) and model accuracy.

## Features

### 1. Multiple Sparsity Loss Types

The implementation supports four different sparsity loss formulations:

#### L2 Loss (Default)
```python
loss = (current_sparsity - target_sparsity)² 
```
- Quadratic penalty for deviations from target
- Smooth gradients
- Good for stable training

#### L1 Loss
```python
loss = |current_sparsity - target_sparsity|
```
- Linear penalty
- More robust to outliers
- Faster convergence in some cases

#### KL Divergence Loss
```python
loss = KL(target || current)
```
- Treats sparsity as probability distribution
- Information-theoretic interpretation
- Asymmetric penalty

#### Adaptive Loss (Huber Loss)
```python
loss = {
    0.5 * diff² if |diff| < δ
    δ * (|diff| - 0.5δ) otherwise
}
```
- Quadratic for small errors, linear for large errors
- Robust to large deviations
- Best balance between L1 and L2

### 2. Balanced Loss

The balanced loss combines accuracy and sparsity objectives:

```python
total_loss = accuracy_weight * accuracy_loss + sparsity_weight * sparsity_loss
```

**Parameters:**
- `accuracy_weight`: Weight for accuracy loss (default: 1.0)
- `sparsity_weight`: Weight for sparsity loss (default: 0.01)

**Trade-offs:**
- Higher `sparsity_weight`: More aggressive pruning, potentially lower accuracy
- Higher `accuracy_weight`: Better accuracy, less sparsity
- Optimal balance depends on application requirements

### 3. Adaptive Sparsity Scheduling

The `AdaptiveSparsityScheduler` dynamically adjusts sparsity targets and loss weights during training.

#### Schedule Types

**Linear Schedule:**
```python
sparsity_target = initial + progress * (final - initial)
```

**Cosine Schedule:**
```python
sparsity_target = initial + 0.5 * (final - initial) * (1 - cos(π * progress))
```

**Step Schedule:**
```python
sparsity_target = {
    initial if progress < 0.25
    initial + 0.33 * (final - initial) if progress < 0.5
    initial + 0.67 * (final - initial) if progress < 0.75
    final otherwise
}
```

#### Adaptive Adjustment

The scheduler can automatically adjust based on accuracy:

```python
if current_accuracy > threshold:
    # Accuracy is poor, reduce sparsity target
    sparsity_target *= 0.9
    loss_weight *= 0.5
```

## Usage

### Basic Usage

```python
from src.models.sparse_bk_core import SparseBKCore, SparseMoEResNetBKLayer

# Create sparse layer with target sparsity
layer = SparseMoEResNetBKLayer(
    d_model=64,
    n_seq=128,
    target_sparsity=0.5,  # 50% sparsity
    sparsity_loss_weight=0.01,
    sparsity_loss_type='adaptive'
)

# Forward pass
output = layer(x)

# Get sparsity loss
sparsity_loss = layer.get_sparsity_loss()

# Or get balanced loss
total_loss, loss_dict = layer.get_balanced_loss(
    accuracy_loss,
    sparsity_weight=0.01,
    accuracy_weight=1.0
)
```

### With Adaptive Scheduling

```python
from src.models.sparse_bk_core import AdaptiveSparsityScheduler

# Create scheduler
scheduler = AdaptiveSparsityScheduler(
    initial_sparsity=0.2,
    final_sparsity=0.5,
    initial_weight=0.001,
    final_weight=0.01,
    warmup_steps=1000,
    schedule_type='cosine'
)

# Training loop
for step in range(num_steps):
    # Update scheduler
    state = scheduler.step(current_accuracy=val_loss)
    
    # Update layer's sparsity target
    layer.sparse_bk_core.target_sparsity = state['sparsity_target']
    layer.sparsity_loss_weight = state['loss_weight']
    
    # Forward pass
    output = layer(x)
    
    # Compute loss
    total_loss, loss_dict = layer.get_balanced_loss(accuracy_loss)
    
    # Backward pass
    total_loss.backward()
    optimizer.step()
```

### Monitoring Sparsity

```python
# Get sparsity statistics
stats = layer.get_sparsity_stats()

print(f"Sparsity ratio: {stats['sparsity_ratio']:.3f}")
print(f"Positions computed: {stats['num_computed']:.1f} / {n_seq}")
print(f"Target sparsity: {stats['target_sparsity']:.3f}")
```

## Implementation Details

### SparseBKCore.sparsity_loss()

```python
def sparsity_loss(self, mask, loss_type='l2'):
    """
    Encourage target sparsity level.
    
    Args:
        mask: (B, N) binary mask (1 = compute, 0 = skip)
        loss_type: 'l2', 'l1', 'kl', or 'adaptive'
    
    Returns:
        loss: scalar sparsity loss
    """
```

### SparseBKCore.balanced_sparsity_loss()

```python
def balanced_sparsity_loss(self, mask, accuracy_loss, 
                          sparsity_weight=1.0, accuracy_weight=1.0):
    """
    Balanced loss that trades off sparsity and accuracy.
    
    Returns:
        total_loss: scalar balanced loss
        loss_dict: dictionary with individual loss components
    """
```

### AdaptiveSparsityScheduler

```python
class AdaptiveSparsityScheduler:
    def __init__(
        self,
        initial_sparsity=0.2,
        final_sparsity=0.5,
        initial_weight=0.001,
        final_weight=0.01,
        warmup_steps=1000,
        schedule_type='cosine',
        accuracy_threshold=None
    ):
        ...
    
    def step(self, current_accuracy=None):
        """
        Update scheduler state.
        
        Returns:
            dict with current_sparsity_target and current_weight
        """
```

## Hyperparameter Tuning

### Sparsity Target

- **0.2-0.3**: Conservative, minimal accuracy loss
- **0.4-0.5**: Balanced, good trade-off
- **0.6-0.7**: Aggressive, significant speedup but may impact accuracy

### Loss Weight

- **0.001-0.01**: Typical range for most applications
- **0.1+**: Strong sparsity enforcement, may hurt accuracy
- **<0.001**: Weak enforcement, may not achieve target sparsity

### Schedule Type

- **Linear**: Simple, predictable
- **Cosine**: Smooth, gradual transition
- **Step**: Discrete stages, good for curriculum learning

## Performance Considerations

### Computational Cost

The sparsity loss adds minimal overhead:
- Mask mean computation: O(B*N)
- Loss computation: O(1)
- Total overhead: <1% of forward pass time

### Memory Usage

- Mask storage: B*N floats (negligible)
- No additional activations stored

### Training Stability

Tips for stable training:
1. Start with low sparsity target (0.2-0.3)
2. Use cosine schedule for smooth transitions
3. Monitor accuracy and adjust weights if needed
4. Use adaptive loss type for robustness

## Examples

See `examples/sparsity_loss_demo.py` for comprehensive demonstrations:

```bash
python examples/sparsity_loss_demo.py
```

This demonstrates:
1. Different sparsity loss types
2. Balanced loss with various weight configurations
3. Adaptive sparsity scheduling
4. Training with sparsity loss

## Testing

Run tests with:

```bash
pytest tests/test_sparse_bk_core.py::TestSparseBKCore::test_sparsity_loss_types -v
pytest tests/test_sparse_bk_core.py::TestSparseBKCore::test_balanced_sparsity_loss -v
pytest tests/test_sparse_bk_core.py::TestAdaptiveSparsityScheduler -v
```

## References

- Requirement 6.14: "THE System SHALL implement early exiting for inference"
- Task 7.6: "Implement sparsity loss - Encourage target sparsity level (e.g., 50%), Balance sparsity vs accuracy"
- Related: Adaptive Computation Time (ACT) for dynamic computation

## Future Enhancements

Potential improvements:
1. Per-layer sparsity targets
2. Token-wise sparsity adaptation
3. Automatic weight tuning via meta-learning
4. Sparsity-aware learning rate scheduling
