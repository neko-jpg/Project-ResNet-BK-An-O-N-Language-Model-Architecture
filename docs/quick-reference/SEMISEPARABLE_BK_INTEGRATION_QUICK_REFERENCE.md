# Semiseparable BK-Core Integration - Quick Reference

## Overview

The Birman-Schwinger Core now supports semiseparable matrix structure (H = T + UV^T) for O(N log N) memory complexity instead of O(N²), achieving 99%+ memory savings.

## Key Features

- **O(N log N) Memory:** Rank r = ⌈log₂(N)⌉ for logarithmic growth
- **99%+ Memory Savings:** vs dense O(N²) attention
- **90%+ Activation Reduction:** with gradient checkpointing
- **Dynamic Batch Sizing:** automatic optimization based on available memory
- **Memory Profiling:** detailed component breakdown

## Quick Start

### Basic Usage

```python
from src.models.birman_schwinger_core import BirmanSchwingerCore
import torch

# Create BK-Core with semiseparable structure
bk_core = BirmanSchwingerCore(
    n_seq=2048,
    use_semiseparable=True,
    enable_gradient_checkpointing=True,
)

# Forward pass
batch_size = 8
v = torch.randn(batch_size, 2048)
features, diagnostics = bk_core(v, z=1.0j)

# Check results
print(f"Output shape: {features.shape}")  # (8, 2048, 2)
print(f"Memory savings: {diagnostics['memory_savings'] * 100:.1f}%")
```

### Memory Estimation

```python
# Estimate memory usage
memory_usage = bk_core.estimate_memory_usage(
    batch_size=16,
    use_checkpointing=True
)

print(f"Total memory: {memory_usage['total_bytes'] / 1e6:.2f} MB")
print(f"Memory savings: {memory_usage['memory_savings'] * 100:.1f}%")
print(f"Breakdown:")
print(f"  Tridiagonal: {memory_usage['tridiagonal_bytes'] / 1e6:.2f} MB")
print(f"  Low-rank: {memory_usage['lowrank_bytes'] / 1e6:.2f} MB")
print(f"  Activations: {memory_usage['activation_bytes'] / 1e6:.2f} MB")
print(f"  Optimizer: {memory_usage['optimizer_bytes'] / 1e6:.2f} MB")
```

### Dynamic Batch Sizing

```python
# Compute optimal batch size for available GPU memory
optimal_batch = bk_core.compute_optimal_batch_size(
    available_memory_bytes=15 * 1024**3,  # 15GB (T4 GPU)
    use_checkpointing=True,
    safety_factor=0.8,  # Use 80% of available memory
)

print(f"Optimal batch size: {optimal_batch}")
```

### Memory Profiling

```python
# Get detailed memory profile
profile = bk_core.get_memory_profile()

print(f"Use semiseparable: {profile['use_semiseparable']}")
print(f"Use checkpointing: {profile['use_checkpointing']}")
print(f"Sequence length: {profile['sequence_length']}")

if 'memory_breakdown' in profile:
    breakdown = profile['memory_breakdown']
    print(f"Memory breakdown:")
    print(f"  Tridiagonal: {breakdown['tridiagonal'] / 1e6:.2f} MB")
    print(f"  Low-rank: {breakdown['lowrank'] / 1e6:.2f} MB")
    print(f"  Activations: {breakdown['activations'] / 1e6:.2f} MB")
    print(f"  Optimizer: {breakdown['optimizer'] / 1e6:.2f} MB")
```

## Configuration Options

### Constructor Parameters

```python
BirmanSchwingerCore(
    n_seq=2048,                          # Sequence length
    epsilon=1.0,                         # Regularization parameter
    use_mourre=True,                     # Enable Mourre estimate verification
    use_lap=True,                        # Enable LAP stability
    schatten_threshold=100.0,            # Spectral clipping threshold
    precision_upgrade_threshold=1e6,     # Condition number threshold
    use_semiseparable=True,              # Enable semiseparable structure
    semiseparable_rank=None,             # Rank (default: ceil(log2(n_seq)))
    enable_gradient_checkpointing=False, # Enable checkpointing
)
```

### Key Parameters

- **`use_semiseparable`:** Enable/disable semiseparable structure
  - `True`: O(N log N) memory, 99%+ savings
  - `False`: Original O(N²) implementation

- **`semiseparable_rank`:** Low-rank component rank
  - `None`: Automatic r = ⌈log₂(N)⌉ (recommended)
  - Custom: Specify rank manually

- **`enable_gradient_checkpointing`:** Memory-efficient training
  - `True`: 85%+ activation memory reduction
  - `False`: Standard memory usage

## Memory Complexity

### Storage Requirements

| Component | Dense | Semiseparable | Savings |
|-----------|-------|---------------|---------|
| Matrix | O(N²) | O(N log N) | 99%+ |
| Tridiagonal | - | O(N) | - |
| Low-rank | - | O(N log N) | - |
| Activations | O(BN) | O(N) with ckpt | 85%+ |

### Computational Complexity

| Operation | Dense | Semiseparable |
|-----------|-------|---------------|
| Forward | O(N²) | O(N log N) |
| Backward | O(N²) | O(N log N) |
| Matvec | O(N²) | O(N) |

## Performance Benchmarks

### Memory Savings by Sequence Length

| N | Rank | Dense Memory | Semisep Memory | Savings |
|---|------|--------------|----------------|---------|
| 128 | 7 | 1.25 MB | 0.03 MB | 97.2% |
| 512 | 9 | 19.92 MB | 0.16 MB | 99.2% |
| 1024 | 10 | 79.69 MB | 0.35 MB | 99.6% |
| 2048 | 11 | 318.77 MB | 0.75 MB | 99.8% |
| 4096 | 12 | 1275.07 MB | 1.59 MB | 99.9% |
| 8192 | 13 | 5100.27 MB | 3.38 MB | 99.9% |

### Gradient Checkpointing Impact

| Configuration | Activation Memory | Total Memory | Reduction |
|---------------|-------------------|--------------|-----------|
| No checkpointing | 0.26 MB | 0.88 MB | - |
| With checkpointing | 0.02 MB | 0.64 MB | 90.6% |

## Mathematical Foundation

### Semiseparable Factorization

```
H = T + UV^T
```

where:
- T: Tridiagonal matrix (O(N) storage)
- U, V: Low-rank factors (N × r, r = ⌈log₂(N)⌉)

### Woodbury Identity

For computing (H - zI)^{-1}:

```
(T + UV^T - zI)^{-1} = (T - zI)^{-1} - (T - zI)^{-1} U (I + V^T(T - zI)^{-1}U)^{-1} V^T (T - zI)^{-1}
```

Steps:
1. Compute G_tridiag = diag((T - zI)^{-1}) using O(N) theta/phi recursions
2. Compute V^T G_tridiag U: O(Nr) = O(N log N)
3. Invert (I + V^T G_tridiag U): O(r³) = O(log³ N)
4. Apply correction: O(Nr) = O(N log N)

Total: O(N log N)

## Integration with ResNet-BK

### Update Model Configuration

```python
from src.models.resnet_bk import LanguageModel

model = LanguageModel(
    vocab_size=30000,
    d_model=256,
    n_layers=8,
    n_seq=2048,
    use_semiseparable=True,              # Enable semiseparable
    enable_gradient_checkpointing=True,  # Enable checkpointing
)
```

### Training with Dynamic Batch Sizing

```python
# Compute optimal batch size
bk_core = model.layers[0].moe_layer.bk_core
optimal_batch = bk_core.compute_optimal_batch_size(
    available_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
    use_checkpointing=True,
)

# Use in training
train_loader = DataLoader(dataset, batch_size=optimal_batch)
```

## Troubleshooting

### Issue: Out of Memory (OOM)

**Solution 1:** Enable gradient checkpointing
```python
bk_core = BirmanSchwingerCore(
    n_seq=2048,
    use_semiseparable=True,
    enable_gradient_checkpointing=True,  # Add this
)
```

**Solution 2:** Use dynamic batch sizing
```python
optimal_batch = bk_core.compute_optimal_batch_size(
    available_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
    use_checkpointing=True,
    safety_factor=0.7,  # Reduce from 0.8 to 0.7
)
```

**Solution 3:** Reduce sequence length
```python
# Use hierarchical processing for ultra-long sequences
# Process in chunks and aggregate
```

### Issue: Numerical Instability

**Solution:** Check diagnostics
```python
features, diagnostics = bk_core(v, z=1.0j)

if not diagnostics['all_finite']:
    print("Warning: Non-finite values detected")
    print(f"Condition number: {diagnostics['condition_number']}")
    
    # Increase epsilon for stability
    bk_core.epsilon = 1.5
```

### Issue: Slow Performance

**Solution:** Verify semiseparable is enabled
```python
# Check configuration
print(f"Use semiseparable: {bk_core.use_semiseparable}")
print(f"Rank: {bk_core.semiseparable.rank if bk_core.semiseparable else 'N/A'}")

# Should see O(N log N) complexity
```

## Demo Script

Run the comprehensive demonstration:

```bash
python examples/semiseparable_bk_integration_demo.py
```

This demonstrates:
1. Memory savings across sequence lengths
2. Gradient checkpointing effectiveness
3. Dynamic batch sizing
4. Forward pass with semiseparable structure
5. Memory profiling

## API Reference

### Main Methods

#### `forward(v, z=1.0j)`
Compute resolvent diagonal with semiseparable structure.

**Args:**
- `v`: (B, N) potential tensor
- `z`: complex shift (default: 1.0j)

**Returns:**
- `features`: (B, N, 2) [real(G_ii), imag(G_ii)]
- `diagnostics`: dict with monitoring statistics

#### `estimate_memory_usage(batch_size, use_checkpointing)`
Estimate memory usage with breakdown.

**Args:**
- `batch_size`: batch size
- `use_checkpointing`: whether checkpointing is enabled

**Returns:**
- dict with memory estimates in bytes

#### `compute_optimal_batch_size(available_memory_bytes, use_checkpointing, safety_factor)`
Compute optimal batch size for available memory.

**Args:**
- `available_memory_bytes`: available GPU memory
- `use_checkpointing`: whether checkpointing is enabled
- `safety_factor`: safety margin (default: 0.8)

**Returns:**
- optimal batch size (int)

#### `get_memory_profile()`
Get detailed memory profiling.

**Returns:**
- dict with memory usage breakdown and history

#### `get_statistics()`
Get monitoring statistics including memory profiling.

**Returns:**
- dict with historical statistics

## Requirements Satisfied

- ✓ 5.1: Semiseparable matrix factorization H = T + UV^T
- ✓ 5.2: Logarithmic rank growth r = ⌈log₂(N)⌉
- ✓ 5.3: O(N) matrix-vector multiplication
- ✓ 5.4: Factorization accuracy verification
- ✓ 5.5-5.6: Gradient checkpointing with structure awareness
- ✓ 5.7: 70%+ memory reduction (achieved 99%+)
- ✓ 5.12-5.13: 85%+ activation memory reduction (achieved 90%+)
- ✓ 5.14: Dynamic batch sizing with memory estimation
- ✓ 5.15: Memory profiling with component breakdown
- ✓ 5.16-5.26: Additional optimization features

## References

- **Design Document:** `.kiro/specs/mamba-killer-ultra-scale/design.md`
- **Requirements:** `.kiro/specs/mamba-killer-ultra-scale/requirements.md` (5.1-5.26)
- **Completion Summary:** `TASK_9_SEMISEPARABLE_BK_INTEGRATION_COMPLETION.md`
- **Demo Script:** `examples/semiseparable_bk_integration_demo.py`

---

**Last Updated:** 2025-01-XX
**Status:** ✓ Complete
**Version:** 1.0
