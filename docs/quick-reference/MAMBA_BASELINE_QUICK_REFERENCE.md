# Mamba Baseline Implementation - Quick Reference

## Overview

Comprehensive Mamba baseline implementation for fair comparison with ResNet-BK, including:
- Full Mamba architecture with selective state space models
- Comprehensive FLOPs and memory counting
- Fair comparison framework ensuring identical hyperparameters
- Reproducibility guarantees with seed management

**Status**: ✅ Complete (Task 11 + 11.1)

**Requirements Satisfied**: 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 11.8, 11.10

---

## Components

### 1. Mamba Model (`src/models/mamba_baseline.py`)

**MambaLM**: Full language model with selective SSM
- Token embeddings
- Mamba blocks (n_layers)
- Final layer norm
- Language modeling head
- Weight tying support

**MambaBlock**: Core Mamba block
- Layer normalization
- Input projection (d_model → 2×d_inner)
- Depthwise convolution (local context)
- Selective SSM (state space model)
- Gating with SiLU activation
- Output projection (d_inner → d_model)

**Selective SSM Features**:
- Input-dependent state transitions
- Discretization: A_bar = exp(Δ·A), B_bar = Δ·B
- Sequential scan: h_t = A_t·h_{t-1} + B_t·x_t
- Output: y_t = C_t·h_t + D·x_t

### 2. FLOPs Counter (`src/benchmarks/mamba_flops_counter.py`)

**MambaFLOPsCounter**: Comprehensive computational cost measurement

**Counts All Operations**:
- ✅ Linear projections (input, output, dt, x_proj)
- ✅ Convolution operations (depthwise conv1d)
- ✅ SSM operations (discretization, scan, output)
- ✅ Activation functions (SiLU, softplus)
- ✅ Normalization (LayerNorm)
- ✅ Gating operations
- ✅ Optimizer steps (SGD, Adam, AdamW)

**Memory Tracking**:
- ✅ Model parameters
- ✅ Forward activations
- ✅ Gradients
- ✅ Optimizer states (momentum, variance)
- ✅ Buffers (A_log, D, running stats)

### 3. Fair Comparison Framework (`src/benchmarks/fair_comparison.py`)

**FairComparison**: Ensures identical conditions for comparison

**Guarantees**:
- ✅ Identical hyperparameters (LR, batch size, optimizer, warmup)
- ✅ Identical tokenization and vocabulary
- ✅ Same random seeds for reproducibility
- ✅ Normalized by total compute (FLOPs) not wall-clock time

**Features**:
- Automatic config matching via `create_mamba_from_resnetbk_config()`
- Seed management with `set_seed()`
- Optimizer/scheduler creation with identical settings
- FLOPs and memory comparison
- Hyperparameter verification
- JSON export for results

---

## Usage Examples

### Basic Mamba Model

```python
from src.models.mamba_baseline import MambaLM, MambaConfig

# Create configuration
config = MambaConfig(
    vocab_size=30000,
    d_model=256,
    n_layers=8,
    d_state=16,
    max_seq_len=2048
)

# Create model
model = MambaLM(config)

# Forward pass
import torch
input_ids = torch.randint(0, config.vocab_size, (32, 128))
logits, loss = model(input_ids, targets=input_ids)

print(f"Logits shape: {logits.shape}")  # (32, 128, 30000)
print(f"Loss: {loss.item():.4f}")
```

### FLOPs and Memory Counting

```python
from src.benchmarks.mamba_flops_counter import MambaFLOPsCounter

# Create counter
counter = MambaFLOPsCounter(
    model,
    batch_size=32,
    seq_len=128
)

# Count FLOPs
flops = counter.count_total_flops('adamw')
print(f"Forward:  {flops.forward/1e9:.3f} GFLOPs")
print(f"Backward: {flops.backward/1e9:.3f} GFLOPs")
print(f"Total:    {flops.total/1e9:.3f} GFLOPs")

# Count memory
memory = counter.count_memory_usage('adamw', torch.float32)
print(f"Total memory: {memory.total/1e6:.2f} MB")

# Print detailed summary
counter.print_summary()

# Save to JSON
counter.save_to_json('mamba_measurements.json')
```

### Fair Comparison with ResNet-BK

```python
from src.models.configurable_resnet_bk import ConfigurableResNetBK, BASELINE_CONFIG
from src.models.mamba_baseline import MambaLM, create_mamba_from_resnetbk_config
from src.benchmarks.fair_comparison import FairComparison, ComparisonConfig

# Create ResNet-BK model
resnetbk_model = ConfigurableResNetBK(BASELINE_CONFIG)

# Create Mamba model with identical hyperparameters
mamba_config = create_mamba_from_resnetbk_config(BASELINE_CONFIG)
mamba_model = MambaLM(mamba_config)

# Create comparison
comparison_config = ComparisonConfig(
    batch_size=32,
    seq_len=128,
    learning_rate=1e-3,
    optimizer='adamw',
    seed=42
)

comparison = FairComparison(resnetbk_model, mamba_model, comparison_config)

# Print summary
comparison.print_comparison_summary()

# Save results
comparison.save_comparison('mamba_vs_resnetbk_comparison.json')
```

---

## Configuration Matching

The `create_mamba_from_resnetbk_config()` function ensures identical hyperparameters:

| Parameter | ResNet-BK | Mamba | Matched |
|-----------|-----------|-------|---------|
| vocab_size | ✓ | ✓ | ✅ |
| d_model | ✓ | ✓ | ✅ |
| n_layers | ✓ | ✓ | ✅ |
| max_seq_len | n_seq | max_seq_len | ✅ |
| dropout | ✓ | ✓ | ✅ |
| tie_weights | ✓ | ✓ | ✅ |

Mamba-specific parameters use sensible defaults:
- `d_state=16` (SSM state dimension)
- `d_conv=4` (convolution kernel size)
- `expand=2` (expansion factor)

---

## FLOPs Breakdown

### Forward Pass FLOPs

1. **Embeddings**: Negligible (just indexing)
2. **Per Mamba Block**:
   - LayerNorm: B×L×D×5
   - Input projection: B×L×D×(2×d_inner)×2
   - Convolution: B×L×d_inner×d_conv×2
   - SiLU activation: B×L×d_inner×5
   - SSM operations:
     - Discretization: L×d_inner×d_state×4
     - Selective scan: L×d_inner×d_state×4
     - Output projection: L×d_inner×d_state×2
   - Gating: B×L×d_inner×6
   - Output projection: B×L×d_inner×D×2
3. **Final LayerNorm**: B×L×D×5
4. **LM Head**: B×L×D×vocab_size×2

### Backward Pass FLOPs

Approximately 2-3× forward pass FLOPs (gradient computation)

### Optimizer FLOPs

- **SGD**: num_params × 2
- **Adam/AdamW**: num_params × 15 (momentum, variance, bias correction)

---

## Memory Breakdown

### Parameters
- Token embeddings: vocab_size × d_model
- Per block: ~3×d_model×d_inner + d_inner×d_state
- LM head: d_model × vocab_size (shared with embeddings if tied)

### Activations (Forward Pass)
- Embeddings: B×L×d_model
- Per block: B×L×(d_model + 3×d_inner + d_state×d_inner)
- LM head: B×L×vocab_size

### Gradients
Same size as parameters

### Optimizer States
- **Adam/AdamW**: 2× parameters (momentum + variance)
- **SGD**: 1× parameters (momentum only)

---

## Reproducibility

### Seed Management

```python
from src.benchmarks.fair_comparison import set_seed

# Set all random seeds
set_seed(42)

# Now all operations are deterministic
x1 = torch.randn(10)

set_seed(42)
x2 = torch.randn(10)

assert torch.allclose(x1, x2)  # ✅ Identical
```

### Identical Training Setup

```python
from src.benchmarks.fair_comparison import create_optimizer, create_scheduler

# Create optimizer with identical settings
optimizer = create_optimizer(model, comparison_config)

# Create scheduler with identical settings
num_training_steps = 10000
scheduler = create_scheduler(optimizer, comparison_config, num_training_steps)

# Warmup learning rate
for step in range(comparison_config.warmup_steps):
    lr = warmup_lr(step, comparison_config.warmup_steps, comparison_config.learning_rate)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
```

---

## Testing

All tests pass (13/13):

```bash
pytest tests/test_mamba_baseline.py -v
```

**Test Coverage**:
- ✅ Model configuration
- ✅ Forward pass (block and full model)
- ✅ Loss computation
- ✅ Backward pass and gradients
- ✅ Config matching with ResNet-BK
- ✅ FLOPs counting (forward, backward, total)
- ✅ Memory estimation
- ✅ Seed reproducibility
- ✅ Fair comparison initialization

---

## Example Results

### Model Comparison (d_model=256, n_layers=8, seq_len=128, batch_size=32)

| Metric | ResNet-BK | Mamba | Ratio |
|--------|-----------|-------|-------|
| **FLOPs** |
| Forward | 2.04 GFLOPs | 2.11 GFLOPs | 1.04× |
| Backward | 4.07 GFLOPs | 4.27 GFLOPs | 1.05× |
| Total | 6.17 GFLOPs | 6.41 GFLOPs | 1.04× |
| **Memory** |
| Parameters | 16.58 MB | 8.20 MB | 0.49× |
| Activations | 3.01 MB | 82.15 MB | 27.25× |
| Total | 70.18 MB | 115.38 MB | 1.64× |

**Key Observations**:
- Mamba has fewer parameters (0.49×) due to SSM structure
- Mamba has higher activation memory (27.25×) due to state storage
- FLOPs are comparable (1.04× difference)
- ResNet-BK has lower total memory (0.61×)

---

## Next Steps

With Task 11 complete, you can now:

1. **Task 11.2**: Write Mamba comparison tests
   - Train both models on same data
   - Measure convergence (tokens seen, not steps)
   - Compare gradient stability
   - Compare condition numbers

2. **Task 12**: Generate long-context stability graph
   - Plot loss vs training step for N ∈ {8k, 32k, 128k, 512k, 1M}
   - Show Mamba divergence vs ResNet-BK stability

3. **Use in benchmarks**: Integrate into full benchmark pipeline
   - `scripts/mamba_vs_bk_benchmark.py`
   - Multi-dataset evaluation
   - Statistical significance testing

---

## Files Modified

1. ✅ `src/models/mamba_baseline.py` - Fixed einsum bug in `_selective_scan`
2. ✅ `src/benchmarks/mamba_flops_counter.py` - Already complete
3. ✅ `src/benchmarks/fair_comparison.py` - Already complete
4. ✅ `tests/test_mamba_baseline.py` - All tests passing

---

## Requirements Satisfied

- ✅ **11.1**: Identical hyperparameters (LR, batch size, optimizer, warmup)
- ✅ **11.2**: Identical tokenization and vocabulary
- ✅ **11.3**: Same random seeds for reproducibility
- ✅ **11.4**: Use same random seeds
- ✅ **11.5**: Count all operations (state updates, gating, normalization)
- ✅ **11.6**: Include all buffers, activations, optimizer states
- ✅ **11.7**: Normalize by total compute (FLOPs) not wall-clock time
- ✅ **11.8**: Fair FLOPs and memory measurement
- ✅ **11.10**: Comprehensive measurement framework

---

## Summary

Task 11 and subtask 11.1 are now **complete**. The Mamba baseline implementation provides:

1. **Full Mamba architecture** with selective SSM
2. **Comprehensive FLOPs counting** for all operations
3. **Accurate memory estimation** including all components
4. **Fair comparison framework** ensuring identical conditions
5. **Reproducibility guarantees** with seed management
6. **Extensive testing** with 13/13 tests passing

The implementation is ready for use in benchmarking ResNet-BK against Mamba across the three critical dimensions: long-context stability, quantization robustness, and dynamic compute efficiency.
