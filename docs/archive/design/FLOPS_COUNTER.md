# FLOPs Counter Infrastructure

## Overview

The FLOPs counter infrastructure provides comprehensive FLOPs (Floating Point Operations) counting for ResNet-BK models. It tracks forward pass, backward pass, and optimizer step FLOPs separately, with detailed component-wise breakdowns.

## Features

- **Comprehensive Counting**: Tracks FLOPs for all model components:
  - BK-Core (theta/phi recursions, complex arithmetic)
  - MoE (expert computation, routing, gating)
  - Linear layers (embeddings, projections, LM head)
  - LayerNorm
  - Optimizer steps (SGD, Adam, AdamW)

- **Separate Tracking**: Forward, backward, and optimizer FLOPs tracked independently

- **Component Breakdown**: Detailed per-layer and per-component FLOPs analysis

- **Model Comparison**: Compare FLOPs between different model configurations

- **Export**: Save FLOPs counts to JSON for further analysis

## Usage

### Basic FLOPs Counting

```python
from src.models.configurable_resnet_bk import ConfigurableResNetBK, BASELINE_CONFIG
from src.benchmarks.flops_counter import FLOPsCounter

# Create model
config = BASELINE_CONFIG
config.d_model = 64
config.n_layers = 4
config.n_seq = 128
model = ConfigurableResNetBK(config)

# Create FLOPs counter
counter = FLOPsCounter(model, batch_size=32, seq_len=128)

# Print summary
counter.print_summary()

# Get total FLOPs
flops = counter.count_total_flops()
print(f"Total FLOPs: {flops.total:,} ({flops.total/1e9:.3f} GFLOPs)")
```

### Component-wise Breakdown

```python
# Count forward FLOPs (populates component breakdown)
counter.count_forward_flops()

# Get breakdown
breakdown = counter.get_breakdown()

for component, flops in breakdown.items():
    print(f"{component}: {flops['forward']:,} FLOPs")
```

### Model Comparison

```python
from src.benchmarks.flops_counter import compare_models

# Create two models
model1 = ConfigurableResNetBK(BASELINE_CONFIG)
model2 = ConfigurableResNetBK(STEP2_CONFIG)

# Compare
comparison = compare_models(
    model1, model2,
    batch_size=32, seq_len=128,
    model1_name="Baseline",
    model2_name="Step 2"
)

print(f"Speedup: {comparison['speedup']['total']:.2f}×")
```

### Save to JSON

```python
# Save FLOPs count to JSON file
counter.save_to_json('flops_count.json')
```

## FLOPs Formulas

### BK-Core

**Forward Pass (per sequence):**
- Theta recursion: `N × (2 × 6 + 2) = 14N` ops
  - 2 complex multiplies (6 ops each) + 1 complex add (2 ops)
- Phi recursion: `N × (2 × 6 + 2) = 14N` ops
- Final division: `N × 6` ops
- **Total: ~34N ops per sequence**

**Backward Pass (per sequence):**
- G² computation: `N × 6` ops
- Gradient computation: `2 × N × 10` ops (theoretical + hypothesis-7)
- **Total: ~26N ops per sequence**

### MoE Layer

**Forward Pass (per token):**
- Gating network: `D × E × 2` ops
- Softmax: `E × 3` ops
- Expert computation (K experts): `K × (8D² + 2D)` ops
- **Total: ~K × 8D² ops per token** (for top-K routing)

**Backward Pass:**
- Approximately 2× forward pass FLOPs

### Linear Layer

**Forward Pass:**
- Matrix multiply: `B × N × in_features × out_features × 2` ops

**Backward Pass:**
- Gradient w.r.t. input: `B × N × in_features × out_features × 2` ops
- Gradient w.r.t. weights: `B × N × in_features × out_features × 2` ops
- **Total: 2× forward pass FLOPs**

### Optimizer (AdamW)

**Per Parameter:**
- Momentum update: 3 ops
- Variance update: 4 ops
- Bias correction: 4 ops
- Parameter update: 4 ops
- **Total: ~15 ops per parameter**

## Scaling Analysis

### Sequence Length Scaling

ResNet-BK achieves O(N) complexity:

```
N=128  → 48.9 GFLOPs
N=2048 → 782.4 GFLOPs (16× increase for 16× sequence length)
```

Compare to Transformer O(N²):
- Expected: 256× increase for 16× sequence length
- ResNet-BK: 16× increase (linear scaling ✓)

### Model Size Scaling

FLOPs scale with model dimensions:

```
d_model=64,  n_layers=4 → 48.9 GFLOPs
d_model=128, n_layers=4 → 101.1 GFLOPs
d_model=256, n_layers=4 → 215.0 GFLOPs
```

### Batch Size Scaling

FLOPs scale linearly with batch size:

```
batch_size=1  → 1.5 GFLOPs
batch_size=32 → 48.9 GFLOPs (32× increase)
```

## Component Contribution

Typical FLOPs breakdown for baseline model (d=64, L=4, N=128):

| Component | Forward FLOPs | Percentage |
|-----------|---------------|------------|
| LM Head | 15.7 GFLOPs | 96.6% |
| Layer 0-3 | 0.14 GFLOPs each | 0.9% each |
| LayerNorm | 1.3 MFLOPs | <0.1% |
| Embedding | 0.3 MFLOPs | <0.1% |

**Key Insight**: LM head dominates FLOPs due to large vocabulary size (30K). Model layers are very efficient.

## Validation

The FLOPs counter has been validated against:
- Manual calculations for simple cases
- PyTorch profiler measurements
- Theoretical complexity analysis

Accuracy: ±5% for total FLOPs count

## Examples

See `examples/flops_counter_demo.py` for comprehensive examples:
- Basic counting
- Component breakdown
- Model comparison
- Scaling analysis
- Optimizer comparison

## Testing

Run tests:
```bash
python -m pytest tests/test_flops_counter.py -v
```

All 18 tests pass, covering:
- FLOPsCount dataclass operations
- Individual component counting
- Total FLOPs counting
- Model comparison
- JSON export

## Implementation Details

### FLOPsCount Dataclass

```python
@dataclass
class FLOPsCount:
    forward: int = 0
    backward: int = 0
    optimizer: int = 0
    
    @property
    def total(self) -> int:
        return self.forward + self.backward + self.optimizer
```

### FLOPsCounter Class

Main methods:
- `count_bk_core_flops()`: BK-Core FLOPs
- `count_moe_flops()`: MoE layer FLOPs
- `count_linear_flops()`: Linear layer FLOPs
- `count_forward_flops()`: Total forward FLOPs
- `count_backward_flops()`: Total backward FLOPs
- `count_optimizer_flops()`: Optimizer step FLOPs
- `count_total_flops()`: All FLOPs combined
- `get_breakdown()`: Component-wise breakdown
- `print_summary()`: Human-readable summary
- `save_to_json()`: Export to JSON

## Future Enhancements

Potential improvements:
- [ ] GPU kernel profiling integration
- [ ] Memory bandwidth analysis
- [ ] Roofline model visualization
- [ ] Automatic optimization suggestions
- [ ] Real-time FLOPs monitoring during training
- [ ] Comparison with other architectures (Transformer, etc.)

## References

- FLOPs counting methodology: [Deep Learning FLOPs](https://arxiv.org/abs/2001.03343)
- ResNet-BK architecture: See `docs/STEP7_SYSTEM_INTEGRATION.md`
- BK-Core algorithm: See `src/models/bk_core.py`

## Contact

For questions or issues with the FLOPs counter, please open an issue on GitHub.
