# FLOPs Counter - Quick Reference

## Installation

No additional dependencies required. Uses standard PyTorch and Python libraries.

## Quick Start

```python
from src.models.configurable_resnet_bk import ConfigurableResNetBK, BASELINE_CONFIG
from src.benchmarks.flops_counter import FLOPsCounter

# Create model
config = BASELINE_CONFIG
config.d_model = 64
config.n_layers = 4
config.n_seq = 128
model = ConfigurableResNetBK(config)

# Count FLOPs
counter = FLOPsCounter(model, batch_size=32, seq_len=128)
counter.print_summary()
```

## Common Operations

### Get Total FLOPs

```python
flops = counter.count_total_flops()
print(f"Total: {flops.total:,} FLOPs ({flops.total/1e9:.3f} GFLOPs)")
print(f"Forward: {flops.forward:,} FLOPs")
print(f"Backward: {flops.backward:,} FLOPs)")
print(f"Optimizer: {flops.optimizer:,} FLOPs")
```

### Get Component Breakdown

```python
counter.count_forward_flops()
breakdown = counter.get_breakdown()

for component, flops in breakdown.items():
    print(f"{component}: {flops['forward']:,} FLOPs")
```

### Compare Models

```python
from src.benchmarks.flops_counter import compare_models

comparison = compare_models(
    model1, model2,
    batch_size=32, seq_len=128,
    model1_name="Model A",
    model2_name="Model B"
)

print(f"Speedup: {comparison['speedup']['total']:.2f}×")
```

### Save to JSON

```python
counter.save_to_json('flops_count.json')
```

## FLOPs Formulas (Quick Reference)

| Component | Forward FLOPs | Backward FLOPs |
|-----------|---------------|----------------|
| BK-Core | ~34N per sequence | ~26N per sequence |
| MoE | K×8D² per token | 2× forward |
| Linear(in, out) | B×N×in×out×2 | 2× forward |
| LayerNorm | B×N×D×5 | B×N×D×5 |
| Embedding | B×N×D | B×N×D |
| AdamW | - | 15 per param |

Where:
- B = batch size
- N = sequence length
- D = d_model
- K = top_k experts
- in, out = input/output dimensions

## Typical Results

### Baseline Model (d=64, L=4, N=128, batch=32)

```
Forward:   16.3 GFLOPs
Backward:  32.6 GFLOPs
Optimizer:  0.06 GFLOPs
Total:     48.9 GFLOPs
```

### Component Breakdown

```
LM Head:    96.6% (vocabulary size dominates)
Layers:      3.6% (4 layers × 0.9% each)
Other:      <0.1%
```

### Sequence Length Scaling

```
N=128  →  48.9 GFLOPs
N=256  →  97.8 GFLOPs (2× increase)
N=512  → 202.0 GFLOPs (4× increase)
N=1024 → 410.2 GFLOPs (8× increase)
N=2048 → 826.6 GFLOPs (16× increase)

Scaling: O(N) ✓ (linear with sequence length)
```

## Running Examples

```bash
# Run comprehensive demo
python examples/flops_counter_demo.py

# Run tests
python -m pytest tests/test_flops_counter.py -v
```

## Common Use Cases

### 1. Measure Training Cost

```python
counter = FLOPsCounter(model, batch_size=32, seq_len=128)
flops_per_step = counter.count_total_flops()
total_steps = 10000
total_flops = flops_per_step.total * total_steps
print(f"Total training cost: {total_flops/1e12:.2f} TFLOPs")
```

### 2. Compare Configurations

```python
baseline = ConfigurableResNetBK(BASELINE_CONFIG)
optimized = ConfigurableResNetBK(FULL_CONFIG)

comparison = compare_models(baseline, optimized, 32, 128)
speedup = comparison['speedup']['total']
print(f"Optimization provides {speedup:.2f}× speedup")
```

### 3. Analyze Bottlenecks

```python
counter.count_forward_flops()
breakdown = counter.get_breakdown()

# Sort by FLOPs
sorted_components = sorted(
    breakdown.items(),
    key=lambda x: x[1]['forward'],
    reverse=True
)

print("Top 3 bottlenecks:")
for component, flops in sorted_components[:3]:
    pct = 100 * flops['forward'] / counter.count_forward_flops().forward
    print(f"  {component}: {pct:.1f}%")
```

### 4. Validate O(N) Complexity

```python
seq_lengths = [128, 256, 512, 1024]
flops_list = []

for N in seq_lengths:
    config.n_seq = N
    model = ConfigurableResNetBK(config)
    counter = FLOPsCounter(model, batch_size=32, seq_len=N)
    flops = counter.count_total_flops()
    flops_list.append(flops.total)

# Check scaling
for i in range(1, len(seq_lengths)):
    ratio = flops_list[i] / flops_list[i-1]
    expected = seq_lengths[i] / seq_lengths[i-1]
    print(f"N={seq_lengths[i-1]}→{seq_lengths[i]}: {ratio:.2f}× (expected {expected:.2f}×)")
```

## Tips

1. **LM Head Dominates**: For large vocabularies, LM head accounts for >95% of FLOPs
   - Consider vocabulary reduction for faster training
   - Or use adaptive softmax

2. **Batch Size**: FLOPs scale linearly with batch size
   - Larger batches = more FLOPs but better GPU utilization

3. **Sequence Length**: ResNet-BK scales O(N), Transformer scales O(N²)
   - ResNet-BK advantage increases with longer sequences

4. **Optimizer Choice**: AdamW has ~7.5× more FLOPs than SGD
   - But optimizer FLOPs are negligible (<1% of total)

## Troubleshooting

### Issue: AttributeError when creating counter

**Solution**: Ensure model is ConfigurableResNetBK or LanguageModel
```python
# Correct
model = ConfigurableResNetBK(config)
counter = FLOPsCounter(model, batch_size=32, seq_len=128)
```

### Issue: FLOPs seem too high/low

**Solution**: Check batch size and sequence length
```python
# FLOPs scale with batch_size × seq_len
counter = FLOPsCounter(model, batch_size=1, seq_len=128)  # Minimal FLOPs
```

### Issue: Component breakdown empty

**Solution**: Call count_forward_flops() first
```python
counter.count_forward_flops()  # Populates breakdown
breakdown = counter.get_breakdown()
```

## Documentation

- Full documentation: `docs/FLOPS_COUNTER.md`
- Examples: `examples/flops_counter_demo.py`
- Tests: `tests/test_flops_counter.py`
- Implementation: `src/benchmarks/flops_counter.py`

## Contact

For issues or questions, please open a GitHub issue.
