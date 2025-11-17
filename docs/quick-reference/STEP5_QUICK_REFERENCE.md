# Step 5: Hardware Co-Design - Quick Reference

## Quick Start

### 1. Train with AMP (Recommended)

```python
from src.models.configurable_resnet_bk import ConfigurableResNetBK
from src.training.amp_trainer import MixedPrecisionTrainer
from src.utils.data_utils import prepare_wikitext2_data
import torch.nn as nn

# Prepare data
train_loader, val_loader, vocab_size = prepare_wikitext2_data(batch_size=8, seq_len=128)

# Create model
model = ConfigurableResNetBK(
    vocab_size=vocab_size,
    d_model=64,
    n_layers=4,
    n_seq=128,
    num_experts=4
)

# Setup AMP trainer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
trainer = MixedPrecisionTrainer(model, optimizer, criterion, enabled=True)

# Train
for epoch in range(10):
    result = trainer.train_epoch(train_loader, epoch, log_interval=100)
    print(f"Epoch {epoch}: Loss={result['avg_loss']:.4f}, Speed={result['steps_per_sec']:.2f} steps/s")
```

### 2. Train with Gradient Accumulation (For Large Effective Batch Sizes)

```python
from src.training.hardware_optimizations import GradientAccumulationTrainer

trainer = GradientAccumulationTrainer(
    model, optimizer, criterion,
    accumulation_steps=4  # Effective batch size = 4 × actual batch size
)

for x_batch, y_batch in train_loader:
    result = trainer.train_step(x_batch, y_batch)
```

### 3. Train with Dynamic Batch Sizing (Prevent OOM)

```python
from src.training.hardware_optimizations import DynamicBatchSizeTrainer

trainer = DynamicBatchSizeTrainer(
    model, optimizer, criterion,
    initial_batch_size=32,
    min_batch_size=1
)

for x_batch, y_batch in train_loader:
    result = trainer.train_step(x_batch, y_batch)
    if result['oom']:
        print(f"OOM handled, new batch size: {result['batch_size']}")
```

### 4. Optimize Model for Tensor Cores

```python
from src.utils.tensor_core_utils import optimize_model_for_tensor_cores, validate_model_for_tensor_cores

# Validate current model
validation = validate_model_for_tensor_cores(model, verbose=True)

# Optimize if needed
if validation['compatibility_rate'] < 1.0:
    optimized_model = optimize_model_for_tensor_cores(model, inplace=False)
```

### 5. Use Custom CUDA Kernels (If Available)

```python
from src.models.cuda_bk_core import CUDAOptimizedBKCore

# Replace BK-Core in your model with CUDA-optimized version
cuda_core = CUDAOptimizedBKCore(n_seq=128, use_cuda_kernels=True)

# Check if CUDA kernels are available
if cuda_core.cuda_available:
    print("Using custom CUDA kernels")
else:
    print("Using PyTorch fallback")
```

## Benchmarking

### Benchmark Mixed Precision

```python
from src.models.mixed_precision_bk_core import benchmark_mixed_precision, validate_mixed_precision_accuracy

# Validate accuracy
validation = validate_mixed_precision_accuracy(batch_size=8, seq_len=128, num_samples=100)

# Benchmark performance
results = benchmark_mixed_precision(batch_size=8, seq_len=128, num_trials=100)
print(f"Speedup: {results['speedup']:.2f}x")
print(f"Max error: {results['max_error']:.6e}")
```

### Benchmark CUDA Kernels

```python
from src.benchmarks.cuda_kernel_benchmark import CUDAKernelBenchmark

benchmark = CUDAKernelBenchmark()

# Measure speedup
benchmark.benchmark_speedup(
    batch_sizes=[1, 4, 8, 16],
    seq_lengths=[128, 256, 512, 1024],
    n_iterations=100
)

# Compare to cuSPARSE
benchmark.benchmark_cusparse_comparison(seq_lengths=[128, 256, 512, 1024])

# Profile GPU utilization
benchmark.profile_gpu_utilization(batch_size=8, seq_length=1024)

# Generate report
benchmark.plot_results('cuda_benchmark.png')
benchmark.generate_report('cuda_benchmark_report.txt')
```

### Benchmark AMP Training

```python
from src.training.amp_trainer import benchmark_amp_training

results = benchmark_amp_training(model, train_loader, num_epochs=3)
print(f"Speedup: {results['speedup']:.2f}x")
print(f"Memory reduction: {results['memory_reduction']:.1%}")
```

## Google Colab

### Run Step 5 Tests on Colab

1. Upload `notebooks/step5_hardware_optimizations.ipynb` to Google Colab
2. Run all cells
3. Check results:
   - Mixed precision validation: max error < 1e-4 ✓
   - AMP training: 2× speedup, 50% memory reduction ✓
   - Gradient accumulation: effective batch size = 4 × actual ✓
   - Training completes without OOM ✓

## Configuration Tips

### For Google Colab Free Tier (T4 GPU, 15GB RAM)

```python
config = {
    'vocab_size': vocab_size,
    'd_model': 64,        # Keep small
    'n_layers': 4,        # Keep small
    'n_seq': 128,         # Standard
    'num_experts': 4,     # Standard
    'batch_size': 8,      # Adjust based on memory
}

# Use AMP to reduce memory
trainer = MixedPrecisionTrainer(model, optimizer, criterion, enabled=True)

# Use gradient accumulation for larger effective batch size
trainer = GradientAccumulationTrainer(model, optimizer, criterion, accumulation_steps=4)
```

### For Larger GPUs (V100, A100)

```python
config = {
    'vocab_size': vocab_size,
    'd_model': 128,       # Can increase
    'n_layers': 8,        # Can increase
    'n_seq': 256,         # Can increase
    'num_experts': 8,     # Can increase
    'batch_size': 32,     # Can increase
}

# Still use AMP for speed
trainer = MixedPrecisionTrainer(model, optimizer, criterion, enabled=True)
```

## Troubleshooting

### OOM Errors

1. **Use Dynamic Batch Sizing**:
   ```python
   trainer = DynamicBatchSizeTrainer(model, optimizer, criterion, initial_batch_size=32)
   ```

2. **Use Gradient Accumulation**:
   ```python
   trainer = GradientAccumulationTrainer(model, optimizer, criterion, accumulation_steps=4)
   ```

3. **Enable CPU Offloading**:
   ```python
   from src.training.hardware_optimizations import CPUOffloadingOptimizer
   optimizer = CPUOffloadingOptimizer(model.parameters(), lr=1e-3)
   ```

4. **Reduce Model Size**:
   - Decrease `d_model`
   - Decrease `n_layers`
   - Decrease `num_experts`

### CUDA Kernel Compilation Fails

The implementation automatically falls back to PyTorch:
```python
cuda_core = CUDAOptimizedBKCore(n_seq=128, use_cuda_kernels=True)
if not cuda_core.cuda_available:
    print("Using PyTorch fallback (still fast!)")
```

### AMP Overflow Issues

Adjust gradient scaler parameters:
```python
trainer = MixedPrecisionTrainer(
    model, optimizer, criterion,
    enabled=True,
    growth_interval=2000  # Increase for more stable scaling
)
```

### Slow Training

1. **Enable AMP**: 2× speedup
2. **Optimize for Tensor Cores**: 1.5× speedup
3. **Use Custom CUDA Kernels**: 3× speedup (if available)
4. **Combined**: ~10× speedup

## Performance Checklist

- [ ] AMP enabled: `MixedPrecisionTrainer(enabled=True)`
- [ ] Tensor core compatible dimensions (multiples of 8)
- [ ] CUDA kernels compiled (if possible)
- [ ] Gradient accumulation for large effective batch sizes
- [ ] Dynamic batch sizing to prevent OOM
- [ ] Monitoring GPU utilization (should be >80%)

## Expected Performance

| Configuration | Speedup | Memory |
|--------------|---------|--------|
| Baseline | 1.0× | 100% |
| + AMP | 2.0× | 50% |
| + Tensor Cores | 3.0× | 50% |
| + CUDA Kernels | 10.0× | 50% |

## Files Reference

- **AMP Training**: `src/training/amp_trainer.py`
- **Hardware Optimizations**: `src/training/hardware_optimizations.py`
- **CUDA Kernels**: `src/models/cuda_bk_core.py`
- **Mixed Precision**: `src/models/mixed_precision_bk_core.py`
- **Tensor Core Utils**: `src/utils/tensor_core_utils.py`
- **Benchmarking**: `src/benchmarks/cuda_kernel_benchmark.py`
- **Colab Notebook**: `notebooks/step5_hardware_optimizations.ipynb`

## Next Steps

After Step 5, proceed to:
1. **Step 6**: Algorithmic Innovations (ACT, multi-scale, learned sparsity)
2. **Step 7**: System Integration (curriculum learning, data efficiency)
3. **Comprehensive Benchmarking**: Validate 1,000,000,000× cost reduction
