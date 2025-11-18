# Performance Tuning Guide

Optimize ResNet-BK for maximum performance on your hardware.

---

## Table of Contents

1. [Quick Wins](#quick-wins)
2. [Training Optimization](#training-optimization)
3. [Inference Optimization](#inference-optimization)
4. [Memory Optimization](#memory-optimization)
5. [Hardware-Specific Tuning](#hardware-specific-tuning)
6. [Profiling and Debugging](#profiling-and-debugging)

---

## Quick Wins

### Enable TF32 (Ampere GPUs)

```python
import torch

# Enable TF32 for matmul and convolutions
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

**Expected speedup**: Potential 1.5-2× on A100/A6000

### Use torch.compile (PyTorch 2.0+)

```python
model = torch.compile(model, mode='max-autotune')
```

**Expected speedup**: Potential 1.3-1.8×

### Increase Batch Size

```yaml
# configs/base_config.yaml
batch_size: 32  # Increase until OOM
```

**Expected speedup**: Scales with batch size (up to hardware limit)

### Enable Mixed Precision

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in train_loader:
    with autocast():
        output = model(batch)
        loss = criterion(output, target)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**Expected speedup**: Potential 1.5-2× with minimal accuracy loss

---

## Training Optimization

### Gradient Accumulation

Simulate larger batch sizes without OOM:

```python
accumulation_steps = 4
optimizer.zero_grad()

for i, batch in enumerate(train_loader):
    output = model(batch)
    loss = criterion(output, target) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**Benefit**: Effective batch size = batch_size × accumulation_steps

### Gradient Checkpointing

Trade compute for memory:

```python
from src.models.memory_optimization import enable_gradient_checkpointing

enable_gradient_checkpointing(model)
```

**Trade-off**: 
- Memory: -40%
- Speed: -20%

### Optimal Learning Rate

Use learning rate finder:

```python
from src.training.dynamic_lr_scheduler import LRFinder

lr_finder = LRFinder(model, optimizer, criterion)
lr_finder.range_test(train_loader, start_lr=1e-7, end_lr=1)
lr_finder.plot()
optimal_lr = lr_finder.suggest_lr()
```

### Data Loading Optimization

```python
train_loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=8,  # CPU cores - 1
    pin_memory=True,  # Faster GPU transfer
    persistent_workers=True,  # Keep workers alive
    prefetch_factor=2,  # Prefetch batches
)
```

### Distributed Training

Multi-GPU training with DDP:

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize process group
dist.init_process_group(backend='nccl')

# Wrap model
model = DDP(model, device_ids=[local_rank])

# Train normally
for batch in train_loader:
    output = model(batch)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
```

**Expected speedup**: Scales with GPU count

---

## Inference Optimization

### KV Caching

Cache key-value pairs for autoregressive generation:

```python
# Enable caching
output = model.generate(
    input_ids,
    max_length=100,
    use_cache=True  # Enable KV caching
)
```

**Expected speedup**: Potential 2-5× for generation

### Quantization

Reduce precision for faster inference:

```python
from src.models.quantized_bk_core import quantize_model

# INT8 quantization
model_int8 = quantize_model(model, bits=8)

# INT4 quantization (more aggressive)
model_int4 = quantize_model(model, bits=4)
```

**Trade-off**:
- INT8: 2× faster, minimal accuracy loss
- INT4: 4× faster, some accuracy loss

### Batch Inference

Process multiple sequences simultaneously:

```python
# Instead of processing one by one
for input_ids in inputs:
    output = model(input_ids)

# Batch process
batch_input_ids = torch.stack(inputs)
batch_outputs = model(batch_input_ids)
```

**Expected speedup**: Potential 3-5× for batch size 32

### ONNX Export

Export to ONNX for optimized inference:

```python
from src.models.onnx_export import export_to_onnx

export_to_onnx(
    model,
    'model.onnx',
    input_shape=(1, 512),
    opset_version=14
)

# Load with ONNX Runtime
import onnxruntime as ort
session = ort.InferenceSession('model.onnx')
output = session.run(None, {'input': input_array})
```

**Expected speedup**: Potential 1.5-2× on CPU, 1.2-1.5× on GPU

### TensorRT Optimization

For NVIDIA GPUs:

```python
import torch_tensorrt

# Compile with TensorRT
trt_model = torch_tensorrt.compile(
    model,
    inputs=[torch_tensorrt.Input((1, 512), dtype=torch.long)],
    enabled_precisions={torch.float16}
)

# Inference
output = trt_model(input_ids)
```

**Expected speedup**: Potential 2-4× on NVIDIA GPUs

---

## Memory Optimization

### CPU Offloading

Offload parameters to CPU:

```python
from src.models.memory_optimization import enable_cpu_offloading

enable_cpu_offloading(
    model,
    offload_ratio=0.5  # Offload 50% of parameters
)
```

**Trade-off**:
- Memory: -50%
- Speed: -30%

### Activation Checkpointing

Recompute activations instead of storing:

```python
from torch.utils.checkpoint import checkpoint

def forward_with_checkpointing(x):
    return checkpoint(model.layer, x)
```

**Trade-off**:
- Memory: -40%
- Speed: -20%

### Semiseparable Structure

Use memory-efficient semiseparable matrices:

```python
config = ResNetBKConfig(
    use_semiseparable=True,
    semiseparable_rank=64  # Lower rank = less memory
)
```

**Memory savings**: 70% vs dense attention

### ZeRO Optimization

Use DeepSpeed ZeRO for large models:

```python
from deepspeed import initialize

model_engine, optimizer, _, _ = initialize(
    model=model,
    optimizer=optimizer,
    config={
        "zero_optimization": {
            "stage": 2,  # Stage 2 or 3
        }
    }
)
```

**Memory savings**: 
- Stage 1: 4× reduction
- Stage 2: 8× reduction
- Stage 3: 16× reduction

---

## Hardware-Specific Tuning

### NVIDIA A100

```python
# Optimal settings for A100
config = {
    'batch_size': 64,
    'use_tf32': True,
    'use_amp': True,
    'num_workers': 16,
}

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

### NVIDIA V100

```python
# Optimal settings for V100
config = {
    'batch_size': 32,
    'use_amp': True,
    'num_workers': 8,
}
```

### NVIDIA T4

```python
# Optimal settings for T4 (Colab)
config = {
    'batch_size': 16,
    'use_amp': True,
    'gradient_checkpointing': True,
    'num_workers': 2,
}
```

### AMD MI250X

```python
# Optimal settings for AMD GPUs
config = {
    'batch_size': 48,
    'use_amp': True,
    'num_workers': 12,
}
```

### CPU-Only

```python
# Optimal settings for CPU
config = {
    'batch_size': 8,
    'num_workers': os.cpu_count() - 1,
    'use_mkldnn': True,
}

# Enable MKL-DNN
torch.backends.mkldnn.enabled = True
```

---

## Profiling and Debugging

### PyTorch Profiler

Profile your code to find bottlenecks:

```python
import torch.profiler as profiler

with profiler.profile(
    activities=[
        profiler.ProfilerActivity.CPU,
        profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    for _ in range(10):
        output = model(input_ids)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

# Print results
print(prof.key_averages().table(
    sort_by="cuda_time_total",
    row_limit=10
))

# Export to Chrome trace
prof.export_chrome_trace("trace.json")
```

### Memory Profiler

Track memory usage:

```python
import torch

torch.cuda.reset_peak_memory_stats()

# Your code here
output = model(input_ids)

peak_memory = torch.cuda.max_memory_allocated() / 1e9
print(f"Peak memory: {peak_memory:.2f} GB")
```

### FLOPs Counter

Measure computational cost:

```python
from src.benchmarks.flops_counter import count_flops

flops = count_flops(model, input_ids)
print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")
```

### Benchmark Script

Run comprehensive benchmarks:

```bash
python scripts/benchmarks/run_performance_benchmark.py \
    --model resnet_bk \
    --batch-sizes 1,4,16,32 \
    --seq-lengths 512,1024,2048 \
    --device cuda
```

---

## Performance Checklist

Before deploying to production:

- [ ] Enable TF32 (if using Ampere GPU)
- [ ] Use torch.compile (PyTorch 2.0+)
- [ ] Enable mixed precision training
- [ ] Optimize batch size
- [ ] Use gradient accumulation if needed
- [ ] Enable gradient checkpointing for large models
- [ ] Optimize data loading (num_workers, pin_memory)
- [ ] Profile code to find bottlenecks
- [ ] Use quantization for inference
- [ ] Enable KV caching for generation
- [ ] Consider ONNX/TensorRT export
- [ ] Test on target hardware

---

## Performance Targets

### Training (A100 GPU)

| Model Size | Batch Size | Seq Length | Throughput | Memory |
|------------|------------|------------|------------|--------|
| 100M | 64 | 2048 | 50k tok/s | 20 GB |
| 1B | 32 | 2048 | 25k tok/s | 40 GB |
| 10B | 8 | 2048 | 6k tok/s | 70 GB |

### Inference (A100 GPU)

| Model Size | Batch Size | Latency | Throughput |
|------------|------------|---------|------------|
| 100M | 1 | 10 ms | 100 tok/s |
| 1B | 1 | 30 ms | 33 tok/s |
| 10B | 1 | 100 ms | 10 tok/s |

---

## Advanced Optimization

### Custom CUDA Kernels

For maximum performance, implement custom CUDA kernels:

```python
from src.models.cuda_bk_core import CUDABKCore

# Use optimized CUDA implementation
model = CUDABKCore(n_seq=2048, epsilon=1.0)
```

**Expected speedup**: Potential 2-5× over PyTorch implementation

### Triton Kernels

Use Triton for easier kernel development:

```python
import triton
import triton.language as tl

@triton.jit
def custom_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Your kernel code here
    pass
```

### Flash Attention

Use Flash Attention for efficient attention:

```python
from flash_attn import flash_attn_func

# Replace standard attention
output = flash_attn_func(q, k, v)
```

**Expected speedup**: Potential 2-4× for attention computation

---

## Questions?

For performance-related questions:
- Check [FAQ.md](docs/FAQ.md)
- Ask in [GitHub Discussions](https://github.com/neko-jpg/Project-ResNet-BK-An-O-N-Language-Model-Architecture/discussions)
- Email: arat252539@gmail.com

---

**Last Updated**: 2025-01-15
