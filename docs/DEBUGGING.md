# Debugging Guide

This guide helps you diagnose and fix common issues with ResNet-BK.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Training Issues](#training-issues)
- [Numerical Stability Issues](#numerical-stability-issues)
- [Memory Issues](#memory-issues)
- [Performance Issues](#performance-issues)
- [Quantization Issues](#quantization-issues)
- [Long-Context Issues](#long-context-issues)
- [Debugging Tools](#debugging-tools)

---

## Installation Issues

### Issue: `pip install resnet-bk` fails

**Symptoms:**
```
ERROR: Could not find a version that satisfies the requirement resnet-bk
```

**Solutions:**

1. **Update pip:**
```bash
pip install --upgrade pip
```

2. **Check Python version:**
```bash
python --version  # Should be 3.8+
```

3. **Install from source:**
```bash
git clone https://github.com/neko-jpg/Project-ResNet-BK-An-O-N-Language-Model-Architecture.git
cd Project-ResNet-BK-An-O-N-Language-Model-Architecture
pip install -e .
```

### Issue: CUDA not available

**Symptoms:**
```python
>>> import torch
>>> torch.cuda.is_available()
False
```

**Solutions:**

1. **Check CUDA installation:**
```bash
nvidia-smi
```

2. **Install PyTorch with CUDA:**
```bash
# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

3. **Verify CUDA version matches:**
```python
import torch
print(torch.version.cuda)  # Should match your CUDA version
```

### Issue: Import errors

**Symptoms:**
```python
ImportError: cannot import name 'ResNetBK' from 'src.models.resnet_bk'
```

**Solutions:**

1. **Check installation:**
```bash
pip show resnet-bk
```

2. **Reinstall:**
```bash
pip uninstall resnet-bk
pip install resnet-bk
```

3. **Check Python path:**
```python
import sys
print(sys.path)
```

---

## Training Issues

### Issue: Training loss is NaN

**Symptoms:**
```
Epoch 1, Step 100: loss = nan
```

**Diagnosis:**

1. **Check for NaN in inputs:**
```python
import torch

def check_nan(tensor, name):
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name}")
        return True
    return False

# In training loop
check_nan(input_ids, "input_ids")
check_nan(loss, "loss")
```

2. **Check gradient norms:**
```python
total_norm = 0
for p in model.parameters():
    if p.grad is not None:
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
total_norm = total_norm ** 0.5
print(f"Gradient norm: {total_norm}")
```

**Solutions:**

1. **Reduce learning rate:**
```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # Instead of 1e-3
```

2. **Enable gradient clipping:**
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

3. **Use automatic recovery:**
```python
from src.training.auto_recovery import AutoRecovery

recovery = AutoRecovery(checkpoint_dir='checkpoints')
if recovery.detect_failure(training_state):
    recovery.recover('nan_detected', model, optimizer)
```

4. **Increase epsilon:**
```python
model = ResNetBK(epsilon=1.0)  # Instead of 0.5
```

### Issue: Training is very slow

**Symptoms:**
- Training takes much longer than expected
- GPU utilization is low

**Diagnosis:**

1. **Profile the code:**
```python
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    output = model(input_ids)
    loss = criterion(output, labels)
    loss.backward()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

2. **Check GPU utilization:**
```bash
nvidia-smi -l 1  # Monitor every second
```

**Solutions:**

1. **Increase batch size:**
```python
batch_size = 16  # Instead of 8
```

2. **Enable mixed precision:**
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    output = model(input_ids)
    loss = criterion(output, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

3. **Use gradient checkpointing:**
```python
model = ResNetBK(gradient_checkpointing=True)
```

4. **Optimize data loading:**
```python
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=4,  # Use multiple workers
    pin_memory=True,  # Pin memory for faster transfer
    prefetch_factor=2  # Prefetch batches
)
```

### Issue: Loss not decreasing

**Symptoms:**
- Loss stays constant or increases
- Model not learning

**Diagnosis:**

1. **Check learning rate:**
```python
for param_group in optimizer.param_groups:
    print(f"Learning rate: {param_group['lr']}")
```

2. **Check if gradients are flowing:**
```python
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm = {param.grad.norm().item()}")
    else:
        print(f"{name}: no gradient")
```

**Solutions:**

1. **Increase learning rate:**
```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # Instead of 1e-4
```

2. **Use learning rate warmup:**
```python
from src.training.dynamic_lr_scheduler import DynamicLRScheduler

scheduler = DynamicLRScheduler(optimizer, warmup_steps=1000)
```

3. **Check data preprocessing:**
```python
# Verify labels are correct
print(f"Labels: {labels}")
print(f"Unique labels: {torch.unique(labels)}")
```

4. **Try different initialization:**
```python
model = ResNetBK(use_prime_bump=True)  # Use Prime-Bump initialization
```

---

## Numerical Stability Issues

### Issue: Schatten norm exceeds bounds

**Symptoms:**
```
WARNING: Schatten norm ||K||_S2 = 150.5 exceeds bound 100.0
```

**Diagnosis:**

1. **Monitor Schatten norms:**
```python
from src.models.birman_schwinger_core import BirmanSchwingerCore

core = BirmanSchwingerCore(n_seq=2048, epsilon=1.0)
s1_norm, s2_norm = core.compute_schatten_norms()
print(f"||K||_S1 = {s1_norm}, ||K||_S2 = {s2_norm}")
```

**Solutions:**

1. **Apply spectral clipping:**
```python
core.apply_spectral_clipping(threshold=100.0)
```

2. **Increase epsilon:**
```python
model = ResNetBK(epsilon=1.0)  # Instead of 0.5
```

3. **Use automatic precision upgrade:**
```python
model = ResNetBK(precision_upgrade_threshold=1e6)
```

### Issue: Condition number too high

**Symptoms:**
```
WARNING: Condition number κ = 1.5e7 exceeds threshold 1e6
```

**Diagnosis:**

1. **Check condition number:**
```python
from src.training.stability_monitor import StabilityMonitor

monitor = StabilityMonitor()
kappa = monitor.check_condition_number(H)
print(f"Condition number: {kappa}")
```

**Solutions:**

1. **Upgrade to complex128:**
```python
model = ResNetBK(use_complex128=True)
```

2. **Use LAP-based stability:**
```python
model = ResNetBK(use_lap=True)
```

3. **Reduce sequence length:**
```python
model = ResNetBK(n_seq=1024)  # Instead of 2048
```

### Issue: Mourre estimate fails

**Symptoms:**
```
ERROR: Mourre estimate verification failed: [H_0, iA] ≠ I
```

**Diagnosis:**

1. **Verify Mourre estimate:**
```python
from src.models.mourre_lap import verify_mourre_estimate

is_valid = verify_mourre_estimate(H_0, A)
print(f"Mourre estimate valid: {is_valid}")
```

**Solutions:**

1. **Check operator construction:**
```python
# Ensure H_0 is correctly constructed
H_0 = construct_free_hamiltonian(n_seq)
```

2. **Use numerical tolerance:**
```python
is_valid = verify_mourre_estimate(H_0, A, tol=1e-5)
```

---

## Memory Issues

### Issue: Out of Memory (OOM)

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Diagnosis:**

1. **Check memory usage:**
```python
import torch

print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

2. **Profile memory:**
```python
from src.models.memory_optimization import MemoryProfiler

profiler = MemoryProfiler(model)
breakdown = profiler.profile()
print(breakdown)
```

**Solutions:**

1. **Reduce batch size:**
```python
batch_size = 4  # Instead of 8
```

2. **Enable gradient checkpointing:**
```python
model = ResNetBK(gradient_checkpointing=True)
```

3. **Use semiseparable structure:**
```python
model = ResNetBK(use_semiseparable=True)
```

4. **Enable CPU offloading:**
```python
from src.models.memory_optimization import MemoryOptimizer

optimizer = MemoryOptimizer(use_cpu_offload=True)
model = optimizer.optimize(model)
```

5. **Use ZeRO optimizer:**
```python
from src.training.distributed_optimizations import setup_zero

model, optimizer = setup_zero(model, optimizer, stage=1)
```

6. **Clear cache:**
```python
torch.cuda.empty_cache()
```

### Issue: Memory leak

**Symptoms:**
- Memory usage increases over time
- Eventually runs out of memory

**Diagnosis:**

1. **Track memory over time:**
```python
import gc
import torch

for epoch in range(num_epochs):
    # Training loop
    ...
    
    # Check memory
    print(f"Epoch {epoch}: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    # Force garbage collection
    gc.collect()
    torch.cuda.empty_cache()
```

**Solutions:**

1. **Detach tensors:**
```python
loss = loss.detach()  # Detach from computation graph
```

2. **Delete unused variables:**
```python
del output, loss
torch.cuda.empty_cache()
```

3. **Use context managers:**
```python
with torch.no_grad():
    # Evaluation code
    ...
```

---

## Performance Issues

### Issue: Slow inference

**Symptoms:**
- Inference takes much longer than expected
- Low throughput (tokens/second)

**Diagnosis:**

1. **Benchmark inference:**
```python
import time
import torch

model.eval()
with torch.no_grad():
    start = time.time()
    for _ in range(100):
        output = model(input_ids)
    end = time.time()

tokens_per_second = (100 * input_ids.size(1)) / (end - start)
print(f"Throughput: {tokens_per_second:.2f} tokens/s")
```

**Solutions:**

1. **Use TensorRT:**
```python
from src.models.onnx_export import export_to_tensorrt

tensorrt_model = export_to_tensorrt(model, 'model.trt')
```

2. **Use ONNX:**
```python
from src.models.onnx_export import export_to_onnx

export_to_onnx(model, 'model.onnx')
```

3. **Enable torch.compile:**
```python
model = torch.compile(model, mode='max-autotune')
```

4. **Use mixed precision:**
```python
model = model.half()  # Convert to FP16
```

### Issue: Slow routing

**Symptoms:**
- MoE routing is bottleneck
- High routing overhead

**Diagnosis:**

1. **Profile routing:**
```python
from src.benchmarks.flops_counter import FLOPsCounter

counter = FLOPsCounter(model)
with counter:
    output = model(input_ids)

print(f"Routing FLOPs: {counter.routing_flops}")
print(f"Total FLOPs: {counter.total_flops}")
```

**Solutions:**

1. **Use scattering-based routing:**
```python
model = ResNetBK(use_scattering_router=True)
```

2. **Reduce number of experts:**
```python
model = ResNetBK(num_experts=4)  # Instead of 8
```

---

## Quantization Issues

### Issue: High quantization error

**Symptoms:**
- Perplexity increases significantly after quantization
- Model performance degrades

**Diagnosis:**

1. **Measure quantization error:**
```python
from src.models.quantized_birman_schwinger import QuantizedBirmanSchwinger

# Original model
ppl_fp32 = evaluate(model, dataset)

# Quantized model
quantized_model = QuantizedBirmanSchwinger(model, bits=8)
ppl_int8 = evaluate(quantized_model, dataset)

print(f"FP32 PPL: {ppl_fp32:.2f}")
print(f"INT8 PPL: {ppl_int8:.2f}")
print(f"Degradation: {(ppl_int8 - ppl_fp32) / ppl_fp32 * 100:.1f}%")
```

**Solutions:**

1. **Use quantization-aware training:**
```python
from src.models.quantized_birman_schwinger import QuantizationAwareTraining

qat = QuantizationAwareTraining(model, bits=8)
qat.train(dataloader, num_epochs=5)
```

2. **Use mixed-precision quantization:**
```python
from src.models.mixed_precision_quantization import MixedPrecisionQuantization

quant_config = {
    'experts': 4,
    'bk_core': 8,
    'output': 16
}
mixed_quant = MixedPrecisionQuantization(model, quant_config)
```

3. **Use group-wise quantization:**
```python
quantized_model = QuantizedBirmanSchwinger(
    model,
    bits=4,
    group_size=128
)
```

---

## Long-Context Issues

### Issue: Divergence at long sequences

**Symptoms:**
- Training diverges at N > 32k
- Loss becomes NaN or infinity

**Diagnosis:**

1. **Monitor gradient norms:**
```python
from src.training.stability_monitor import StabilityMonitor

monitor = StabilityMonitor()
for step in range(num_steps):
    loss.backward()
    grad_norm = monitor.check_gradient_norm(model)
    print(f"Step {step}: grad_norm = {grad_norm}")
```

**Solutions:**

1. **Use semiseparable structure:**
```python
model = ResNetBK(use_semiseparable=True)
```

2. **Use hierarchical structure:**
```python
from src.models.semiseparable_matrix import HierarchicalSemiseparable

model = ResNetBK(use_hierarchical_semiseparable=True)
```

3. **Increase epsilon:**
```python
model = ResNetBK(epsilon=1.0)
```

4. **Use streaming evaluation:**
```python
from src.benchmarks.streaming_evaluator import StreamingEvaluator

evaluator = StreamingEvaluator(model, chunk_size=8192)
ppl = evaluator.evaluate(dataset, max_length=131072)
```

---

## Debugging Tools

### 1. Stability Monitor

Real-time monitoring of numerical health:

```python
from src.training.stability_monitor import StabilityMonitor

monitor = StabilityMonitor()

# In training loop
health = monitor.check_all(model, loss, optimizer)
if not health['is_healthy']:
    print(f"Unhealthy: {health['issues']}")
    # Take corrective action
```

### 2. Memory Profiler

Detailed memory breakdown:

```python
from src.models.memory_optimization import MemoryProfiler

profiler = MemoryProfiler(model)
breakdown = profiler.profile()

print(f"Model weights: {breakdown['weights']:.2f} GB")
print(f"Activations: {breakdown['activations']:.2f} GB")
print(f"Optimizer: {breakdown['optimizer']:.2f} GB")
```

### 3. FLOPs Counter

Accurate FLOPs measurement:

```python
from src.benchmarks.flops_counter import FLOPsCounter

counter = FLOPsCounter(model)
with counter:
    output = model(input_ids)

print(f"Forward FLOPs: {counter.forward_flops}")
print(f"Backward FLOPs: {counter.backward_flops}")
print(f"Total FLOPs: {counter.total_flops}")
```

### 4. Visualization Tools

Visualize training dynamics:

```python
from src.utils.visualization import plot_training_curves

plot_training_curves(
    losses=losses,
    grad_norms=grad_norms,
    schatten_norms=schatten_norms,
    save_path='training_curves.png'
)
```

### 5. Interactive Dashboard

Real-time monitoring dashboard:

```python
from scripts.interactive_dashboard import launch_dashboard

launch_dashboard(
    model=model,
    dataloader=dataloader,
    port=8080
)
# Open http://localhost:8080 in browser
```

---

## Getting Help

If you can't resolve the issue:

1. **Check FAQ:** [FAQ.md](FAQ.md)
2. **Search issues:** [GitHub Issues](https://github.com/neko-jpg/Project-ResNet-BK-An-O-N-Language-Model-Architecture/issues)
3. **Ask in discussions:** [GitHub Discussions](https://github.com/neko-jpg/Project-ResNet-BK-An-O-N-Language-Model-Architecture/discussions)
4. **Join Discord:** [Discord Server](https://discord.gg/resnet-bk)

When reporting issues, include:
- ResNet-BK version
- PyTorch version
- CUDA version
- Full error message and stack trace
- Minimal reproducible example
- System information (OS, GPU, RAM)
