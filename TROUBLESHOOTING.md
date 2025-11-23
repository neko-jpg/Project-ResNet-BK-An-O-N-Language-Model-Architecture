# Troubleshooting Guide

Common issues and solutions for ResNet-BK.

---

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Training Issues](#training-issues)
3. [Memory Issues](#memory-issues)
4. [Performance Issues](#performance-issues)
5. [Numerical Stability Issues](#numerical-stability-issues)
6. [GPU Issues](#gpu-issues)
7. [Data Loading Issues](#data-loading-issues)
8. [Model Loading Issues](#model-loading-issues)

---

## Installation Issues

### Issue: `pip install` fails with dependency conflicts

**Symptoms**:
```
ERROR: Cannot install package due to conflicting dependencies
```

**Solutions**:

1. **Use a clean virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -e .
```

2. **Install PyTorch first**:
```bash
pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cu118
pip install -e .
```

3. **Use specific versions**:
```bash
pip install -r requirements.txt
pip install -e .
```

### Issue: CUDA not available

**Symptoms**:
```python
>>> import torch
>>> torch.cuda.is_available()
False
```

**Solutions**:

1. **Check CUDA installation**:
```bash
nvidia-smi
```

2. **Install correct PyTorch version**:
```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

3. **Verify installation**:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")
```

---

## Training Issues

### Issue: Loss becomes NaN

**Symptoms**:
```
Step 100: loss = nan
```

**Solutions**:

1. **Reduce learning rate**:
```yaml
# In config file
lr: 1e-4  # Instead of 1e-3
```

2. **Enable gradient clipping**:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

3. **Check for numerical instability**:
```python
# Enable stability diagnostics
model.enable_stability_monitoring()
diagnostics = model.get_stability_diagnostics()
print(diagnostics)
```

4. **Use mixed precision carefully**:
```python
# Disable AMP if causing issues
use_amp = False
```

### Issue: Training is very slow

**Symptoms**:
- Training takes much longer than expected
- Low GPU utilization

**Solutions**:

1. **Increase batch size**:
```yaml
batch_size: 32  # Instead of 8
```

2. **Enable gradient accumulation**:
```python
accumulation_steps = 4
```

3. **Use mixed precision**:
```python
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    output = model(input)
    loss = criterion(output, target)
```

4. **Profile the code**:
```python
import torch.profiler as profiler

with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
    record_shapes=True
) as prof:
    model(input)

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

### Issue: Model diverges during training

**Symptoms**:
- Loss increases instead of decreasing
- Perplexity explodes

**Solutions**:

1. **Check learning rate**:
```python
# Use learning rate warmup
from torch.optim.lr_scheduler import LinearLR, SequentialLR

warmup = LinearLR(optimizer, start_factor=0.1, total_iters=1000)
main = CosineAnnealingLR(optimizer, T_max=10000)
scheduler = SequentialLR(optimizer, [warmup, main], milestones=[1000])
```

2. **Verify data preprocessing**:
```python
# Check for invalid tokens
assert (input_ids >= 0).all()
assert (input_ids < vocab_size).all()
```

3. **Enable stability monitoring**:
```python
from src.training.stability_monitor import StabilityMonitor

monitor = StabilityMonitor(model)
monitor.check_stability()
```

---

## Memory Issues

### Issue: Out of memory (OOM)

**Symptoms**:
```
RuntimeError: CUDA out of memory
```

**Solutions**:

1. **Reduce batch size**:
```yaml
batch_size: 4  # Instead of 16
```

2. **Reduce sequence length**:
```yaml
n_seq: 1024  # Instead of 2048
```

3. **Enable gradient checkpointing**:
```python
model.enable_gradient_checkpointing()
```

4. **Use CPU offloading**:
```python
from src.models.memory_optimization import enable_cpu_offloading
enable_cpu_offloading(model)
```

5. **Clear cache**:
```python
import torch
torch.cuda.empty_cache()
```

6. **Use smaller model**:
```python
# Use fewer layers or smaller dimension
config.n_layers = 6  # Instead of 12
config.d_model = 256  # Instead of 512
```

### Issue: Memory leak during training

**Symptoms**:
- Memory usage increases over time
- Eventually runs out of memory

**Solutions**:

1. **Detach tensors**:
```python
loss_value = loss.detach().item()  # Not just loss.item()
```

2. **Clear gradients properly**:
```python
optimizer.zero_grad(set_to_none=True)  # More efficient
```

3. **Delete unused tensors**:
```python
del intermediate_output
torch.cuda.empty_cache()
```

4. **Check for circular references**:
```python
import gc
gc.collect()
```

---

## Performance Issues

### Issue: Low GPU utilization

**Symptoms**:
- GPU usage < 50%
- Training is slow

**Solutions**:

1. **Increase batch size**:
```yaml
batch_size: 32
```

2. **Use data loader workers**:
```python
train_loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,  # Use multiple workers
    pin_memory=True
)
```

3. **Enable TF32**:
```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

4. **Profile bottlenecks**:
```python
import torch.profiler as profiler

with profiler.profile() as prof:
    model(input)

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

### Issue: Inference is slow

**Symptoms**:
- Generation takes too long
- Low throughput

**Solutions**:

1. **Use torch.compile (PyTorch 2.0+)**:
```python
model = torch.compile(model)
```

2. **Enable KV caching**:
```python
output = model.generate(input_ids, use_cache=True)
```

3. **Use quantization**:
```python
from src.models.quantized_bk_core import quantize_model
model = quantize_model(model, bits=8)
```

4. **Batch inference**:
```python
# Process multiple sequences at once
outputs = model(batch_input_ids)
```

---

## Numerical Stability Issues

### Issue: Schatten norm bounds violated

**Symptoms**:
```
Warning: Schatten S1 bound violated: 1.234 > 1.000
```

**Solutions**:

1. **Increase epsilon**:
```python
config.epsilon = 1.5  # Instead of 1.0
```

2. **Enable precision upgrade**:
```python
config.auto_precision_upgrade = True
```

3. **Check potential values**:
```python
# Verify potential is not too large
assert torch.abs(V).max() < 10.0
```

### Issue: Mourre estimate fails

**Symptoms**:
```
Warning: Mourre estimate not satisfied
```

**Solutions**:

1. **Check imaginary part of z**:
```python
# Ensure Im(z) > 0
z = lambda_val + 1j * eta
assert eta > 0
```

2. **Increase eta**:
```python
config.eta = 0.1  # Instead of 0.01
```

3. **Verify operator construction**:
```python
# Check that H_0 is self-adjoint
assert torch.allclose(H_0, H_0.conj().T)
```

---

## GPU Issues

### Issue: CUDA error during training

**Symptoms**:
```
RuntimeError: CUDA error: device-side assert triggered
```

**Solutions**:

1. **Run on CPU to get better error message**:
```python
device = torch.device('cpu')
model = model.to(device)
```

2. **Check for invalid indices**:
```python
assert (input_ids >= 0).all()
assert (input_ids < vocab_size).all()
```

3. **Reduce batch size**:
```yaml
batch_size: 1  # Minimal batch for debugging
```

### Issue: Multiple GPU training fails

**Symptoms**:
```
RuntimeError: Expected all tensors to be on the same device
```

**Solutions**:

1. **Use DistributedDataParallel**:
```python
from torch.nn.parallel import DistributedDataParallel as DDP

model = DDP(model, device_ids=[local_rank])
```

2. **Ensure all tensors on same device**:
```python
input_ids = input_ids.to(device)
labels = labels.to(device)
```

3. **Use accelerate**:
```python
from accelerate import Accelerator

accelerator = Accelerator()
model, optimizer, train_loader = accelerator.prepare(
    model, optimizer, train_loader
)
```

---

## Data Loading Issues

### Issue: Dataset download fails

**Symptoms**:
```
ConnectionError: Failed to download dataset
```

**Solutions**:

1. **Use manual download**:
```bash
python scripts/prepare_datasets.py --dataset wikitext2
```

2. **Check internet connection**:
```bash
ping huggingface.co
```

3. **Use cached dataset**:
```python
from datasets import load_dataset
dataset = load_dataset('wikitext', 'wikitext-2-v1', cache_dir='./data')
```

### Issue: Tokenization is slow

**Symptoms**:
- Data loading takes very long
- CPU usage is high

**Solutions**:

1. **Pre-tokenize dataset**:
```python
python scripts/prepare_datasets.py --dataset wikitext2 --tokenize
```

2. **Use fast tokenizer**:
```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('gpt2', use_fast=True)
```

3. **Increase num_workers**:
```python
train_loader = DataLoader(dataset, num_workers=8)
```

---

## Model Loading Issues

### Issue: Checkpoint loading fails

**Symptoms**:
```
RuntimeError: Error loading checkpoint
```

**Solutions**:

1. **Check checkpoint format**:
```python
checkpoint = torch.load('checkpoint.pt', map_location='cpu')
print(checkpoint.keys())
```

2. **Load with strict=False**:
```python
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
```

3. **Verify compatibility**:
```python
# Check model architecture matches
print(f"Model config: {model.config}")
print(f"Checkpoint config: {checkpoint['config']}")
```

### Issue: Hugging Face model not found

**Symptoms**:
```
OSError: resnet-bk/model-1b is not a valid model identifier
```

**Solutions**:

1. **Check model name**:
```python
# Use correct model name
model = torch.hub.load('username/resnet-bk', 'resnet_bk_1b')
```

2. **Load from local path**:
```python
from src.models.hf_resnet_bk import ResNetBKForCausalLM
model = ResNetBKForCausalLM.from_pretrained('./checkpoints/model')
```

3. **Check authentication**:
```bash
huggingface-cli login
```

---

## Getting More Help

If you're still experiencing issues:

1. **Check existing issues**: [GitHub Issues](https://github.com/neko-jpg/Project-ResNet-BK-An-O-N-Language-Model-Architecture/issues)
2. **Ask in discussions**: [GitHub Discussions](https://github.com/neko-jpg/Project-ResNet-BK-An-O-N-Language-Model-Architecture/discussions)
3. **Read documentation**: [Full Documentation](docs/)
4. **Contact us**: arat252539@gmail.com

When reporting issues, please include:
- Python version
- PyTorch version
- CUDA version (if using GPU)
- Full error message
- Minimal code to reproduce
- System information

---

**Last Updated**: 2025-01-15

## 🐧 WSL Issues

### Read-only File System Error
If you see `wsl: An error occurred mounting the distribution disk...`, your WSL disk is corrupted or mounted incorrectly.
**Solution:**
1. Open PowerShell as Administrator.
2. Run `wsl --shutdown`.
3. Try opening Ubuntu again.

### Command Not Found (`make`)
If `make` is not found:
```bash
sudo apt update && sudo apt install -y make
```

### Windows vs WSL Sync
If you edit files in Windows but don't see changes in WSL (or vice versa), check if you are running in `/mnt/c/` or `/home/user/`.
- `/mnt/c/...`: Direct access to Windows files. Slow but synced.
- `/home/...`: Fast (Linux FS), but requires `git pull` to sync with Windows changes pushed to GitHub.
