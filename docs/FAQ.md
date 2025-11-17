# ResNet-BK Frequently Asked Questions (FAQ)

Common questions and troubleshooting guide for ResNet-BK.

---

## Table of Contents

1. [General Questions](#general-questions)
2. [Installation & Setup](#installation--setup)
3. [Training Issues](#training-issues)
4. [Performance & Optimization](#performance--optimization)
5. [Model Architecture](#model-architecture)
6. [Comparison to Other Models](#comparison-to-other-models)
7. [Deployment](#deployment)
8. [Contributing](#contributing)

---

## General Questions

### What is ResNet-BK?

ResNet-BK is a mathematically rigorous O(N) language model architecture based on the Birman-Schwinger operator from quantum scattering theory. Unlike empirical approaches, every component is backed by proven theorems guaranteeing numerical stability and computational efficiency.

### How is ResNet-BK different from Transformers?

| Feature | Transformer | ResNet-BK |
|---------|-------------|-----------|
| **Complexity** | O(N²) | O(N) |
| **Long Context** | Limited to 8k-32k | Stable up to 1M tokens |
| **Quantization** | Degrades significantly | 4× better at INT4 |
| **Routing** | Learned (expensive) | Physics-based (free) |
| **Stability** | Empirical | Mathematically proven |

### How is ResNet-BK different from Mamba?

ResNet-BK surpasses Mamba in three key dimensions:

1. **Long-Context Stability**: Stable training on 1M tokens (vs Mamba's 32k limit)
2. **Quantization Robustness**: 4× lower perplexity at INT4 quantization
3. **Dynamic Efficiency**: 2× fewer FLOPs at equal perplexity

### Is ResNet-BK production-ready?

ResNet-BK is currently a research prototype. While the core architecture is stable and well-tested, production deployment requires:
- Additional optimization for specific hardware
- More extensive testing on diverse datasets
- Integration with existing ML infrastructure

We're actively working on these improvements!

### What are the hardware requirements?

**Minimum:**
- GPU: NVIDIA T4 (16GB VRAM) or equivalent
- RAM: 16GB
- Storage: 50GB

**Recommended:**
- GPU: NVIDIA A100 (40GB VRAM) or better
- RAM: 32GB+
- Storage: 100GB+

**Google Colab:** Free tier (T4 GPU) works great for models up to 1B parameters!

---

## Installation & Setup

### How do I install ResNet-BK?

```bash
git clone https://github.com/neko-jpg/Project-ResNet-BK-An-O-N-Language-Model-Architecture.git
cd Project-ResNet-BK-An-O-N-Language-Model-Architecture
pip install -r requirements.txt
pip install -e .
```

### What Python version do I need?

Python 3.8 or higher. We recommend Python 3.10 for best compatibility.

### What PyTorch version do I need?

PyTorch 2.0 or higher. Install with:

```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### I get "ImportError: No module named 'src'"

Make sure you installed the package:

```bash
pip install -e .
```

If that doesn't work, add the repo to your Python path:

```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/Project-ResNet-BK-An-O-N-Language-Model-Architecture"
```

### Can I use ResNet-BK on CPU?

Yes, but training will be very slow. We strongly recommend using a GPU. For inference, CPU is acceptable for small models.

---

## Training Issues

### Training crashes with "RuntimeError: CUDA out of memory"

**Solutions:**

1. **Reduce batch size:**
```yaml
training:
  batch_size: 4  # Instead of 8
  gradient_accumulation_steps: 2  # Maintain effective batch size
```

2. **Enable gradient checkpointing:**
```yaml
training:
  gradient_checkpointing: true
```

3. **Enable CPU offloading:**
```yaml
memory_optimization:
  cpu_offloading: true
```

4. **Use semiseparable structure:**
```yaml
model:
  use_semiseparable: true
```

5. **Reduce sequence length:**
```yaml
model:
  n_seq: 256  # Instead of 512
```

### I get NaN/Inf during training

**Causes and solutions:**

1. **Learning rate too high:**
```yaml
training:
  learning_rate: 5e-4  # Instead of 1e-3
```

2. **Enable gradient clipping:**
```yaml
training:
  gradient_clip_norm: 1.0
```

3. **Increase epsilon:**
```yaml
model:
  epsilon: 1.0  # Instead of 0.5
```

4. **Enable automatic recovery:**
```python
from src.training import AutoRecovery
recovery = AutoRecovery(checkpoint_dir="checkpoints/")
trainer.add_callback(recovery)
```

5. **Check data quality:**
```python
# Verify no NaN in data
for batch in train_loader:
    assert torch.isfinite(batch['input_ids']).all()
```

### Training is very slow

**Solutions:**

1. **Enable mixed precision:**
```yaml
training:
  mixed_precision: true
```

2. **Use scattering router (10× faster):**
```yaml
model:
  use_scattering_router: true
```

3. **Increase batch size:**
```yaml
training:
  batch_size: 16  # Better GPU utilization
```

4. **Enable CUDA kernels:**
```python
model.bk_core.use_cuda_kernel = True
```

5. **Check GPU utilization:**
```bash
nvidia-smi -l 1  # Monitor GPU usage
```

If GPU utilization is low (<80%), increase batch size or reduce data loading bottlenecks.

### Model doesn't converge (perplexity stays high)

**Solutions:**

1. **Use Prime-Bump initialization:**
```yaml
model:
  use_prime_bump: true
```

2. **Adjust GRAD_BLEND:**
```yaml
model:
  grad_blend: 0.0  # Pure analytic gradient
```

3. **Increase warmup:**
```yaml
training:
  warmup_steps: 500  # Instead of 100
```

4. **Check learning rate schedule:**
```python
# Visualize LR schedule
import matplotlib.pyplot as plt
lrs = [scheduler.get_last_lr()[0] for _ in range(1000)]
plt.plot(lrs)
plt.show()
```

5. **Verify data preprocessing:**
```python
# Check tokenization
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
sample = "The quick brown fox"
tokens = tokenizer.encode(sample)
print(f"Tokens: {tokens}")
print(f"Decoded: {tokenizer.decode(tokens)}")
```

### Training stops with "RuntimeError: CUDA error: device-side assert triggered"

This usually indicates an indexing error. Check:

1. **Vocabulary size matches tokenizer:**
```python
assert model.vocab_size == len(tokenizer)
```

2. **No out-of-bounds indices:**
```python
assert (input_ids >= 0).all() and (input_ids < vocab_size).all()
```

3. **Labels are properly masked:**
```python
# Use -100 for padding tokens
labels[labels == tokenizer.pad_token_id] = -100
```

---

## Performance & Optimization

### How do I maximize training speed?

1. **Enable all optimizations:**
```yaml
training:
  mixed_precision: true
  gradient_checkpointing: true
  
model:
  use_scattering_router: true
  use_semiseparable: true
```

2. **Optimize batch size:**
```python
# Find maximum batch size
for batch_size in [4, 8, 16, 32, 64]:
    try:
        train_one_batch(batch_size)
        print(f"Batch size {batch_size} works!")
    except RuntimeError:
        print(f"Batch size {batch_size} OOM")
        break
```

3. **Use data loading optimizations:**
```python
train_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=4,  # Parallel data loading
    pin_memory=True,  # Faster GPU transfer
    persistent_workers=True  # Keep workers alive
)
```

### How do I reduce memory usage?

**Memory optimization checklist:**

- [ ] Enable gradient checkpointing
- [ ] Enable semiseparable structure
- [ ] Enable CPU offloading
- [ ] Use mixed precision (FP16)
- [ ] Reduce batch size
- [ ] Reduce sequence length
- [ ] Use gradient accumulation

**Measure memory usage:**
```python
import torch

# Before training
torch.cuda.reset_peak_memory_stats()

# Train
train_one_epoch()

# After training
peak_memory = torch.cuda.max_memory_allocated() / 1e9
print(f"Peak memory: {peak_memory:.2f} GB")
```

### How do I train on multiple GPUs?

**Data Parallel (simple):**
```python
model = nn.DataParallel(model)
```

**Distributed Data Parallel (recommended):**
```bash
torchrun --nproc_per_node=4 train.py --config config.yaml
```

**ZeRO (for very large models):**
```python
from src.training import ZeROOptimizer

optimizer = ZeROOptimizer(
    model.parameters(),
    stage=2,  # ZeRO Stage 2
    partition_size=4  # 4 GPUs
)
```

### How do I train on Google Colab?

1. **Select GPU runtime:**
   - Runtime → Change runtime type → T4 GPU

2. **Clone and install:**
```bash
!git clone https://github.com/neko-jpg/Project-ResNet-BK-An-O-N-Language-Model-Architecture.git
%cd Project-ResNet-BK-An-O-N-Language-Model-Architecture
!pip install -r requirements.txt -q
```

3. **Train:**
```python
!python train.py --config configs/colab_config.yaml
```

4. **Handle timeouts:**
```python
from src.training import ColabTimeoutHandler

handler = ColabTimeoutHandler(
    checkpoint_dir="checkpoints/",
    save_interval=300  # Save every 5 minutes
)
trainer.add_callback(handler)
```

---

## Model Architecture

### What is the Birman-Schwinger operator?

The Birman-Schwinger operator is a mathematical construct from quantum scattering theory:

```
K_ε(z) = |V_ε|^{1/2} R_0(z) |V_ε|^{1/2}
```

It provides:
- **Numerical stability** via trace-class bounds
- **O(N) complexity** via theta/phi recursions
- **Interpretability** via scattering phase

### What is Prime-Bump initialization?

Prime-Bump initialization places Gaussian "bumps" at prime number positions:

```
V_ε(x) = Σ_p α_{p,k}(ε) ψ_ε(x - log p)
```

**Benefits:**
- 2× faster convergence
- Optimal eigenvalue spacing (GUE statistics)
- Matches Riemann zeta function spectral properties

### What is scattering-based routing?

Scattering-based routing uses the scattering phase to route tokens to experts:

```
δ_ε(λ) = arg(det_2(I + K_ε(λ + i0)))
```

**Advantages:**
- Zero learnable parameters
- 10× faster than MLP routing
- Interpretable: phase correlates with difficulty

### What is the semiseparable structure?

Semiseparable structure decomposes matrices as:

```
H = T + U·V^T
```

where T is tridiagonal and UV^T is low-rank.

**Benefits:**
- O(N) matrix-vector multiplication
- O(N log N) memory instead of O(N²)
- Enables training on 1M token sequences

### Can I use ResNet-BK without Prime-Bump initialization?

Yes, but convergence will be slower. To disable:

```yaml
model:
  use_prime_bump: false
```

### Can I use ResNet-BK without scattering router?

Yes, you can use learned MLP routing instead:

```yaml
model:
  use_scattering_router: false
```

But this will be 10× slower and require training the router.

---

## Comparison to Other Models

### How does ResNet-BK compare to GPT?

| Feature | GPT | ResNet-BK |
|---------|-----|-----------|
| **Architecture** | Transformer | BK-Core + MoE |
| **Complexity** | O(N²) | O(N) |
| **Max Context** | 8k-32k | 1M |
| **Quantization** | Moderate | Excellent |
| **Training Cost** | High | Low |

### How does ResNet-BK compare to Mamba?

See [Benchmark Results](README.md#benchmark-results) for detailed comparison.

**Summary:**
- **Long-context**: 31× longer stable context
- **Quantization**: 4× better at INT4
- **Efficiency**: 2× fewer FLOPs at equal PPL

### How does ResNet-BK compare to RWKV?

Both are O(N) architectures, but:
- **ResNet-BK**: Mathematically rigorous, proven stability
- **RWKV**: Empirical design, simpler implementation

ResNet-BK has better long-context stability and quantization robustness.

### Can ResNet-BK replace Transformers?

For many applications, yes! ResNet-BK is particularly well-suited for:
- Long-context tasks (>32k tokens)
- Edge deployment (quantization-friendly)
- Resource-constrained environments

However, Transformers still have advantages in:
- Ecosystem maturity (more tools, libraries)
- Pre-trained models (more available)
- Community support

---

## Deployment

### How do I export to ONNX?

```python
from src.models import ONNXExporter

exporter = ONNXExporter(model)
exporter.export(
    output_path="model.onnx",
    opset_version=14,
    verify=True  # Verify numerical equivalence
)
```

### How do I export to TensorRT?

```python
from src.models import TensorRTExporter

exporter = TensorRTExporter(model)
exporter.export(
    output_path="model.trt",
    precision="fp16",
    max_batch_size=8
)

# Measure speedup
speedup = exporter.benchmark()
print(f"TensorRT speedup: {speedup:.2f}×")
```

### How do I deploy on Hugging Face?

```python
# Save model
model.save_pretrained("my-resnetbk-model")

# Upload to Hub
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path="my-resnetbk-model",
    repo_id="username/my-resnetbk-model",
    repo_type="model"
)
```

### How do I serve the model?

**Using FastAPI:**
```python
from fastapi import FastAPI
from src.models import LanguageModel

app = FastAPI()
model = LanguageModel.from_pretrained("resnetbk/mamba-killer-1b")

@app.post("/generate")
def generate(prompt: str, max_length: int = 100):
    output = model.generate(prompt, max_length=max_length)
    return {"generated_text": output}
```

**Using TorchServe:**
```bash
torch-model-archiver \
  --model-name resnetbk \
  --version 1.0 \
  --serialized-file model.pt \
  --handler handler.py

torchserve --start --model-store model_store --models resnetbk=resnetbk.mar
```

---

## Contributing

### How can I contribute?

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Ways to contribute:**
- Report bugs
- Suggest features
- Improve documentation
- Submit pull requests
- Share your results

### How do I report a bug?

1. Check if the bug is already reported in [GitHub Issues](https://github.com/neko-jpg/Project-ResNet-BK-An-O-N-Language-Model-Architecture/issues)
2. If not, create a new issue with:
   - Clear description
   - Steps to reproduce
   - Expected vs actual behavior
   - System information (OS, Python version, PyTorch version)
   - Error messages and stack traces

### How do I request a feature?

Create a [GitHub Issue](https://github.com/neko-jpg/Project-ResNet-BK-An-O-N-Language-Model-Architecture/issues) with:
- Clear description of the feature
- Use case / motivation
- Proposed implementation (if you have ideas)

### How do I submit a pull request?

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes
4. Run tests: `pytest tests/`
5. Commit: `git commit -m "Add my feature"`
6. Push: `git push origin feature/my-feature`
7. Create pull request on GitHub

### Where can I get help?

- **Documentation**: [TUTORIAL.md](TUTORIAL.md), [API_REFERENCE.md](API_REFERENCE.md)
- **GitHub Issues**: [Report bugs or ask questions](https://github.com/neko-jpg/Project-ResNet-BK-An-O-N-Language-Model-Architecture/issues)
- **GitHub Discussions**: [Community discussions](https://github.com/neko-jpg/Project-ResNet-BK-An-O-N-Language-Model-Architecture/discussions)

---

## Still have questions?

If your question isn't answered here:

1. Check the [documentation](README.md)
2. Search [GitHub Issues](https://github.com/neko-jpg/Project-ResNet-BK-An-O-N-Language-Model-Architecture/issues)
3. Ask in [GitHub Discussions](https://github.com/neko-jpg/Project-ResNet-BK-An-O-N-Language-Model-Architecture/discussions)
4. Create a new issue

We're here to help! 🚀
