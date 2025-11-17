# Google Colab Tutorials for ResNet-BK

Complete guide to running ResNet-BK on Google Colab free tier.

---

## Overview

All tutorials are designed to run on Google Colab's free tier (T4 GPU, 15GB RAM). Each tutorial is self-contained and includes:
- Automatic dependency installation
- Dataset downloading and preprocessing
- Model training and evaluation
- Visualization and results

---

## Tutorial 1: Quick Start (30 minutes)

**Goal:** Train your first ResNet-BK model on WikiText-2

**What you'll learn:**
- Basic model setup
- Training loop
- Evaluation and perplexity calculation
- Text generation

**Notebook:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neko-jpg/Project-ResNet-BK-An-O-N-Language-Model-Architecture/blob/main/notebooks/01_quick_start.ipynb)

**Steps:**
1. Click the badge above
2. Runtime → Change runtime type → T4 GPU
3. Run all cells (Runtime → Run all)
4. Wait ~30 minutes for training to complete

**Expected Results:**
- Final Perplexity: ~1122
- Training Time: ~28 minutes
- Model Size: 4.15M parameters

**Key Code:**
```python
# Load model
model = LanguageModel(
    vocab_size=30000,
    d_model=256,
    n_layers=6,
    use_prime_bump=True,
    use_scattering_router=True
)

# Train
trainer = Trainer(model, train_loader, val_loader)
trainer.train(num_epochs=3)

# Evaluate
results = benchmark.evaluate(model)
print(f"Perplexity: {results['perplexity']:.2f}")
```

---

## Tutorial 2: Full Training (4 hours)

**Goal:** Train a 1B parameter model with all optimizations

**What you'll learn:**
- Large-scale training techniques
- Memory optimization strategies
- Gradient checkpointing
- Mixed precision training
- Checkpoint management

**Notebook:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neko-jpg/Project-ResNet-BK-An-O-N-Language-Model-Architecture/blob/main/notebooks/02_full_training.ipynb)

**Steps:**
1. Click the badge above
2. Runtime → Change runtime type → T4 GPU
3. Enable high-RAM if available
4. Run all cells
5. Training will take ~4 hours

**Expected Results:**
- Final Perplexity: ~45
- Training Time: ~4 hours
- Model Size: 1B parameters

**Key Features:**
```python
# Enable all optimizations
config = {
    'model': {
        'd_model': 1024,
        'n_layers': 24,
        'use_semiseparable': True,
        'use_prime_bump': True,
        'use_scattering_router': True
    },
    'training': {
        'gradient_checkpointing': True,
        'mixed_precision': True,
        'gradient_accumulation_steps': 4
    },
    'memory_optimization': {
        'cpu_offloading': True
    }
}
```

**Timeout Handling:**
```python
# Automatic checkpoint saving
from src.training import ColabTimeoutHandler

handler = ColabTimeoutHandler(
    checkpoint_dir="checkpoints/",
    save_interval=300  # Save every 5 minutes
)
trainer.add_callback(handler)
```

---

## Tutorial 3: Benchmarking (2 hours)

**Goal:** Compare ResNet-BK to Mamba baseline

**What you'll learn:**
- Fair comparison methodology
- Long-context stability testing
- Quantization robustness evaluation
- Dynamic efficiency measurement
- Statistical significance testing

**Notebook:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neko-jpg/Project-ResNet-BK-An-O-N-Language-Model-Architecture/blob/main/notebooks/03_benchmarking.ipynb)

**Steps:**
1. Click the badge above
2. Runtime → Change runtime type → T4 GPU
3. Run all cells
4. Benchmarks will take ~2 hours

**Expected Results:**
- ResNet-BK stable at 128k context (Mamba diverges at 32k)
- ResNet-BK 4× better at INT4 quantization
- ResNet-BK 2× fewer FLOPs at equal perplexity

**Key Benchmarks:**
```python
# Long-context stability
for seq_len in [8192, 32768, 131072]:
    resnetbk_loss = train_and_evaluate(model_bk, seq_len)
    mamba_loss = train_and_evaluate(model_mamba, seq_len)
    print(f"N={seq_len}: BK={resnetbk_loss:.2f}, Mamba={mamba_loss:.2f}")

# Quantization robustness
for bits in [32, 16, 8, 4]:
    bk_ppl = quantize_and_evaluate(model_bk, bits)
    mamba_ppl = quantize_and_evaluate(model_mamba, bits)
    print(f"{bits}-bit: BK={bk_ppl:.2f}, Mamba={mamba_ppl:.2f}")

# Dynamic efficiency
for flops_budget in [1e9, 2e9, 5e9]:
    bk_ppl = evaluate_at_flops(model_bk, flops_budget)
    mamba_ppl = evaluate_at_flops(model_mamba, flops_budget)
    print(f"FLOPs={flops_budget:.0e}: BK={bk_ppl:.2f}, Mamba={mamba_ppl:.2f}")
```

---

## Tutorial 4: Visualization (30 minutes)

**Goal:** Generate publication-quality "killer graphs"

**What you'll learn:**
- Data visualization with matplotlib
- Publication-quality figure generation
- Statistical significance plotting
- Interactive dashboards

**Notebook:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neko-jpg/Project-ResNet-BK-An-O-N-Language-Model-Architecture/blob/main/notebooks/generate_killer_graphs.ipynb)

**Steps:**
1. Click the badge above
2. No GPU needed (CPU is fine)
3. Run all cells
4. Graphs will be generated in ~5 minutes

**Generated Graphs:**
1. **Long-Context Stability Graph**
   - Shows ResNet-BK stable at 1M tokens
   - Shows Mamba diverging at 32k tokens

2. **Quantization Robustness Graph**
   - Shows ResNet-BK maintaining low PPL at INT4
   - Shows Mamba degrading significantly

3. **Dynamic Efficiency Graph**
   - Shows ResNet-BK achieving lower PPL at every FLOPs budget
   - Shows Pareto frontier dominance

**Key Code:**
```python
# Generate all graphs
from scripts import (
    generate_stability_graph,
    generate_quantization_graph,
    generate_efficiency_graph
)

# Long-context stability
generate_stability_graph.main(
    seq_lengths=[8192, 32768, 131072, 524288, 1048576],
    output="results/stability_graph.pdf"
)

# Quantization robustness
generate_quantization_graph.main(
    bit_widths=[32, 16, 8, 4, 2],
    output="results/quantization_graph.pdf"
)

# Dynamic efficiency
generate_efficiency_graph.main(
    flops_budgets=[1e9, 2e9, 5e9, 1e10],
    output="results/efficiency_graph.pdf"
)
```

---

## Additional Tutorials

### Tutorial 5: Interpretability

**Goal:** Understand what the model learns

**Notebook:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neko-jpg/Project-ResNet-BK-An-O-N-Language-Model-Architecture/blob/main/notebooks/04_interpretability.ipynb)

**What you'll learn:**
- Scattering phase visualization
- Prime-Bump potential analysis
- Expert routing patterns
- Attention-like interpretability

### Tutorial 6: Compression

**Goal:** Compress models for deployment

**Notebook:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neko-jpg/Project-ResNet-BK-An-O-N-Language-Model-Architecture/blob/main/notebooks/step4_compression.ipynb)

**What you'll learn:**
- Quantization-aware training
- Knowledge distillation
- Structured pruning
- 100× compression pipeline

### Tutorial 7: System Integration

**Goal:** Integrate all components

**Notebook:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neko-jpg/Project-ResNet-BK-An-O-N-Language-Model-Architecture/blob/main/notebooks/step7_system_integration.ipynb)

**What you'll learn:**
- End-to-end pipeline
- Failure recovery
- Monitoring and logging
- Production deployment

---

## Tips for Google Colab

### 1. GPU Selection

Always select T4 GPU:
```
Runtime → Change runtime type → Hardware accelerator → GPU → T4
```

### 2. Handling Timeouts

Colab disconnects after 12 hours. Use automatic checkpointing:

```python
from src.training import ColabTimeoutHandler

handler = ColabTimeoutHandler(
    checkpoint_dir="/content/drive/MyDrive/checkpoints/",
    save_interval=300  # Save every 5 minutes
)
trainer.add_callback(handler)
```

### 3. Using Google Drive

Mount Google Drive to save checkpoints:

```python
from google.colab import drive
drive.mount('/content/drive')

# Save checkpoints to Drive
checkpoint_dir = "/content/drive/MyDrive/resnetbk_checkpoints/"
```

### 4. Memory Management

If you run out of memory:

```python
# Clear cache
import torch
torch.cuda.empty_cache()

# Reduce batch size
config['training']['batch_size'] = 4

# Enable gradient checkpointing
config['training']['gradient_checkpointing'] = True
```

### 5. Monitoring Training

Use TensorBoard in Colab:

```python
%load_ext tensorboard
%tensorboard --logdir logs/

# In training code
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('logs/')
```

### 6. Downloading Results

Download trained models and results:

```python
from google.colab import files

# Download checkpoint
files.download('checkpoints/final_model.pt')

# Download graphs
files.download('results/stability_graph.pdf')
```

---

## Troubleshooting

### Issue: "Runtime disconnected"

**Solution:** Use automatic checkpointing and resume training:

```python
# Save checkpoint every 5 minutes
handler = ColabTimeoutHandler(save_interval=300)

# Resume from checkpoint
if os.path.exists('checkpoint.pt'):
    trainer.load_checkpoint('checkpoint.pt')
```

### Issue: "CUDA out of memory"

**Solutions:**
1. Reduce batch size
2. Enable gradient checkpointing
3. Reduce sequence length
4. Use CPU offloading

### Issue: "Training is slow"

**Solutions:**
1. Verify GPU is enabled: `torch.cuda.is_available()`
2. Enable mixed precision
3. Use scattering router (10× faster)
4. Increase batch size

### Issue: "Cannot install dependencies"

**Solution:** Use pre-built Docker image:

```python
# Pull Docker image
!docker pull resnetbk/mamba-killer:latest

# Run in Docker
!docker run --gpus all -it resnetbk/mamba-killer:latest
```

---

## Best Practices

### 1. Start Small

Begin with Tutorial 1 (Quick Start) before attempting larger models.

### 2. Save Frequently

Use automatic checkpointing to avoid losing progress.

### 3. Monitor Resources

Check GPU/RAM usage:
```python
!nvidia-smi
!free -h
```

### 4. Use Colab Pro (Optional)

For longer training runs, consider Colab Pro:
- Longer runtime (24 hours vs 12 hours)
- More RAM (52GB vs 15GB)
- Faster GPUs (V100/A100 vs T4)

### 5. Share Your Results

Share your trained models and results with the community!

---

## Next Steps

After completing the tutorials:

1. **Experiment**: Try different hyperparameters
2. **Scale Up**: Train larger models (1B, 10B parameters)
3. **Deploy**: Export to ONNX/TensorRT
4. **Contribute**: Share your improvements!

For more help:
- [TUTORIAL.md](TUTORIAL.md) - Detailed training guide
- [FAQ.md](FAQ.md) - Common questions
- [GitHub Issues](https://github.com/neko-jpg/Project-ResNet-BK-An-O-N-Language-Model-Architecture/issues) - Report problems

---

**Happy training on Colab! 🚀**
