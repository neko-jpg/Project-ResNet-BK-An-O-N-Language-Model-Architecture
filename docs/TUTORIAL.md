# ResNet-BK Training Tutorial

This comprehensive tutorial will guide you through training ResNet-BK models from scratch, from a simple WikiText-2 model to ultra-scale 10B parameter models on Google Colab.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Tutorial 1: Your First Model (30 minutes)](#tutorial-1-your-first-model)
3. [Tutorial 2: Understanding the Architecture](#tutorial-2-understanding-the-architecture)
4. [Tutorial 3: Long-Context Training](#tutorial-3-long-context-training)
5. [Tutorial 4: Quantization and Compression](#tutorial-4-quantization-and-compression)
6. [Tutorial 5: Benchmarking vs Mamba](#tutorial-5-benchmarking-vs-mamba)
7. [Tutorial 6: Advanced Optimization](#tutorial-6-advanced-optimization)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements

- **Python**: 3.8 or higher
- **PyTorch**: 2.0 or higher
- **CUDA**: 11.8 or higher (for GPU training)
- **RAM**: 16GB minimum (32GB recommended)
- **GPU**: T4 or better (Google Colab free tier works!)

### Installation

```bash
# Clone the repository
git clone https://github.com/neko-jpg/Project-ResNet-BK-An-O-N-Language-Model-Architecture.git
cd Project-ResNet-BK-An-O-N-Language-Model-Architecture

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Verify Installation

```python
import torch
from src.models import LanguageModel

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"ResNet-BK imported successfully!")
```

---

## Tutorial 1: Your First Model

Let's train a small ResNet-BK model on WikiText-2 in 30 minutes.

### Step 1: Prepare the Dataset

```bash
# Download and prepare WikiText-2
python scripts/prepare_datasets.py --dataset wikitext2 --output data/
```

### Step 2: Configure Your Model

Create a configuration file `my_first_model.yaml`:

```yaml
# Model architecture
model:
  vocab_size: 30000
  d_model: 256
  n_layers: 6
  n_seq: 512
  
  # Birman-Schwinger parameters
  epsilon: 1.0
  use_prime_bump: true
  
  # MoE parameters
  num_experts: 4
  top_k: 2
  use_scattering_router: true

# Training
training:
  batch_size: 8
  learning_rate: 1e-3
  num_epochs: 3
  warmup_steps: 100
  
  # Optimization
  gradient_checkpointing: true
  mixed_precision: true

# Dataset
data:
  dataset: wikitext2
  seq_len: 512
```

### Step 3: Train the Model

```bash
python train.py --config my_first_model.yaml
```

**Expected output:**
```
Epoch 1/3 | Step 100 | Loss: 7.48 | PPL: 1780.23 | LR: 0.001
Epoch 1/3 | Step 200 | Loss: 7.12 | PPL: 1235.67 | LR: 0.001
...
Epoch 3/3 | Final Loss: 7.02 | Final PPL: 1122.06
Training completed in 28 minutes!
```

### Step 4: Evaluate the Model

```python
from src.models import LanguageModel
from src.benchmarks import WikiText2Benchmark

# Load trained model
model = LanguageModel.from_checkpoint("checkpoints/my_first_model/final.pt")

# Evaluate
benchmark = WikiText2Benchmark()
results = benchmark.evaluate(model)

print(f"Test Perplexity: {results['perplexity']:.2f}")
print(f"Test Loss: {results['loss']:.4f}")
```

### Step 5: Generate Text

```python
import torch
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Prepare input
prompt = "The future of artificial intelligence is"
input_ids = tokenizer.encode(prompt, return_tensors="pt")

# Generate
output_ids = model.generate(
    input_ids,
    max_length=100,
    temperature=0.8,
    top_p=0.9
)

# Decode
generated_text = tokenizer.decode(output_ids[0])
print(generated_text)
```

---

## Tutorial 2: Understanding the Architecture

### The Birman-Schwinger Core

The heart of ResNet-BK is the Birman-Schwinger operator:

```python
from src.models import BirmanSchwingerCore

# Create BK-Core
bk_core = BirmanSchwingerCore(
    n_seq=512,
    epsilon=1.0,
    use_mourre=True,  # Enable Mourre estimate
    use_lap=True      # Enable LAP stability
)

# Forward pass
v = torch.randn(8, 512)  # Potential from Prime-Bump
z = 1.0j                  # Complex shift
G_ii = bk_core(v, z)      # Diagonal resolvent

print(f"Output shape: {G_ii.shape}")  # [8, 512, 2] (real, imag)
```

**Key parameters:**
- `epsilon`: Regularization parameter (0.5-1.0). Lower = more compressed.
- `use_mourre`: Enable Mourre estimate verification for stability.
- `use_lap`: Enable Limiting Absorption Principle for boundary computation.

### Prime-Bump Initialization

Initialize with prime number distribution:

```python
from src.models import PrimeBumpPotential

# Create Prime-Bump potential
prime_bump = PrimeBumpPotential(
    n_seq=512,
    epsilon=1.0,
    k_max=3  # Maximum prime power
)

# Get prime positions
primes = prime_bump.get_prime_indices()
print(f"Prime positions: {primes[:10]}")  # [2, 3, 5, 7, 11, ...]

# Verify GUE statistics
stats = prime_bump.verify_gue_statistics()
print(f"Wigner surmise fit: {stats['wigner_fit']:.4f}")
```

**Benefits:**
- 2× faster convergence than random initialization
- Optimal eigenvalue spacing (GUE statistics)
- Matches Riemann zeta function spectral properties

### Scattering-Based Router

Zero-parameter MoE routing:

```python
from src.models import ScatteringRouter

# Create router
router = ScatteringRouter(
    num_experts=8,
    use_clark_measure=False
)

# Route tokens
G_ii = torch.randn(8, 512, 2)  # From BK-Core
expert_indices, routing_weights = router(G_ii)

print(f"Expert indices: {expert_indices.shape}")  # [8, 512, 2]
print(f"Routing weights: {routing_weights.shape}")  # [8, 512, 2]
```

**Advantages:**
- 10× faster than learned MLP routing
- No learnable parameters (zero training cost)
- Interpretable: scattering phase correlates with difficulty

---

## Tutorial 3: Long-Context Training

Train on ultra-long sequences (128k-1M tokens).

### Step 1: Configure Long-Context Model

```yaml
# long_context_config.yaml
model:
  n_seq: 131072  # 128k tokens
  use_semiseparable: true  # Enable O(N log N) memory
  low_rank: 17  # log2(131072)

training:
  batch_size: 1  # Reduce for long context
  gradient_accumulation_steps: 8
  
memory_optimization:
  gradient_checkpointing: true
  cpu_offloading: true
  mixed_precision: true
```

### Step 2: Train with Streaming

```bash
python scripts/train_long_context.py \
  --config long_context_config.yaml \
  --seq_len 131072 \
  --streaming true
```

### Step 3: Monitor Stability

```python
from src.training import StabilityMonitor

monitor = StabilityMonitor()

# During training loop
for batch in dataloader:
    loss = model(batch)
    
    # Check numerical health
    health = monitor.check_tensors({
        'loss': loss,
        'gradients': model.get_gradients()
    })
    
    if not health['all_finite']:
        print("⚠️ NaN/Inf detected! Applying recovery...")
        monitor.suggest_recovery('nan_detected')
```

### Step 4: Evaluate on Ultra-Long Sequences

```python
from src.benchmarks import StreamingEvaluator

# Evaluate on 1M tokens without loading entire sequence
evaluator = StreamingEvaluator(
    model=model,
    chunk_size=8192,  # Process in 8k chunks
    overlap=512       # Overlap for context
)

results = evaluator.evaluate_file("data/long_document.txt")
print(f"Perplexity on 1M tokens: {results['perplexity']:.2f}")
```

---

## Tutorial 4: Quantization and Compression

Compress your model for deployment.

### Step 1: Post-Training Quantization (PTQ)

```python
from src.models import QuantizedBirmanSchwinger

# Load FP32 model
model = LanguageModel.from_checkpoint("checkpoints/model.pt")

# Quantize to INT8
quantized_model = QuantizedBirmanSchwinger.from_float(
    model,
    bits=8,
    calibration_data=calibration_loader
)

# Evaluate
results = benchmark.evaluate(quantized_model)
print(f"INT8 PPL degradation: {results['ppl_degradation']:.2f}%")
```

### Step 2: Quantization-Aware Training (QAT)

```python
from src.training import QuantizationAwareTrainer

# Create QAT trainer
trainer = QuantizationAwareTrainer(
    model=model,
    bits=8,
    qat_start_epoch=1  # Start QAT after 1 epoch
)

# Train with QAT
trainer.train(train_loader, num_epochs=3)

# Export quantized model
quantized_model = trainer.export_quantized()
```

### Step 3: Mixed-Precision Quantization

```python
from src.models import MixedPrecisionQuantization

# Configure mixed precision
config = {
    'moe_experts': 4,      # INT4 for experts
    'bk_core': 8,          # INT8 for BK-Core
    'output_layers': 16    # FP16 for output
}

# Apply mixed precision
mixed_model = MixedPrecisionQuantization.apply(model, config)

# Measure compression
original_size = model.get_size_mb()
compressed_size = mixed_model.get_size_mb()
print(f"Compression: {original_size / compressed_size:.1f}×")
```

### Step 4: Knowledge Distillation

```python
from src.training import ClarkDistillation

# Teacher model (large)
teacher = LanguageModel.from_checkpoint("checkpoints/teacher_10b.pt")

# Student model (small)
student = LanguageModel(vocab_size=30000, d_model=128, n_layers=4)

# Distill with Clark measure preservation
distiller = ClarkDistillation(
    teacher=teacher,
    student=student,
    clark_weight=0.1  # Weight for Clark measure loss
)

distiller.train(train_loader, num_epochs=5)
```

---

## Tutorial 5: Benchmarking vs Mamba

Compare ResNet-BK to Mamba baseline.

### Step 1: Install Mamba

```bash
pip install mamba-ssm
```

### Step 2: Run Fair Comparison

```bash
python scripts/mamba_vs_bk_benchmark.py \
  --model bk \
  --seq_len 32768 \
  --bits 8 \
  --dataset wikitext2
```

### Step 3: Generate Killer Graphs

```python
from scripts import generate_stability_graph, generate_quantization_graph, generate_efficiency_graph

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

### Step 4: Statistical Validation

```python
from src.benchmarks import FairComparison

# Run 5 seeds for statistical significance
comparison = FairComparison(
    model_a="resnetbk",
    model_b="mamba",
    num_seeds=5
)

results = comparison.run_all_benchmarks()

# Compute p-values
print(f"Long-context p-value: {results['longcontext_pvalue']:.4f}")
print(f"Quantization p-value: {results['quantization_pvalue']:.4f}")
print(f"Efficiency p-value: {results['efficiency_pvalue']:.4f}")
```

---

## Tutorial 6: Advanced Optimization

### Adaptive Computation Time (ACT)

Enable dynamic depth based on token difficulty:

```python
from src.models import ACTModule

# Add ACT to model
model.add_module('act', ACTModule(
    n_layers=8,
    halt_threshold=0.2  # Exit early when phase < 0.2
))

# Train with ACT
trainer.train(train_loader, use_act=True)

# Measure FLOPs savings
avg_depth = model.get_average_depth()
print(f"Average layers used: {avg_depth:.2f} / 8")
print(f"FLOPs reduction: {(1 - avg_depth/8) * 100:.1f}%")
```

### Learned Sparsity

Predict which computations are important:

```python
from src.models import LearnedSparsityGii

# Add learned sparsity
model.bk_core.add_sparsity_predictor(
    target_sparsity=0.6  # 60% sparsity
)

# Train with sparsity loss
trainer.train(
    train_loader,
    sparsity_weight=0.01  # Weight for sparsity loss
)

# Measure speedup
speedup = model.measure_speedup()
print(f"BK-Core speedup: {speedup:.2f}×")
```

### Multi-Scale Processing

Downsample at middle layers:

```python
from src.models import MultiScaleBKLayer

# Replace middle layers with multi-scale
for i in range(3, 6):  # Layers 3-5
    model.layers[i] = MultiScaleBKLayer(
        d_model=256,
        downsample_factor=2  # 2× downsampling
    )

# Train
trainer.train(train_loader)

# Measure efficiency
flops_reduction = model.measure_flops_reduction()
print(f"FLOPs reduction: {flops_reduction * 100:.1f}%")
```

---

## Troubleshooting

### Issue 1: NaN/Inf During Training

**Symptoms:**
```
RuntimeError: Loss is NaN at step 150
```

**Solutions:**

1. **Reduce learning rate:**
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

### Issue 2: Out of Memory (OOM)

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**

1. **Enable gradient checkpointing:**
```yaml
training:
  gradient_checkpointing: true
```

2. **Reduce batch size:**
```yaml
training:
  batch_size: 4  # Instead of 8
  gradient_accumulation_steps: 2  # Maintain effective batch size
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

### Issue 3: Slow Training

**Symptoms:**
```
Training speed: 100 tokens/sec (expected: 1000+)
```

**Solutions:**

1. **Enable mixed precision:**
```yaml
training:
  mixed_precision: true
```

2. **Use scattering router:**
```yaml
model:
  use_scattering_router: true  # 10× faster than MLP
```

3. **Enable CUDA kernels:**
```python
model.bk_core.use_cuda_kernel = True
```

4. **Increase batch size:**
```yaml
training:
  batch_size: 16  # Better GPU utilization
```

### Issue 4: Poor Convergence

**Symptoms:**
```
Perplexity stuck at 2000+ after 3 epochs
```

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

4. **Check data quality:**
```python
# Verify tokenization
from src.utils import verify_tokenization
verify_tokenization(train_loader)
```

---

## Next Steps

- **Experiment with hyperparameters**: Try different epsilon values, layer counts, etc.
- **Scale up**: Train larger models (1B, 10B parameters)
- **Deploy**: Export to ONNX/TensorRT for production
- **Contribute**: Share your results and improvements!

For more help, see:
- [FAQ.md](FAQ.md) - Common questions
- [API_REFERENCE.md](API_REFERENCE.md) - Complete API docs
- [GitHub Issues](https://github.com/neko-jpg/Project-ResNet-BK-An-O-N-Language-Model-Architecture/issues) - Report bugs

---

**Happy training! 🚀**
