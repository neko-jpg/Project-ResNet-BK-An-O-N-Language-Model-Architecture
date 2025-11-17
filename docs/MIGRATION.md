# Migration Guide

This guide helps you migrate between major versions of ResNet-BK.

## Table of Contents

- [Migrating to 1.0.0](#migrating-to-100)
- [Migrating to 0.9.0](#migrating-to-090)
- [Migrating to 0.8.0](#migrating-to-080)
- [Migrating to 0.7.0](#migrating-to-070)
- [Migrating to 0.6.0](#migrating-to-060)
- [Migrating to 0.5.0](#migrating-to-050)

---

## Migrating to 1.0.0

### Breaking Changes

#### 1. Configuration System Refactored

**Before (0.9.x):**
```python
from src.models.resnet_bk import ResNetBK

model = ResNetBK(
    vocab_size=30000,
    d_model=256,
    n_layers=8,
    use_prime_bump=True
)
```

**After (1.0.0):**
```python
from src.models.resnet_bk import ResNetBK
from src.utils.config import MambaKillerConfig

config = MambaKillerConfig(
    vocab_size=30000,
    d_model=256,
    n_layers=8,
    use_prime_bump=True
)
model = ResNetBK(config)
```

#### 2. Checkpoint Format Changed

Checkpoints now include additional metadata for reproducibility.

**Migration Script:**
```python
import torch
from src.utils.checkpoint_manager import CheckpointManager

# Load old checkpoint
old_checkpoint = torch.load('old_model.pt')

# Convert to new format
manager = CheckpointManager('checkpoints')
manager.save_checkpoint(
    model=old_checkpoint['model'],
    optimizer=old_checkpoint['optimizer'],
    epoch=old_checkpoint['epoch'],
    step=old_checkpoint['step'],
    metadata={'version': '1.0.0'}
)
```

#### 3. API Changes

- `BirmanSchwingerCore.forward()` now requires `z` parameter (default: 1.0j)
- `ScatteringRouter.forward()` returns tuple `(expert_indices, routing_weights)` instead of single tensor
- `SemiseparableMatrix.factorize()` now returns `(T, U, V)` instead of `(tridiag, lowrank)`

### New Features

- Automatic mixed-precision training (AMP) enabled by default
- Enhanced failure recovery with automatic hyperparameter adjustment
- Improved Hugging Face integration with pipeline support
- TensorRT optimization for 3× inference speedup

### Deprecations

- `use_hybrid_gradient` parameter deprecated (use `grad_blend` instead)
- `manual_seed` parameter deprecated (use `torch.manual_seed()` directly)
- Old checkpoint format will be removed in 2.0.0

---

## Migrating to 0.9.0

### Breaking Changes

#### 1. Hugging Face Integration

Models are now compatible with `transformers` library.

**Before (0.8.x):**
```python
from src.models.resnet_bk import LanguageModel

model = LanguageModel(vocab_size=30000, d_model=256)
```

**After (0.9.0):**
```python
from transformers import AutoModel

model = AutoModel.from_pretrained('resnet-bk/resnet-bk-1b')
```

#### 2. ONNX Export

New export functionality requires additional dependencies.

**Installation:**
```bash
pip install onnx onnxruntime tensorrt
```

**Usage:**
```python
from src.models.onnx_export import export_to_onnx

export_to_onnx(model, 'model.onnx', input_shape=(1, 512))
```

### New Features

- PyTorch Hub integration: `torch.hub.load('user/resnet-bk', 'resnet_bk_1b')`
- Pre-trained checkpoints on Hugging Face Hub
- Theoretical verification suite
- Epsilon-parametrized model family
- Koopman operator compression

### Deprecations

None

---

## Migrating to 0.8.0

### Breaking Changes

#### 1. ACT Module Integration

Adaptive Computation Time is now integrated into the main model.

**Before (0.7.x):**
```python
model = LanguageModel(vocab_size=30000, d_model=256)
# ACT not available
```

**After (0.8.0):**
```python
model = LanguageModel(
    vocab_size=30000,
    d_model=256,
    use_act=True,
    act_halt_threshold=0.2
)
```

#### 2. FLOPs Counter API

FLOPs counter now requires explicit initialization.

**Before (0.7.x):**
```python
# FLOPs counted automatically
```

**After (0.8.0):**
```python
from src.benchmarks.flops_counter import FLOPsCounter

counter = FLOPsCounter(model)
with counter:
    output = model(input_ids)
print(f"FLOPs: {counter.total_flops}")
```

### New Features

- Learned sparsity for G_ii computation
- Multi-scale processing
- Dynamic efficiency benchmarks
- Scattering-based halting for ACT

### Deprecations

None

---

## Migrating to 0.7.0

### Breaking Changes

#### 1. Quantization API

Quantization is now a separate module.

**Before (0.6.x):**
```python
# Quantization not available
```

**After (0.7.0):**
```python
from src.models.quantized_birman_schwinger import QuantizedBirmanSchwinger

quantized_model = QuantizedBirmanSchwinger(model, bits=8)
```

#### 2. Mixed-Precision Quantization

New mixed-precision strategy requires configuration.

**Usage:**
```python
from src.models.mixed_precision_quantization import MixedPrecisionQuantization

quant_config = {
    'experts': 4,  # INT4 for experts
    'bk_core': 8,  # INT8 for BK-Core
    'output': 16   # FP16 for output
}
mixed_quant = MixedPrecisionQuantization(model, quant_config)
```

### New Features

- Post-training quantization (PTQ)
- Quantization-aware training (QAT)
- INT4/INT8 support
- Group-wise quantization
- Quantization robustness benchmarks

### Deprecations

None

---

## Migrating to 0.6.0

### Breaking Changes

#### 1. Long-Context Support

Sequence length parameter now supports up to 1M tokens.

**Before (0.5.x):**
```python
model = LanguageModel(n_seq=2048)  # Max 2048
```

**After (0.6.0):**
```python
model = LanguageModel(n_seq=131072)  # Up to 1M
```

#### 2. Streaming Evaluation

New streaming evaluator for ultra-long sequences.

**Usage:**
```python
from src.benchmarks.streaming_evaluator import StreamingEvaluator

evaluator = StreamingEvaluator(model, chunk_size=8192)
ppl = evaluator.evaluate(dataset, max_length=1_000_000)
```

### New Features

- Long-context training infrastructure
- Mamba baseline for comparison
- Fair FLOPs measurement
- Gradient stability monitoring
- Loss spike detection

### Deprecations

None

---

## Migrating to 0.5.0

### Breaking Changes

#### 1. Semiseparable Structure

BK-Core now uses semiseparable matrix structure by default.

**Before (0.4.x):**
```python
model = LanguageModel(d_model=256)
# Dense matrices used
```

**After (0.5.0):**
```python
model = LanguageModel(
    d_model=256,
    use_semiseparable=True,  # Default
    low_rank=None  # Auto: log(n_seq)
)
```

To disable (not recommended):
```python
model = LanguageModel(d_model=256, use_semiseparable=False)
```

#### 2. Memory Optimization

New memory optimization strategies require configuration.

**Usage:**
```python
from src.models.memory_optimization import MemoryOptimizer

optimizer = MemoryOptimizer(
    use_zero=True,
    use_cpu_offload=True,
    use_gradient_checkpointing=True
)
model = optimizer.optimize(model)
```

### New Features

- Semiseparable matrix factorization
- ZeRO optimizer integration
- CPU offloading for low-rank factors
- Hierarchical semiseparable structure
- 70% memory reduction vs dense attention

### Deprecations

- Dense matrix mode will be removed in 1.0.0

---

## General Migration Tips

### 1. Check Dependencies

Always update dependencies when migrating:
```bash
pip install -r requirements.txt --upgrade
```

### 2. Test Before Deploying

Run tests after migration:
```bash
pytest tests/ -v
```

### 3. Backup Checkpoints

Always backup checkpoints before migration:
```bash
cp -r checkpoints/ checkpoints_backup/
```

### 4. Review Configuration

Check configuration files for deprecated parameters:
```bash
python scripts/check_config.py configs/base_config.yaml
```

### 5. Consult Documentation

See full documentation at: https://resnet-bk.readthedocs.io

---

## Getting Help

If you encounter issues during migration:

1. Check the [FAQ](FAQ.md)
2. Search [GitHub Issues](https://github.com/neko-jpg/Project-ResNet-BK-An-O-N-Language-Model-Architecture/issues)
3. Ask in [GitHub Discussions](https://github.com/neko-jpg/Project-ResNet-BK-An-O-N-Language-Model-Architecture/discussions)
4. Join our [Discord](https://discord.gg/resnet-bk)

---

## Version Support Policy

- **Current version (1.0.x)**: Full support
- **Previous major version (0.9.x)**: Security fixes only
- **Older versions**: No support

We recommend upgrading to the latest version for best performance and features.
