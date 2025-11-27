# Quick Start Guide

Get up and running with ResNet-BK in 5 minutes!

## 🚀 Installation

### Option 1: Docker (Recommended)

The easiest way to get started:

```bash
# Clone the repository
git clone https://github.com/neko-jpg/Project-ResNet-BK-An-O-N-Language-Model-Architecture.git
cd Project-ResNet-BK-An-O-N-Language-Model-Architecture

# Build and start Docker container
docker-compose up -d

# Enter the container
docker exec -it mamba-killer-dev bash

# You're ready to go!
pytest tests/ --tb=short
```

### Option 2: Local Installation

If you prefer local installation:

```bash
# Clone the repository
git clone https://github.com/neko-jpg/Project-ResNet-BK-An-O-N-Language-Model-Architecture.git
cd Project-ResNet-BK-An-O-N-Language-Model-Architecture

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .

# Verify installation
pytest tests/ --tb=short
```

## 🎯 Quick Examples

### Example 1: Phase 1 Model (Memory Efficient)

```python
from src.models.phase1.factory import create_phase1_model

# Create a small Phase 1 model
model = create_phase1_model(
    preset="small",
    device="cuda"
)

# Generate text
import torch
input_ids = torch.randint(0, model.vocab_size, (1, 128)).cuda()
output = model(input_ids)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Output shape: {output.shape}")
```

### Example 2: Phase 2 Model (Dynamic Memory)

```python
from src.models.phase2.factory import create_phase2_model

# Create a small Phase 2 model
model = create_phase2_model(
    preset="small",
    device="cuda"
)

# Generate text
import torch
input_ids = torch.randint(0, model.vocab_size, (1, 128)).cuda()
output = model(input_ids)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Output shape: {output.shape}")
```

### Example 3: BK-Core Computation

```python
import torch
from src.models.bk_core import BKCoreFunction

# Create tridiagonal matrix
batch_size = 4
seq_len = 512

h_diag = torch.randn(batch_size, seq_len).cuda()
h_super = torch.randn(batch_size, seq_len - 1).cuda()
h_sub = torch.randn(batch_size, seq_len - 1).cuda()
z = torch.tensor(0.1 + 0.1j).cuda()

# Compute Green's function diagonal
g_diag = BKCoreFunction.apply(h_diag, h_super, h_sub, z, use_triton=True)

print(f"Green's function shape: {g_diag.shape}")
print(f"Is complex: {g_diag.is_complex()}")
```

## 📊 Running Benchmarks

### Memory Benchmark

```bash
python scripts/validate_phase1_memory.py
```

Expected output:
```
Phase 1 Memory Validation
========================
Baseline Peak VRAM: 1902.39 MB
Phase 1 Peak VRAM: 1810.39 MB
Reduction: 4.8%
✓ PASS
```

### Throughput Benchmark

```bash
python scripts/benchmark_phase1_throughput.py
```

Expected output:
```
Phase 1 Throughput Benchmark
============================
Baseline: 798.28 tokens/sec
Phase 1: 824.74 tokens/sec
Improvement: 3.3%
✓ PASS
```

### BK-Core Triton Benchmark

```bash
python scripts/benchmark_bk_triton.py
```

Expected output:
```
BK-Core Triton Benchmark
========================
PyTorch (vmap): 554.18 ms
Triton Kernel: 2.99 ms
Speedup: 185.10×
✓ PASS (Target: 3.0×)
```

## 🧪 Running Tests

### All Tests

```bash
pytest tests/ -v
```

### Specific Test Suites

```bash
# Phase 1 tests
pytest tests/test_phase1_integration.py -v

# Phase 2 tests
pytest tests/test_phase2_integration.py -v

# BK-Core tests
pytest tests/test_bk_triton.py -v
```

### With Coverage

```bash
pytest --cov=src tests/
```

## 📈 Training a Model

### 🚀 Phase 7 Chat AI Training (1.8B Monster) - Recommended

8GB VRAMで1.8Bパラメータのチャットモデルを訓練！

```bash
# WSL Ubuntu環境で実行
wsl -d ubuntu

# 仮想環境を有効化
source venv_ubuntu/bin/activate

# データセットの準備（初回のみ）
make recipe

# 🚀 チャットAI訓練開始！
make train-chat

# または、ダミーデータでテスト
make train-chat-test
```

設定ファイル: `configs/phase7_max_push.yaml`
- d_model: 4096
- n_layers: 32
- seq_len: 512
- VRAM使用量: ~6.89GB

### Phase 1 Training

```bash
python scripts/train_phase1.py \
    --model_size small \
    --batch_size 4 \
    --seq_length 512 \
    --num_epochs 10 \
    --output_dir checkpoints/phase1_small
```

### Phase 2 Training

```bash
python scripts/train_phase2.py \
    --model_size small \
    --batch_size 4 \
    --seq_length 512 \
    --num_epochs 10 \
    --output_dir checkpoints/phase2_small
```

## 🔍 Exploring Examples

We provide many example scripts in the `examples/` directory:

```bash
# Phase 1 examples
python examples/phase1_integration_demo.py
python examples/htt_compression_demo.py
python examples/demo_ar_ssm.py

# Phase 2 examples
python examples/phase2_basic_usage.py
python examples/phase2_training_demo.py
python examples/non_hermitian_demo.py
python examples/memory_resonance_demo.py

# BK-Core examples
python examples/bk_triton_demo.py
```

## 📚 Next Steps

- Read the [full README](README.md) for detailed information
- Check [CONTRIBUTING.md](CONTRIBUTING.md) to contribute
- Explore [docs/](docs/) for implementation details
- Review [paper/main.pdf](paper/main.pdf) for theoretical background

## 🐛 Troubleshooting

### CUDA Out of Memory

If you encounter CUDA OOM errors:

```python
# Reduce batch size
model = create_phase1_model(preset="small", device="cuda")
batch_size = 2  # Instead of 4

# Or use gradient checkpointing
model.gradient_checkpointing_enable()
```

### Import Errors

If you get import errors:

```bash
# Make sure you installed in development mode
pip install -e .

# Or add src to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

### Triton Not Available

If Triton is not available on your system:

```python
# Disable Triton and use PyTorch fallback
model = create_phase1_model(preset="small", use_triton=False)
```

## 💬 Getting Help

- **GitHub Issues**: https://github.com/neko-jpg/Project-ResNet-BK-An-O-N-Language-Model-Architecture/issues
- **Email**: arat252539@gmail.com

Happy experimenting! 🚀
