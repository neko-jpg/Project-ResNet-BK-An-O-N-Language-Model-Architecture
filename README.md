# ResNet-BK: A Mathematically Rigorous O(N) Language Model

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![Paper](https://img.shields.io/badge/paper-PDF-red.svg)](paper/main.pdf)

**A memory-efficient language model architecture based on Birman-Schwinger operator theory, achieving 93% memory reduction and 185× speedup through mathematical rigor.**

---

## 🎯 Quick Start

```bash
# Clone and setup
git clone https://github.com/neko-jpg/Project-ResNet-BK-An-O-N-Language-Model-Architecture.git
cd Project-ResNet-BK-An-O-N-Language-Model-Architecture

# Option 1: Docker (Recommended)
docker-compose up -d
docker exec -it mamba-killer-dev bash

# Option 2: Local installation
pip install -r requirements.txt
pip install -e .

# Run tests
pytest tests/ -v
```

See [QUICK_START.md](QUICK_START.md) for detailed instructions.

---

## 🤝 Introduction

I am a first-year undergraduate student in business administration. While existing models like Transformer and Mamba have achieved remarkable results, I wondered whether there might be room for improvement from a mathematical physics perspective, particularly in memory efficiency and long-context stability. This led me to start this project.

**ResNet-BK** is an experimental implementation of a new O(N) language model architecture that applies:
- **Birman-Schwinger operator theory** for numerical stability
- **Semiseparable matrix structure** for O(N log N) memory complexity
- **Riemann zeta function** for initialization and memory resonance
- **Non-Hermitian dynamics** for adaptive forgetting

---

## 🚀 Key Features

### Phase 1: Efficiency Engine
- **HTT Embedding**: 99.6% parameter compression via Tensor Train decomposition
- **AR-SSM Layers**: O(N) sequence processing with adaptive rank
- **BK-Core**: Semiseparable structure achieving 610× parameter reduction
- **Triton Kernels**: 185× speedup over PyTorch baseline

### Phase 2: Breath of Life (Dynamic Memory)
- **Non-Hermitian Potential**: Adaptive forgetting with decay rate Γ
- **Dissipative Hebbian**: Fast weights with Lyapunov stability
- **SNR Memory Filter**: Signal-to-noise ratio based memory selection
- **Memory Resonance**: Zeta-based frequency filtering

---

## 📊 Experimental Results

### Memory Efficiency

| Configuration | Parameters | VRAM (GB) | Reduction |
|---------------|-----------|-----------|-----------|
| Baseline | 1.62B | 6.89 | -- |
| Phase 1 | 1.26B | 5.14 | 25.4% |
| **Phase 2** | **0.11B** | **0.48** | **93.0%** |

*Inference mode, d_model=4096, 6 layers, FP16*

### Performance

| Metric | Baseline | Phase 1 | Improvement |
|--------|----------|---------|-------------|
| Throughput | 798.28 tok/s | 824.74 tok/s | +3.3% |
| Perplexity | 50738.89 | 50505.61 | -0.46% |
| Complexity | O(N²) | O(N log N) | Memory |

### BK-Core Triton Kernel

| Implementation | Time (ms) | Speedup |
|----------------|-----------|---------|
| PyTorch (vmap) | 554.18 | 1.00× |
| **Triton Kernel** | **2.99** | **185.10×** |

*Batch=16, Seq=4096, RTX 3080*

### Phase 2 Stability

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Warnings | 107 | 8 | 92.5% |
| Lyapunov Violations | 630 | 0 | 100% |
| Test Status | -- | PASSED | ✓ |

---

## 🏗️ Architecture Overview

```
ResNet-BK Architecture
├── Phase 1: Efficiency Engine
│   ├── HTT Embedding (99.6% compression)
│   ├── AR-SSM Layers (O(N) complexity)
│   ├── BK-Core (Semiseparable structure)
│   └── Triton Kernels (185× speedup)
│
└── Phase 2: Breath of Life
    ├── Non-Hermitian Potential (Γ = 0.001)
    ├── Dissipative Hebbian (Fast weights)
    ├── SNR Memory Filter (τ = 2.0)
    └── Memory Resonance (Zeta regularization)
```

---

## 📚 Documentation

- **[Quick Start Guide](QUICK_START.md)**: Get started in 5 minutes
- **[Contributing Guide](CONTRIBUTING.md)**: How to contribute
- **[Paper](paper/main.pdf)**: Full theoretical background (20 pages)
- **[Phase 1 Guide](docs/PHASE1_IMPLEMENTATION_GUIDE.md)**: Phase 1 implementation details
- **[Phase 2 Guide](docs/PHASE2_IMPLEMENTATION_GUIDE.md)**: Phase 2 implementation details
- **[Performance Analysis](PERFORMANCE.md)**: Detailed performance metrics

---

## 🙋‍♀️ Call for Collaboration

This project is in an experimental stage. We welcome collaborators of all skill levels:

- **Researchers**: Validate theoretical claims, propose improvements
- **Engineers**: Optimize implementations, add features
- **Students**: Learn and contribute to cutting-edge research
- **Users**: Test on real tasks, report issues

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

---

## 💻 Installation

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (for GPU support)
- 8GB+ VRAM (RTX 3080 or better recommended)

### Option 1: Docker (Recommended)

```bash
# Clone repository
git clone https://github.com/neko-jpg/Project-ResNet-BK-An-O-N-Language-Model-Architecture.git
cd Project-ResNet-BK-An-O-N-Language-Model-Architecture

# Build and start container
docker-compose up -d

# Enter container
docker exec -it mamba-killer-dev bash

# Run tests
pytest tests/ -v
```

### Option 2: Local Installation

```bash
# Clone repository
git clone https://github.com/neko-jpg/Project-ResNet-BK-An-O-N-Language-Model-Architecture.git
cd Project-ResNet-BK-An-O-N-Language-Model-Architecture

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .

# Run tests
pytest tests/ -v
```

---

## 🎮 Usage Examples

### Phase 1: Memory-Efficient Model

```python
from src.models.phase1.factory import create_phase1_model
import torch

# Create model
model = create_phase1_model(preset="small", device="cuda")

# Generate
input_ids = torch.randint(0, model.vocab_size, (1, 512)).cuda()
output = model(input_ids)

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Output shape: {output.shape}")
```

### Phase 2: Dynamic Memory Model

```python
from src.models.phase2.factory import create_phase2_model
import torch

# Create model with dynamic memory
model = create_phase2_model(preset="small", device="cuda")

# Generate
input_ids = torch.randint(0, model.vocab_size, (1, 512)).cuda()
output = model(input_ids)

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### BK-Core Computation

```python
from src.models.bk_core import BKCoreFunction
import torch

# Setup
h_diag = torch.randn(4, 512).cuda()
h_super = torch.randn(4, 511).cuda()
h_sub = torch.randn(4, 511).cuda()
z = torch.tensor(0.1 + 0.1j).cuda()

# Compute Green's function
g = BKCoreFunction.apply(h_diag, h_super, h_sub, z, use_triton=True)
print(f"Green's function shape: {g.shape}")
```

---

## 🧪 Running Experiments

### Benchmarks

```bash
# Memory benchmark
python scripts/validate_phase1_memory.py

# Throughput benchmark
python scripts/benchmark_phase1_throughput.py

# BK-Core Triton benchmark
python scripts/benchmark_bk_triton.py

# Phase 2 integration test
pytest tests/test_phase2_integration.py -v
```

### Training

```bash
# Phase 1 training
python scripts/train_phase1.py --config configs/phase1_small.yaml

# Phase 2 training
python scripts/train_phase2.py --config configs/phase2_small.yaml
```

---

## 📖 Citation

If you use this work in your research, please cite:

```bibtex
@article{arai2025resnetbk,
  title={ResNet-BK: A Memory-Efficient Language Model Based on Birman-Schwinger Operator Theory},
  author={Arai, Teppei},
  journal={arXiv preprint},
  year={2025}
}
```

---

## 📞 Contact

- **GitHub Issues**: [Report bugs or request features](https://github.com/neko-jpg/Project-ResNet-BK-An-O-N-Language-Model-Architecture/issues)
- **Email**: arat252539@gmail.com
- **Paper**: [paper/main.pdf](paper/main.pdf)

---

## 🙏 Acknowledgments

- **Mathematical Foundations**: M.Sh. Birman, J. Schwinger, E. Mourre
- **Open Source Community**: PyTorch, Hugging Face, Triton
- **AI Assistance**: Claude (Anthropic), Kiro IDE

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## ⭐ Star History

If you find this project useful, please consider giving it a star! ⭐

---

**Made with ❤️ by Teppei Arai and contributors**

```bash
# Set PYTHONPATH to include the project root
export PYTHONPATH=$PYTHONPATH:.

# Run a small-scale training experiment on WikiText-2
python scripts/train.py --config configs/base_config.yaml

# Run the local efficiency benchmark
python scripts/local_efficiency_benchmark.py --train-steps 10 --seq-length 1024
```

For more details on benchmarking, see the "Benchmarking" section below.

---

## 🏗️ Architecture & Mathematical Foundation

ResNet-BK is built on three core mathematical concepts from operator theory and quantum physics. The goal is to build a model where stability and efficiency are guaranteed by the underlying mathematics.

1.  **Birman-Schwinger Operator Theory**: The model's core uses a kernel based on the Birman-Schwinger principle, which provides proven bounds on the operator's properties, ensuring stability.
2.  **Prime-Bump Initialization**: The model's potential is initialized based on the distribution of prime numbers, a technique inspired by the Riemann zeta function.
3.  **Scattering-Based Routing**: A zero-parameter MoE router that uses the physical concept of a scattering phase to route information, eliminating the need for learnable router parameters.

For a detailed but accessible explanation of these concepts, please read our **[Mathematical Foundations Guide](docs/THEORY.md)**.

The full theoretical proofs are available in the paper:
- **"Riemann Hypothesis and AI: Emergent Theory"** by Teppei Arai
- 📄 Available at: https://doi.org/10.5281/zenodo.17346958 (License: CC BY-NC-ND 4.0)
-　However, the file riemann_hypothesis_main.tex within this repository is licensed under the MIT License, so please feel free to use it as you wish.
---

## 📊 Benchmarking

We provide an enhanced benchmarking script that compares ResNet-BK against a Mamba baseline.

### Running the Benchmark

The script is highly configurable. You can easily test different model sizes and sequence lengths.

```bash
# Set PYTHONPATH to include the project root
export PYTHONPATH=$PYTHONPATH:.

# Run with custom model dimensions and sequence length
python scripts/local_efficiency_benchmark.py \
  --seq-length 4096 \
  --d-model 512 \
  --n-layers 8 \
  --batch-size 1
```

### Output

The script provides a progress bar, a summary table in the console, and saves a detailed JSON file to `results/benchmarks/` with a unique, timestamped filename.

```
--- Benchmark Results ---
Configuration: sequence_length=4096, d_model=512, n_layers=8
--------------------------------------------------
Model                | Avg FLOPs (GFLOPs)   | Final Loss
--------------------------------------------------
ResNet-BK            | 123.4567             | 9.8765
ResNet-BK (GC)       | 123.5678             | 9.8888
Mamba                | 110.9876             | 9.9123
--------------------------------------------------
Results saved to: results/benchmarks/efficiency_seq4096_d512_l8_20240101-123000.json
```

---

## 📚 Documentation

- **[Contributing Guidelines](docs/CONTRIBUTING.md)**: Our main guide for contributors.
- **[Mathematical Foundations](docs/THEORY.md)**: An accessible guide to the core theory.
- **[Tutorial](docs/TUTORIAL.md)**: Step-by-step training guide.
- **[Roadmap](ROADMAP.md)**: Our development roadmap.

---

## 🗺️ Roadmap

- [x] Phase 1: Birman-Schwinger Core Implementation
- [x] Phase 2: Scattering-Based Router
- [x] Phase 3: Semiseparable Matrix Structure
- [x] Phase 4: Initial Stability Tests
- [ ] Phase 5: Large-Scale Validation
- [ ] Phase 6: Performance Optimization
- [ ] Phase 7: Community Feedback Integration
- [ ] Phase 8: Paper Preparation

---

## 📖 Citation

If you use ResNet-BK in your research, please cite:

```bibtex
@misc{resnetbk2025,
  title={ResNet-BK: An Experimental O(N) Language Model via Birman-Schwinger Theory},
  author={Teppei Arai},
  year={2025},
  howpublished={\url{https://github.com/neko-jpg/Project-ResNet-BK-An-O-N-Language-Model-Architecture}}
}
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Third-Party Licenses

See [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) for complete list of open-source libraries used.

---

## 🙏 Acknowledgments

This project builds on foundational work in:
- **Quantum Scattering Theory**: Newton (1982), Mourre (1981)
- **Birman-Schwinger Operator**: Birman & Schwinger (1962), Reed & Simon (1979)
- **State Space Models**: Gu et al. (S4, Mamba)

We thank the open-source community and all contributors who have helped make this project possible.

---

**Made with curiosity and mathematical rigor**
