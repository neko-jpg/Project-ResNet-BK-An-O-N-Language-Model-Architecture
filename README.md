# ResNet-BK: A Mathematically Rigorous O(N) Language Model

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

**An experimental language model architecture exploring mathematical operator theory for improved long-context stability.**

---

## 🤝 Introduction

I am a first-year undergraduate student in business administration. While existing models like Transformer and Mamba have achieved remarkable results, I wondered whether there might be room for improvement from a mathematical physics perspective, particularly in memory efficiency and long-context stability. This led me to start this project.

**ResNet-BK** is an experimental implementation of a new O(N) language model architecture that applies Birman-Schwinger operator theory and spectral analysis of the Riemann zeta function.

### Current Status

In initial experiments on an RTX 3080 (10GB) environment, this model maintained stable training under conditions where Mamba encountered memory errors. However, this is only a first step, and much work remains to be done.

---

## 🚀 What We Aim For

Rather than simply "replacing existing models," we aim to technically support **"AI democratization"** by enabling anyone to stably train and run models with billions of parameters on consumer GPUs (RTX 3080/4090, etc.).

### Key Features (Initial Experimental Results)

- **Theoretical Stability**: Design based on mathematical proofs (Mourre estimate, etc.) to suppress gradient explosion
- **Memory Efficiency**: Lightweight design that can run on consumer GPUs
- **Physics-Based Routing**: MoE router based on scattering theory with no learnable parameters

---

## 📊 Preliminary Results

Initial comparative experiments suggest the following possibilities:

| Metric | ResNet-BK (Ours) | Mamba (Baseline) | Note |
|--------|------------------|------------------|------|
| Stability | ✅ Stable (Loss: 10.82→10.59) | ⚠️ CUDA Error / Unstable | RTX 3080 (10GB), Seq=2048 |
| Complexity | O(N log N) (Memory) | O(N) | Semiseparable Matrix Structure |

**Note**: These are preliminary results from initial experiments and require comprehensive validation with larger datasets and multiple runs.

---

## 🙋‍♀️ Call for Collaboration

As a first-year undergraduate student, I need the community's help with implementation skills, computational resources, and theoretical verification. I would especially appreciate discussions with those who have expertise in the following areas:

### Areas Where We Need Help

1. **Large-Scale Training Verification**
   - Help with validation on larger datasets and parameter sizes
   - Computational resource contributions

2. **CUDA Kernel Optimization**
   - Currently implemented using PyTorch standard features
   - Expertise in custom kernel implementation with Triton or CUDA for further acceleration

3. **Theoretical Feedback**
   - Feedback on the mathematical validity of the model from experts in mathematical physics and operator theory

4. **Benchmark Validation**
   - Independent verification of our experimental results
   - Comparison with other baseline models

### Contact

Please feel free to reach out via Issues, Discussions, or email. Any advice, no matter how small, is welcome!

- **Issues**: [GitHub Issues](https://github.com/neko-jpg/Project-ResNet-BK-An-O-N-Language-Model-Architecture/issues)
- **Discussions**: [GitHub Discussions](https://github.com/neko-jpg/Project-ResNet-BK-An-O-N-Language-Model-Architecture/discussions)
- **Email**: arat252539@gmail.com

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/neko-jpg/Project-ResNet-BK-An-O-N-Language-Model-Architecture.git
cd Project-ResNet-BK-An-O-N-Language-Model-Architecture

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Basic Usage

```python
import torch
from src.models.resnet_bk import LanguageModel

# Create model
model = LanguageModel(
    vocab_size=50257,
    d_model=256,
    n_layers=6,
    n_seq=2048,
    num_experts=4,
    top_k=1
)

# Forward pass
input_ids = torch.randint(0, 50257, (2, 2048))
logits = model(input_ids)
```

### Training Example

```bash
# Train on WikiText-2 (small-scale experiment)
python train.py --config configs/base_config.yaml --dataset wikitext2

# Run local benchmark
python scripts/local_long_context_benchmark.py --seq-lengths 2048 4096 --seeds 42
```

---

## 🏗️ Architecture

ResNet-BK is built on three mathematical concepts:

### 1. Birman-Schwinger Operator Theory

The core computation uses the Birman-Schwinger kernel:

```
K_ε(z) = |V_ε|^{1/2} R_0(z) |V_ε|^{1/2}
```

This provides theoretical guarantees for:
- Hilbert-Schmidt bound: ||K_ε||_S2 ≤ (1/2)(Im z)^{-1/2} ||V_ε||_L2
- Trace-class bound: ||K_ε||_S1 ≤ (1/2)(Im z)^{-1} ||V_ε||_L1
- Mourre estimate: [H_0, iA] = I (optimal stability)

### 2. Prime-Bump Initialization

Initialize potential with prime number distribution:

```
V_ε(x) = Σ_p α_{p,k}(ε) ψ_ε(x - log p)
```

### 3. Scattering-Based Routing

Zero-parameter MoE routing using scattering phase:

```
δ_ε(λ) = arg(det_2(I + K_ε(λ + i0)))
```

---

## 📚 Documentation

### Core Documentation
- **[TUTORIAL.md](docs/TUTORIAL.md)** - Step-by-step training guide
- **[API_REFERENCE.md](docs/API_REFERENCE.md)** - Complete API documentation
- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** - Detailed design documentation
- **[FAQ.md](docs/FAQ.md)** - Troubleshooting and common questions

### Additional Resources
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - Common issues and solutions
- **[PERFORMANCE.md](PERFORMANCE.md)** - Performance optimization guide
- **[ROADMAP.md](ROADMAP.md)** - Development roadmap

### Mathematical Foundation

The theoretical foundations are documented in:
**"Riemann Hypothesis and AI: Emergent Theory"** by Teppei Arai  
📄 Available at:[[ [https://doi.org/10.5281/zenodo.17600573](https://doi.org/10.5281/zenodo.17600573)  ](https://doi.org/10.5281/zenodo.17346958)
License: CC BY-NC-ND 4.0

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run benchmarks
python scripts/local_long_context_benchmark.py
```

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
