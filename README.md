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
| Stability | ✅ Stable (Loss: 10.82→10.59) | ⚠️ CUDA Error / Unstable | RTX 3080 (8GB), Seq=2048 |
| Complexity | O(N log N) (Memory) | O(N) | Semiseparable Matrix Structure |

**Note**: These are preliminary results from initial experiments and require comprehensive validation with larger datasets and multiple runs.

---

## 🙋‍♀️ Call for Collaboration & Contributing

This project is in an early, experimental stage. As a first-year undergraduate student, I welcome and need the community's help to validate, improve, and extend this work. We are looking for collaborators of all skill levels.

**How You Can Help:**
- **Large-Scale Training:** Help us test the model on larger datasets.
- **CUDA Kernel Optimization:** Optimize the core components with Triton or custom CUDA kernels.
- **Theoretical Feedback:** Provide feedback on the mathematical framework.
- **Benchmark Validation:** Independently verify our experimental results.

This is a community-driven project, and we welcome any contribution, from fixing a typo to implementing a new feature.

**Ready to contribute?** Please read our **[Contributing Guidelines](docs/CONTRIBUTING.md)** to get started. It contains detailed instructions on our development setup, coding standards, and pull request process.

### Contact

Please feel free to reach out via Issues, Discussions, or email.
- **Issues**: [GitHub Issues](https://github.com/neko-jpg/Project-ResNet-BK-An-O-N-Language-Model-Architecture/issues)
- **Discussions**: [GitHub Discussions](https://github.com/neko-jpg/Project-ResNet-BK-An-O-N-Language-Model-Architecture/discussions)
- **Email**: arat252539@gmail.com

---

## 🚀 Quick Start

### Installation

We provide a simple script to set up a local development environment.

```bash
# Clone the repository and navigate into it
git clone https://github.com/neko-jpg/Project-ResNet-BK-An-O-N-Language-Model-Architecture.git
cd Project-ResNet-BK-An-O-N-Language-Model-Architecture

# Run the setup script
bash scripts/setup_dev.sh

# Activate the virtual environment
source .venv/bin/activate
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
)

# Forward pass
input_ids = torch.randint(0, 50257, (2, 2048))
logits = model(input_ids)
print(logits.shape)
```

### Training & Benchmarking

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
