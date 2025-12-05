# MUSE: 10B Japanese LLM on Consumer GPU

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)

**Train a 10B parameter Japanese LLM on RTX 3080 (8GB VRAM) using novel compression techniques.**

![Architecture](docs/images/architecture.png)

---

## 🎯 Quick Start

```bash
# 1. Clone
git clone https://github.com/neko-jpg/Project-ResNet-BK-An-O-N-Language-Model-Architecture.git
cd Project-ResNet-BK-An-O-N-Language-Model-Architecture

# 2. Setup (WSL Ubuntu recommended)
wsl -d ubuntu
python3 -m venv venv_ubuntu && source venv_ubuntu/bin/activate
pip install -r requirements.txt

# 3. Train Japanese 10B LLM
make start-japanese
```

**That's it!** The model will:
- Download Japanese datasets (Wikipedia, Dolly, Alpaca)
- Train with Phase 8 optimizations
- Save checkpoints every 500 steps

---

## 📊 Key Features

| Feature | Description |
|---------|-------------|
| 🧠 **10B Parameters** | Dense equivalent ~10B, stored as ~300M (97% compression) |
| 🇯🇵 **Japanese Native** | rinna tokenizer, JP datasets (Wiki, Dolly, Alpaca) |
| 💾 **8GB VRAM** | Runs on RTX 3080, RTX 4070, etc. |
| ⚡ **Phase 8 Stack** | BK-Core + Hyperbolic Attention + BitNet + Triton |
| 🔄 **Resume Support** | Auto-checkpoint every 500 steps |

---

## 🛠 Commands

```bash
make help                  # Show all commands

# Training
make start-japanese        # Full pipeline: data + train
make dry-run-japanese      # Test config (no training)
make resume-japanese       # Resume from latest checkpoint

# Checkpoints
make list-checkpoints      # Show saved models
make resume CHECKPOINT=path # Resume from specific file

# Chat
make chat                  # Interactive chat with model
make export-model          # Export for deployment

# Utils
make setup                 # Install dependencies
make recipe                # Training wizard
make test                  # Run tests
make clean                 # Clean caches
```

---

## 🏗 Architecture (Phase 8)

```
MUSE Architecture
├── BK-Core (Birman-Schwinger Scattering)
│   └── O(N) Green's function computation
├── Hyperbolic Attention
│   └── Poincaré ball for hierarchical structure
├── Low-Rank Compression (r=16)
│   └── 97% parameter reduction
├── BitNet 1.58-bit
│   └── Ternary weights {-1, 0, 1}
├── Triton Safe-Log Kernels
│   └── NaN-proof numerical stability
└── Muon Optimizer
    └── High learning rate (0.02) training
```

### Parameter Calculation

| Metric | Value |
|--------|-------|
| Dense Equivalent | ~9.87B |
| Actual Parameters | ~311M |
| Compression | 97% |
| VRAM Usage | ~7GB |

---

## 📁 Project Structure

```
├── configs/
│   ├── phase8_10b_japanese.yaml  # Japanese model config
│   └── dataset_japanese.yaml     # Dataset mixing
├── scripts/
│   ├── train_phase8.py           # Main training script
│   ├── chat_inference.py         # Chat interface
│   ├── prepare_japanese_data.py  # Data downloader
│   └── configure_recipe.py       # Training wizard
├── src/
│   ├── models/
│   │   ├── resnet_bk.py          # Main model
│   │   ├── bk_core.py            # BK-Core implementation
│   │   └── phase7/               # Hyperbolic attention
│   └── kernels/
│       └── safe_ops_triton.py    # Triton kernels
└── checkpoints/
    └── phase8_10b_japanese/      # Saved models
```

---

## 🐳 Docker

```bash
# Build
docker build -t muse-llm .

# Run with GPU
docker run --gpus all -it muse-llm bash

# Or use docker-compose
docker-compose up -d
docker exec -it muse-dev bash
```

---

## 🔬 Technical Details

### Novel Contributions

1. **BK-Core Scattering**: Applies quantum scattering theory to attention
2. **Hyperbolic Geometry**: Poincaré ball for natural hierarchy representation
3. **Extreme Compression**: 10B → 300M with minimal quality loss
4. **Safe Triton Kernels**: NaN-proof log/exp operations

### Training Configuration

```yaml
# configs/phase8_10b_japanese.yaml
d_model: 4096
n_layers: 48
vocab_size: 32000  # Japanese tokenizer
low_rank_rank: 16
use_bitnet: true
use_gradient_checkpointing: true
use_torch_compile: true
```

---

## 📈 Expected Results

| Stage | Loss | PPL | Time (RTX 3080) |
|-------|------|-----|-----------------|
| Start | ~10 | ~22000 | 0h |
| 1K steps | ~5 | ~150 | ~2h |
| 10K steps | ~3 | ~20 | ~20h |
| 50K steps | ~2 | ~8 | ~100h |

*Estimates based on similar architectures*

---

## 🤝 Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).

Areas of interest:
- Training optimization
- Japanese dataset curation
- Inference speedup
- Documentation

---

## 📖 Citation

```bibtex
@misc{muse2025,
  title={MUSE: 10B Japanese LLM on Consumer GPU},
  author={Teppei Arai},
  year={2025},
  howpublished={\url{https://github.com/neko-jpg/Project-ResNet-BK-An-O-N-Language-Model-Architecture}}
}
```

---

## 📜 License

MIT License - see [LICENSE](LICENSE)

---

**Made with ❤️ by Teppei Arai**

*Train your own 10B LLM today! 🚀*
