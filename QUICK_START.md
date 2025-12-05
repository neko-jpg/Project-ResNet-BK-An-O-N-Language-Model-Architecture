# Quick Start Guide

Train a 10B Japanese LLM on your RTX 3080 (8GB VRAM).

## Prerequisites

- WSL Ubuntu (recommended) or Linux
- NVIDIA GPU with 8GB+ VRAM
- Python 3.10+
- CUDA 11.8+

## Setup

```bash
# 1. Clone
git clone https://github.com/neko-jpg/Project-ResNet-BK-An-O-N-Language-Model-Architecture.git
cd Project-ResNet-BK-An-O-N-Language-Model-Architecture

# 2. Enter WSL
wsl -d ubuntu

# 3. Create venv
python3 -m venv venv_ubuntu
source venv_ubuntu/bin/activate

# 4. Install
pip install -r requirements.txt
```

## Training

```bash
# Full pipeline (download data + train)
make start-japanese

# Or step by step:
make prepare-japanese-data   # Download Japanese datasets
make dry-run-japanese        # Test config first
make train-japanese          # Start training
```

## Resume Training

```bash
# Resume from latest checkpoint
make resume-japanese

# See all checkpoints
make list-checkpoints
```

## Chat

```bash
# After training, chat with the model
make chat
```

## Commands

| Command | Description |
|---------|-------------|
| `make start-japanese` | Full: data + train |
| `make dry-run-japanese` | Test config |
| `make resume-japanese` | Resume training |
| `make list-checkpoints` | Show checkpoints |
| `make chat` | Chat with model |
| `make export-model` | Export for deployment |
| `make recipe` | Training wizard |
| `make test` | Run tests |
| `make help` | Show all commands |

## Docker

```bash
docker-compose up -d
docker-compose exec muse make start-japanese
```

## Troubleshooting

### Out of Memory
```bash
# Use extreme compression
make train-10b-8gb
```

### NaN during training
- Already handled by Safe-Log Triton kernels
- If persists, reduce learning rate in config

### Slow training
- Enable torch.compile: `--compile` flag
- Ensure Flash Attention 2 is installed

## Expected Training Time

| Steps | Time (RTX 3080) |
|-------|-----------------|
| 1K | ~2 hours |
| 10K | ~20 hours |
| 50K | ~100 hours |

## Next Steps

1. Train the model: `make start-japanese`
2. Chat: `make chat`
3. Export: `make export-model`
4. Fine-tune on your own data
