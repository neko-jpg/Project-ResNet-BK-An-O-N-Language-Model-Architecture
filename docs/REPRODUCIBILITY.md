# Reproducibility Guide: Mamba-Killer ResNet-BK

This guide provides complete instructions for reproducing all results from the Mamba-Killer ResNet-BK paper.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Environment Setup](#environment-setup)
3. [Dataset Preparation](#dataset-preparation)
4. [Training](#training)
5. [Evaluation](#evaluation)
6. [Generating Figures](#generating-figures)
7. [Troubleshooting](#troubleshooting)

## Quick Start

### Option 1: Google Colab (Recommended for Quick Start)

1. Open the Colab notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neko-jpg/Project-ResNet-BK-An-O-N-Language-Model-Architecture/blob/main/notebooks/colab_reproducibility.ipynb)

2. Run all cells in order

3. Results will be saved to your Google Drive

**Estimated Time:** 6-12 hours on T4 GPU (free tier)

### Option 2: Docker (Recommended for Local Reproduction)

```bash
# Clone repository
git clone https://github.com/neko-jpg/Project-ResNet-BK-An-O-N-Language-Model-Architecture.git
cd Project-ResNet-BK-An-O-N-Language-Model-Architecture

# Build Docker image
docker build -t mamba-killer:latest .

# Run container
docker run --gpus all -it -p 8888:8888 -p 6006:6006 \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/checkpoints:/workspace/checkpoints \
  mamba-killer:latest

# Inside container, run full pipeline
python scripts/run_full_pipeline.py
```

**Estimated Time:** 24-48 hours on single GPU

### Option 3: Manual Setup

See [Environment Setup](#environment-setup) below.

## Environment Setup

### Requirements

- **Hardware:**
  - GPU: NVIDIA GPU with ≥16GB VRAM (T4, V100, A100)
  - RAM: ≥32GB
  - Storage: ≥100GB free space

- **Software:**
  - Python 3.10+
  - CUDA 11.8+
  - PyTorch 2.1.0+

### Installation

```bash
# Clone repository
git clone https://github.com/neko-jpg/Project-ResNet-BK-An-O-N-Language-Model-Architecture.git
cd Project-ResNet-BK-An-O-N-Language-Model-Architecture

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Verify Installation

```bash
# Check GPU availability
python check_gpu.py

# Run tests
pytest tests/ -v

# Expected output: All tests pass
```

## Dataset Preparation

### Automatic Preparation (Recommended)

```bash
# Prepare all datasets
python scripts/prepare_datasets.py --all --output_dir ./data

# Or prepare specific datasets
python scripts/prepare_datasets.py \
  --datasets wikitext2 wikitext103 \
  --output_dir ./data
```

**Estimated Time:**
- WikiText-2: ~5 minutes
- WikiText-103: ~30 minutes
- C4 (100k samples): ~2 hours
- The Pile (50k samples): ~3 hours

### Manual Preparation

If automatic preparation fails, download datasets manually:

```bash
# WikiText-2
wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip
unzip wikitext-2-raw-v1.zip -d ./data/wikitext2

# WikiText-103
wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip
unzip wikitext-103-raw-v1.zip -d ./data/wikitext103
```

## Training

### Configuration Files

All experiments use YAML configuration files in `configs/`:

- `base_config.yaml`: Default configuration
- `long_context_config.yaml`: Long-context stability experiments
- `quantization_config.yaml`: Quantization robustness experiments
- `efficiency_config.yaml`: Dynamic efficiency experiments
- `mamba_comparison_config.yaml`: Fair comparison with Mamba
- `colab_config.yaml`: Optimized for Google Colab

### Training Commands

#### Base Model (WikiText-2)

```bash
python train.py \
  --config configs/base_config.yaml \
  --data_dir ./data \
  --checkpoint_dir ./checkpoints/base \
  --log_dir ./logs/base
```

**Estimated Time:** 12 hours on T4 GPU

#### Long-Context Model

```bash
python train.py \
  --config configs/long_context_config.yaml \
  --data_dir ./data \
  --checkpoint_dir ./checkpoints/long_context \
  --log_dir ./logs/long_context
```

**Estimated Time:** 48 hours on T4 GPU

#### Quantization-Aware Training

```bash
python train.py \
  --config configs/quantization_config.yaml \
  --data_dir ./data \
  --checkpoint_dir ./checkpoints/quantization \
  --log_dir ./logs/quantization
```

**Estimated Time:** 24 hours on T4 GPU

#### Efficiency Model (ACT + Sparsity)

```bash
python train.py \
  --config configs/efficiency_config.yaml \
  --data_dir ./data \
  --checkpoint_dir ./checkpoints/efficiency \
  --log_dir ./logs/efficiency
```

**Estimated Time:** 18 hours on T4 GPU

### Monitoring Training

#### TensorBoard

```bash
tensorboard --logdir ./logs
# Open http://localhost:6006
```

#### Weights & Biases

```bash
# Set API key
export WANDB_API_KEY=your_api_key

# Enable in config
# monitoring:
#   use_wandb: true
```

### Resuming Training

Training automatically resumes from the latest checkpoint:

```bash
# Just run the same command again
python train.py --config configs/base_config.yaml ...
```

## Evaluation

### Single Model Evaluation

```bash
python scripts/evaluate.py \
  --checkpoint ./checkpoints/base/best.pt \
  --dataset wikitext2 \
  --data_dir ./data \
  --output_dir ./results/base
```

### Multi-Dataset Evaluation

```bash
python scripts/evaluate.py \
  --checkpoint ./checkpoints/base/best.pt \
  --datasets wikitext2 wikitext103 c4 pile \
  --data_dir ./data \
  --output_dir ./results/base
```

### Mamba Comparison

```bash
python scripts/mamba_vs_bk_benchmark.py \
  --model bk \
  --checkpoint ./checkpoints/base/best.pt \
  --seq_len 2048 \
  --bits 32 \
  --dataset wikitext2 \
  --data_dir ./data \
  --output_dir ./results/comparison
```

**Note:** Requires Mamba checkpoint. Download from official repository or train separately.

## Generating Figures

### Individual Figures

```bash
# Long-context stability graph
python scripts/generate_stability_graph.py \
  --results_dir ./results \
  --output_dir ./figures

# Quantization robustness graph
python scripts/generate_quantization_graph.py \
  --results_dir ./results \
  --output_dir ./figures

# Dynamic efficiency graph
python scripts/generate_efficiency_graph.py \
  --results_dir ./results \
  --output_dir ./figures
```

### All Figures (Jupyter Notebook)

```bash
jupyter notebook notebooks/generate_killer_graphs.ipynb
```

### Publication-Quality Figures

Figures are automatically generated in multiple formats:
- PNG (300 DPI)
- PDF (vector)
- SVG (vector)
- EPS (vector, for LaTeX)

## Uploading to Hugging Face Hub

```bash
# Set API token
export HF_TOKEN=your_huggingface_token

# Upload checkpoint
python scripts/upload_to_hub.py \
  --checkpoint ./checkpoints/base/best.pt \
  --repo_id neko-jpg/mamba-killer-1b \
  --model_name mamba-killer-1b

# Upload all checkpoints
python scripts/upload_to_hub.py \
  --checkpoint_dir ./checkpoints/base \
  --upload_all \
  --repo_id neko-jpg/mamba-killer-models
```

## Troubleshooting

### Out of Memory (OOM)

**Symptoms:** CUDA OOM error during training

**Solutions:**
1. Reduce batch size:
   ```yaml
   training:
     batch_size: 4  # Reduce from 8
   ```

2. Enable gradient checkpointing:
   ```yaml
   training:
     gradient_checkpointing: true
   ```

3. Reduce sequence length:
   ```yaml
   model:
     n_seq: 1024  # Reduce from 2048
   ```

4. Enable CPU offloading:
   ```yaml
   model:
     use_cpu_offload: true
   ```

### Slow Training

**Symptoms:** Training is slower than expected

**Solutions:**
1. Check GPU utilization:
   ```bash
   nvidia-smi -l 1
   ```
   Should show >90% GPU utilization

2. Reduce logging frequency:
   ```yaml
   training:
     log_interval: 500  # Increase from 100
   ```

3. Disable expensive monitoring:
   ```yaml
   monitoring:
     log_schatten_norms: false
     log_grad_norm: false
   ```

### NaN/Inf During Training

**Symptoms:** Loss becomes NaN or Inf

**Solutions:**
1. Reduce learning rate:
   ```yaml
   training:
     learning_rate: 5.0e-4  # Reduce from 1.0e-3
   ```

2. Enable gradient clipping:
   ```yaml
   training:
     max_grad_norm: 0.5  # Reduce from 1.0
   ```

3. Increase epsilon (more regularization):
   ```yaml
   model:
     epsilon: 1.5  # Increase from 1.0
   ```

### Dataset Download Fails

**Symptoms:** Error downloading datasets from Hugging Face

**Solutions:**
1. Check internet connection

2. Use manual download (see [Manual Preparation](#manual-preparation))

3. Use smaller sample sizes:
   ```bash
   python scripts/prepare_datasets.py \
     --datasets c4 \
     --c4_samples 10000  # Reduce from 100000
   ```

### Checkpoint Corruption

**Symptoms:** Error loading checkpoint

**Solutions:**
1. Use previous checkpoint:
   ```bash
   # List available checkpoints
   ls -lh ./checkpoints/base/
   
   # Load specific checkpoint
   python train.py --resume ./checkpoints/base/checkpoint_epoch5_step10000.pt
   ```

2. Verify checkpoint integrity:
   ```python
   from src.utils.checkpoint_manager import CheckpointManager
   manager = CheckpointManager()
   is_valid = manager.verify_checkpoint('./checkpoints/base/latest.pt')
   ```

## Expected Results

### WikiText-2 (Base Model)

- **Perplexity:** ~30-35
- **Training Time:** ~12 hours on T4 GPU
- **Memory Usage:** ~12GB GPU RAM

### Long-Context (128k tokens)

- **Perplexity:** <50 (vs Mamba divergence)
- **Training Time:** ~48 hours on T4 GPU
- **Memory Usage:** ~14GB GPU RAM

### Quantization (INT8)

- **Perplexity Degradation:** <5%
- **Model Size:** 4× smaller
- **Inference Speed:** 2× faster

### Efficiency (ACT + Sparsity)

- **FLOPs Reduction:** 40-60%
- **Perplexity:** Within 5% of baseline
- **Inference Speed:** 1.5-2× faster

## Variance and Statistical Significance

All results are averaged over 5 random seeds. Expected variance:

- **Perplexity:** ±2%
- **Training Time:** ±10%
- **Memory Usage:** ±5%

Statistical significance is tested using paired t-tests with Bonferroni correction (p < 0.01).

## Citation

If you use this code or reproduce our results, please cite:

```bibtex
@article{mamba-killer-2024,
  title={Mamba-Killer: Ultra-Scale ResNet-BK with Birman-Schwinger Theory},
  author={Your Name},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```

## Support

For questions and issues:

- **GitHub Issues:** https://github.com/neko-jpg/Project-ResNet-BK-An-O-N-Language-Model-Architecture/issues
- **Discussions:** https://github.com/neko-jpg/Project-ResNet-BK-An-O-N-Language-Model-Architecture/discussions
- **Email:** arat252539@gmail.com

## License

MIT License - see [LICENSE](LICENSE) file for details.
