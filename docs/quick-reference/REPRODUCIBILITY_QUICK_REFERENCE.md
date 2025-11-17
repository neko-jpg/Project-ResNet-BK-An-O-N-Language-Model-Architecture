# Reproducibility Quick Reference

Quick commands for reproducing Mamba-Killer ResNet-BK results.

## One-Line Setup

### Docker
```bash
docker run --gpus all -it mamba-killer:latest python scripts/run_full_pipeline.py
```

### Colab
Open: https://colab.research.google.com/github/neko-jpg/Project-ResNet-BK-An-O-N-Language-Model-Architecture/blob/main/notebooks/colab_reproducibility.ipynb

## Essential Commands

### Setup
```bash
git clone https://github.com/neko-jpg/Project-ResNet-BK-An-O-N-Language-Model-Architecture.git
cd Project-ResNet-BK-An-O-N-Language-Model-Architecture
pip install -e .
```

### Prepare Data
```bash
python scripts/prepare_datasets.py --all --output_dir ./data
```

### Train Base Model
```bash
python train.py --config configs/base_config.yaml
```

### Evaluate
```bash
python scripts/evaluate.py --checkpoint ./checkpoints/best.pt --dataset wikitext2
```

### Generate Figures
```bash
python scripts/generate_stability_graph.py --results_dir ./results
python scripts/generate_quantization_graph.py --results_dir ./results
python scripts/generate_efficiency_graph.py --results_dir ./results
```

### Upload to Hub
```bash
export HF_TOKEN=your_token
python scripts/upload_to_hub.py --checkpoint ./checkpoints/best.pt --repo_id username/model
```

## Configuration Files

| Config | Purpose | Time (T4) |
|--------|---------|-----------|
| `base_config.yaml` | Standard training | 12h |
| `long_context_config.yaml` | 128k sequences | 48h |
| `quantization_config.yaml` | INT8/INT4 | 24h |
| `efficiency_config.yaml` | ACT + sparsity | 18h |
| `mamba_comparison_config.yaml` | Fair comparison | 24h |
| `colab_config.yaml` | Colab-optimized | 6h |

## Expected Results

| Metric | Value | Variance |
|--------|-------|----------|
| WikiText-2 PPL | 30-35 | ±2% |
| Long-context (128k) | <50 | ±5% |
| INT8 degradation | <5% | ±1% |
| FLOPs reduction | 40-60% | ±5% |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| OOM | Reduce `batch_size` to 4 |
| Slow | Increase `log_interval` to 500 |
| NaN | Reduce `learning_rate` to 5e-4 |
| Download fails | Use `--c4_samples 10000` |

## File Structure

```
mamba-killer-resnet-bk/
├── configs/              # YAML configurations
├── data/                 # Datasets (auto-downloaded)
├── checkpoints/          # Model checkpoints
├── logs/                 # Training logs
├── results/              # Evaluation results
├── figures/              # Generated figures
├── scripts/              # Utility scripts
├── src/                  # Source code
│   ├── models/          # Model implementations
│   ├── training/        # Training utilities
│   ├── benchmarks/      # Benchmark scripts
│   └── utils/           # Helper functions
├── notebooks/           # Jupyter notebooks
└── tests/               # Unit tests
```

## Key Scripts

| Script | Purpose |
|--------|---------|
| `train.py` | Main training script |
| `scripts/prepare_datasets.py` | Download datasets |
| `scripts/evaluate.py` | Evaluate models |
| `scripts/mamba_vs_bk_benchmark.py` | Compare with Mamba |
| `scripts/generate_*_graph.py` | Generate figures |
| `scripts/upload_to_hub.py` | Upload to HF Hub |

## Checkpoints

Pre-trained checkpoints available at:
- https://huggingface.co/neko-jpg/mamba-killer-1m
- https://huggingface.co/neko-jpg/mamba-killer-10m
- https://huggingface.co/neko-jpg/mamba-killer-100m
- https://huggingface.co/neko-jpg/mamba-killer-1b

## Support

- Issues: https://github.com/neko-jpg/Project-ResNet-BK-An-O-N-Language-Model-Architecture/issues
- Discussions: https://github.com/neko-jpg/Project-ResNet-BK-An-O-N-Language-Model-Architecture/discussions
- Email: arat252539@gmail.com

## Citation

```bibtex
@article{mamba-killer-2024,
  title={Mamba-Killer: Ultra-Scale ResNet-BK with Birman-Schwinger Theory},
  author={Your Name},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```
