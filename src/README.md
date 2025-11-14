# ResNet-BK Source Code

This directory contains the modular implementation of ResNet-BK.

## Structure

```
src/
├── models/              # Model architectures
│   ├── bk_core.py      # O(N) BK-Core algorithm
│   ├── moe.py          # Sparse Mixture of Experts
│   ├── resnet_bk.py    # ResNet-BK architecture
│   └── configurable_resnet_bk.py  # Configurable model with all optimizations
├── training/            # Training loops and optimization
├── utils/               # Utilities
│   ├── config.py       # Configuration and argument parsing
│   ├── data_utils.py   # Data loading
│   ├── metrics.py      # Metrics tracking
│   ├── visualization.py # Training dashboard
│   └── wandb_logger.py # W&B integration (optional)
└── benchmarks/          # Benchmarking utilities
```

## Quick Start

```python
from src.models import LanguageModel
from src.utils import get_data_loader

# Load data
train_data, vocab, get_batch = get_data_loader(
    batch_size=20,
    n_seq=128
)

# Create model
model = LanguageModel(
    vocab_size=vocab['vocab_size'],
    d_model=64,
    n_layers=4,
    n_seq=128,
    num_experts=4,
    top_k=1,
)

# Train...
```

## Configuration Presets

Use predefined configurations for different optimization levels:

```python
from src.models.configurable_resnet_bk import (
    ConfigurableResNetBK,
    BASELINE_CONFIG,  # Step 1: O(N) architecture + analytic gradient
    STEP2_CONFIG,     # + Koopman learning + physics-informed
    STEP4_CONFIG,     # + Compression (quantization, pruning, distillation)
    STEP5_CONFIG,     # + Hardware optimization (mixed precision, custom kernels)
    STEP6_CONFIG,     # + Algorithmic innovations (ACT, multi-scale, sparsity)
    FULL_CONFIG,      # All optimizations enabled
)

model = ConfigurableResNetBK(BASELINE_CONFIG)
```

## Command-Line Interface

```bash
# Train with baseline configuration
python train.py --config-preset baseline

# Train with all optimizations
python train.py --config-preset full

# Custom configuration
python train.py --config-preset custom \
    --d-model 128 \
    --n-layers 8 \
    --use-mixed-precision \
    --use-koopman
```

## Testing

Run tests:
```bash
pytest tests/ -v
```

Run specific test:
```bash
pytest tests/test_bk_core.py -v
```

## Notebooks

See `notebooks/` directory for Google Colab notebooks:
- `01_quick_start.ipynb` - Train in <5 minutes
- `02_full_training.ipynb` - Reproduce paper results
- `03_benchmarking.ipynb` - Compare configurations
- `04_interpretability.ipynb` - Visualize internals
