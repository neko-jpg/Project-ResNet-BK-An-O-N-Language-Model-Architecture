# ResNet-BK Project Structure

This document describes the modular project structure created for the 1,000,000,000× cost reduction implementation.

## Directory Structure

```
Project-ResNet-BK/
├── src/                          # Source code
│   ├── models/                   # Model architectures
│   │   ├── __init__.py
│   │   ├── bk_core.py           # O(N) BK-Core algorithm
│   │   ├── moe.py               # Sparse Mixture of Experts
│   │   ├── resnet_bk.py         # ResNet-BK architecture
│   │   └── configurable_resnet_bk.py  # Configurable model with all optimizations
│   ├── training/                 # Training loops (to be implemented)
│   ├── utils/                    # Utilities
│   │   ├── __init__.py
│   │   ├── config.py            # Configuration and CLI argument parsing
│   │   ├── data_utils.py        # Data loading utilities
│   │   ├── metrics.py           # Training metrics and logging
│   │   ├── visualization.py     # Training dashboard
│   │   └── wandb_logger.py      # Weights & Biases integration (optional)
│   ├── benchmarks/               # Benchmarking utilities (to be implemented)
│   └── README.md                 # Source code documentation
│
├── tests/                        # Test suite
│   ├── __init__.py
│   ├── test_bk_core.py          # BK-Core unit tests
│   ├── test_gradients.py        # Gradient correctness tests
│   └── test_integration.py      # Integration tests
│
├── notebooks/                    # Google Colab notebooks
│   ├── 01_quick_start.ipynb     # Quick start (< 5 min)
│   ├── 02_full_training.ipynb   # Full training (reproduce paper)
│   ├── 03_benchmarking.ipynb    # Configuration comparison
│   └── 04_interpretability.ipynb # Visualization and analysis
│
├── .github/                      # GitHub Actions CI/CD
│   └── workflows/
│       └── tests.yml            # Automated testing
│
├── 1_BK_Language_Model_PoC/     # Original proof-of-concept code
│   └── BK-MoE_Ultra_v2_Stable.py
│
├── 2_Scaling_Benchmarks/        # Scaling experiments
│
├── .kiro/                        # Kiro spec files
│   └── specs/
│       └── million-x-cost-reduction-plan/
│           ├── requirements.md
│           ├── design.md
│           └── tasks.md
│
├── train.py                      # Main training script
├── requirements.txt              # Python dependencies
├── pytest.ini                    # Pytest configuration
├── README.md                     # Project README
├── PROJECT_STRUCTURE.md          # This file
└── LICENSE                       # MIT License
```

## Key Components

### 1. Models (`src/models/`)

**bk_core.py**
- `get_tridiagonal_inverse_diagonal()`: O(N) theta/phi recursion
- `BKCoreFunction`: Autograd function with hybrid analytic gradient
- Numerical stability: complex128 precision, NaN/Inf handling, magnitude clipping

**moe.py**
- `SparseMoELayer`: Top-k expert routing with Gumbel-Softmax
- Supports both sparse (top-1) and dense (softmax) modes

**resnet_bk.py**
- `MoEResNetBKLayer`: Combines MoE FFN with BK-Core
- `ResNetBKBlock`: Pre-norm residual block
- `LanguageModel`: Full language model with embeddings and LM head

**configurable_resnet_bk.py**
- `ResNetBKConfig`: Dataclass with all optimization flags
- Configuration presets: BASELINE_CONFIG, STEP2_CONFIG, ..., FULL_CONFIG
- `ConfigurableResNetBK`: Wrapper supporting all optimizations

### 2. Utilities (`src/utils/`)

**config.py**
- `parse_args()`: Command-line argument parsing
- `get_config_from_args()`: Convert args to ResNetBKConfig
- Supports all optimization flags from Steps 2-7

**data_utils.py**
- `get_data_loader()`: WikiText-2 data loading
- Simple word tokenization with vocabulary building
- Batching and sequence preparation

**metrics.py**
- `TrainingMetrics`: Comprehensive metrics dataclass
- `MetricsLogger`: CSV and JSON export with real-time logging
- Tracks loss, perplexity, timing, memory, gradients, BK-Core metrics

**visualization.py**
- `TrainingDashboard`: Real-time matplotlib dashboard
- `plot_training_curves()`: Post-training visualization
- Multi-panel display: loss, perplexity, LR, gradients, timing, memory

**wandb_logger.py**
- Optional Weights & Biases integration
- Graceful fallback if wandb not installed

### 3. Tests (`tests/`)

**test_bk_core.py**
- Theta/phi recursion correctness
- Comparison with direct matrix inversion
- Batched computation verification
- Numerical stability with extreme values

**test_gradients.py**
- Analytic vs finite difference comparison
- GRAD_BLEND parameter effect
- Numerical stability
- Gradient clipping verification

**test_integration.py**
- Full model forward/backward passes
- Training step simulation
- Model save/load
- ConfigurableResNetBK testing

### 4. Notebooks (`notebooks/`)

**01_quick_start.ipynb**
- Train small model in < 5 minutes
- Minimal configuration
- Quick validation

**02_full_training.ipynb**
- Reproduce paper results
- Full WikiText-2 training
- Comprehensive logging
- Checkpoint saving

**03_benchmarking.ipynb**
- Compare configuration presets
- Measure performance
- Visualize results

**04_interpretability.ipynb**
- Visualize G_ii features
- Analyze learned potential
- Expert routing patterns
- Attention-like patterns

## Configuration Presets

### BASELINE_CONFIG
- O(N) architecture
- Hybrid analytic gradient (GRAD_BLEND=0.5)
- Sparse MoE (top-1)
- All other optimizations disabled

### STEP2_CONFIG
- Baseline +
- Koopman operator learning
- Physics-informed learning

### STEP4_CONFIG
- Step 2 +
- Quantization (INT8)
- Structured pruning
- Knowledge distillation

### STEP5_CONFIG
- Step 4 +
- Mixed precision training
- Custom CUDA kernels
- Gradient checkpointing

### STEP6_CONFIG
- Step 5 +
- Adaptive computation time (ACT)
- Multi-scale processing
- Learned sparsity

### FULL_CONFIG
- All optimizations enabled
- Target: 1,000,000,000× cost reduction

## Usage

### Training

```bash
# Baseline configuration
python train.py --config-preset baseline

# Full optimization
python train.py --config-preset full --epochs 5

# Custom configuration
python train.py --config-preset custom \
    --d-model 128 \
    --n-layers 8 \
    --use-mixed-precision \
    --use-koopman
```

### Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_bk_core.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Google Colab

1. Open notebook in Colab
2. Run installation cell
3. Clone repository
4. Follow notebook instructions

## Next Steps

The following components are planned for future implementation:

1. **Step 2 Phase 2**: Koopman operator learning
2. **Step 2 Phase 3**: Physics-informed learning
3. **Step 4**: Compression pipeline (quantization, pruning, distillation)
4. **Step 5**: Custom CUDA kernels and hardware optimization
5. **Step 6**: Algorithmic innovations (ACT, multi-scale, sparsity)
6. **Step 7**: System integration (curriculum learning, active learning)

Each step will be implemented incrementally, building on the modular structure.

## Development Guidelines

1. **Modularity**: Each component should be independently testable
2. **Configuration**: All features controlled via ResNetBKConfig
3. **Testing**: Write tests before implementing new features
4. **Documentation**: Update notebooks and README for new features
5. **Numerical Stability**: Always include NaN/Inf checks and clipping
6. **Logging**: Track all relevant metrics for analysis

## CI/CD

GitHub Actions automatically:
- Runs tests on push/PR
- Tests Python 3.9, 3.10, 3.11
- Generates coverage reports
- Uploads to Codecov

## License

MIT License - see LICENSE file
