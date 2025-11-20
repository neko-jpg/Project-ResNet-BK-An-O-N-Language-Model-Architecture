# Phase 2 Training Quick Reference

## Quick Start

```bash
# Basic training (3 epochs, small model)
python scripts/train_phase2.py --preset small --num-epochs 3

# Full training with WandB
python scripts/train_phase2.py --preset base --use-wandb --num-epochs 50
```

## Command-Line Options

### Model Configuration
```bash
--preset {small,base,large}  # Preset configuration
--vocab-size INT             # Vocabulary size (default: 10000)
--d-model INT                # Model dimension (default: 512)
--n-layers INT               # Number of layers (default: 6)
--n-seq INT                  # Sequence length (default: 512)
--num-heads INT              # Number of heads (default: 8)
--head-dim INT               # Head dimension (default: 64)
```

### Training Configuration
```bash
--batch-size INT             # Batch size (default: 4)
--num-epochs INT             # Number of epochs (default: 10)
--learning-rate FLOAT        # Learning rate (default: 1e-4)
--weight-decay FLOAT         # Weight decay (default: 0.01)
--gradient-clip-norm FLOAT   # Gradient clipping (default: 1.0)
```

### Phase 2 Specific
```bash
--use-triton                 # Use Triton kernels (default: True)
--base-decay FLOAT           # Base decay rate (default: 0.01)
--hebbian-eta FLOAT          # Hebbian learning rate (default: 0.1)
--snr-threshold FLOAT        # SNR threshold (default: 2.0)
--resonance-threshold FLOAT  # Resonance threshold (default: 0.1)
```

### WandB Configuration
```bash
--use-wandb                  # Enable WandB logging
--wandb-project STR          # WandB project name
--wandb-name STR             # WandB experiment name
```

## Diagnostic Metrics

### Γ (Forgetting Rate)
- `mean_gamma`: Average forgetting rate
- `std_gamma`: Standard deviation
- `min_gamma`: Minimum value
- `max_gamma`: Maximum value

### SNR (Signal-to-Noise Ratio)
- `mean_snr`: Average SNR
- `low_snr_ratio`: Ratio of low SNR components

### Memory Resonance
- `num_resonant_modes`: Number of resonant modes
- `total_resonance_energy`: Total resonance energy

### Lyapunov Stability
- `lyapunov_stable_ratio`: Ratio of stable layers
- `mean_fast_weight_energy`: Average Fast Weights energy

## Output Files

```
checkpoints/phase2/
├── best_model.pt              # Best model (lowest val loss)
├── checkpoint_epoch{N}.pt     # Periodic checkpoints
└── training_history.json      # Complete training history
```

## Training History JSON

```json
{
  "train_losses": [...],
  "val_losses": [...],
  "learning_rates": [...],
  "gamma_history": [...],
  "snr_history": [...],
  "resonance_history": [...],
  "stability_history": [...],
  "best_val_loss": 2.345,
  "total_epochs": 10
}
```

## WandB Metrics

### Batch-level (Real-time)
- `batch/gamma_mean`, `batch/gamma_std`
- `batch/snr_mean`, `batch/snr_low_ratio`
- `batch/resonant_modes`, `batch/resonance_energy`
- `batch/stability_ratio`, `batch/fast_weight_energy`

### Epoch-level
- `train/loss`, `val/loss`
- `train/perplexity`, `val/perplexity`
- `train/gamma_mean`, `train/snr_mean`
- `train/resonant_modes`, `train/stability_ratio`

## Common Use Cases

### 1. Quick Test (2-3 minutes)
```bash
python scripts/train_phase2.py \
    --preset small \
    --num-epochs 3 \
    --batch-size 2 \
    --num-train-batches 20 \
    --num-val-batches 5
```

### 2. Full Training with WandB
```bash
python scripts/train_phase2.py \
    --preset base \
    --num-epochs 50 \
    --batch-size 4 \
    --use-wandb \
    --wandb-project phase2-breath-of-life \
    --wandb-name experiment-001
```

### 3. Custom Configuration
```bash
python scripts/train_phase2.py \
    --vocab-size 50000 \
    --d-model 768 \
    --n-layers 12 \
    --n-seq 1024 \
    --num-heads 12 \
    --head-dim 64 \
    --batch-size 8 \
    --num-epochs 100 \
    --learning-rate 5e-5 \
    --use-triton \
    --base-decay 0.005 \
    --hebbian-eta 0.15
```

### 4. CPU Training (for testing)
```bash
python scripts/train_phase2.py \
    --preset small \
    --device cpu \
    --num-epochs 2 \
    --batch-size 1
```

## Loading Checkpoints

```python
import torch

# Load best model
checkpoint = torch.load('checkpoints/phase2/best_model.pt')

# Extract information
epoch = checkpoint['epoch']
best_val_loss = checkpoint['best_val_loss']
model_state = checkpoint['model_state_dict']
optimizer_state = checkpoint['optimizer_state_dict']

# Load training history
gamma_history = checkpoint['gamma_history']
snr_history = checkpoint['snr_history']
resonance_history = checkpoint['resonance_history']
stability_history = checkpoint['stability_history']
```

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size
--batch-size 2

# Reduce sequence length
--n-seq 256

# Use smaller model
--preset small
```

### Slow Training
```bash
# Enable Triton kernels
--use-triton

# Reduce logging frequency (modify code)
# Change: if batch_idx % 10 == 0:
# To:     if batch_idx % 50 == 0:
```

### WandB Not Working
```bash
# Install WandB
pip install wandb

# Login
wandb login

# Check if enabled
--use-wandb
```

## Expected Behavior

### Loss
- Initial: ~8.0-9.0
- After 10 epochs: ~5.0-6.0
- After 50 epochs: ~3.0-4.0

### Γ (Forgetting Rate)
- Initial: ~0.01-0.02
- Should vary slightly during training
- Should not exceed 0.1

### SNR
- Initial: ~1.5-2.0
- Should improve to ~2.5-3.0
- Low SNR ratio should decrease

### Stability
- Should stay above 95%
- If below 90%, increase `--base-decay`

## Performance Targets

- **VRAM**: < 8GB (batch=4, seq=1024)
- **Speed**: ~100 tokens/sec (A100)
- **PPL Degradation**: < +10% vs Phase 1

## Next Steps

1. Run quick test
2. Check training_history.json
3. Visualize with WandB
4. Adjust hyperparameters
5. Run full training

## References

- Full Documentation: `results/benchmarks/PHASE2_TRAINING_SCRIPT_REPORT.md`
- Demo Script: `examples/phase2_training_demo.py`
- Japanese Summary: `results/benchmarks/PHASE2_TRAINING_IMPLEMENTATION_SUMMARY_JP.md`
