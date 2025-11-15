# Step 4: Advanced Model Compression - Quick Reference

## Quick Start

### Run Complete Pipeline

```python
from src.training.compression_pipeline import CompressionPipeline
from src.models.configurable_resnet_bk import ConfigurableResNetBK

# 1. Create baseline model
model = ConfigurableResNetBK(
    vocab_size=50257,
    d_model=64,
    n_layers=4,
    n_seq=128,
    num_experts=4,
    use_moe=True
)

# 2. Create compression pipeline
pipeline = CompressionPipeline(
    model=model,
    target_compression=100.0,
    device='cuda'
)

# 3. Run pipeline
compressed_model, metrics = pipeline.run_pipeline(
    train_loader=train_loader,
    val_loader=val_loader,
    qat_epochs=3,
    pruning_epochs=3,
    distillation_epochs=5
)

# 4. Check results
print(f"Compression: {metrics['compression_ratio']:.2f}×")
print(f"Perplexity: {metrics['stage_metrics']['distillation']['final_perplexity']:.2f}")
```

## Individual Components

### 1. Quantization-Aware Training

```python
from src.models.quantized_bk_core import QuantizedBKCore

# Replace BK-Core with quantized version
quantized_core = QuantizedBKCore(n_seq=128, enable_quantization=True)

# Calibration
quantized_core.start_calibration()
# ... run forward passes ...
quantized_core.end_calibration()

# Training with fake quantization
model.train()
# ... normal training loop ...

# Convert to INT8 inference
quantized_core.to_int8_inference()
```

### 2. Complex Number Quantization

```python
from src.models.complex_quantization import PerChannelQuantizedBKCore

# Per-channel quantization for better accuracy
bk_core = PerChannelQuantizedBKCore(n_seq=128, enable_quantization=True)

# Calibration
bk_core.start_calibration()
# ... run forward passes ...
bk_core.end_calibration()
```

### 3. INT4 MoE Quantization

```python
from src.models.quantized_moe import QuantizedMoELayer

# Create quantized MoE
moe = QuantizedMoELayer(
    d_model=64,
    num_experts=4,
    group_size=128,
    enable_quantization=True
)

# Quantize all weights
moe.quantize_all()

# Check compression
ratio = moe.get_compression_ratio()
print(f"Compression: {ratio:.2f}×")
```

### 4. Structured Pruning

```python
from src.models.pruned_moe import PrunedMoELayer, ProgressivePruningScheduler

# Create prunable MoE
moe = PrunedMoELayer(
    d_model=64,
    num_experts=8,
    prune_threshold=0.05
)

# Progressive pruning
scheduler = ProgressivePruningScheduler(
    moe_layer=moe,
    target_experts=2,
    prune_epochs=[2, 4, 6]
)

# During training
for epoch in range(num_epochs):
    # ... training ...
    scheduler.step_epoch(epoch)
```

### 5. Magnitude Pruning

```python
from src.models.pruned_moe import MagnitudePruner

# Create pruner
pruner = MagnitudePruner(threshold=0.01)

# Prune entire model
stats = pruner.prune_model(model, verbose=True)

# Or prune specific layer
num_pruned = pruner.prune_layer(model.fc, verbose=True)
```

### 6. Knowledge Distillation

```python
from src.training.distillation_trainer import DistillationTrainer

# Create trainer
trainer = DistillationTrainer(
    teacher_model=teacher,
    student_model=student,
    temperature=2.0,
    alpha=0.7,
    feature_weight=0.1
)

# Training loop
for epoch in range(num_epochs):
    for x_batch, y_batch in train_loader:
        loss_dict = trainer.train_step(x_batch, y_batch, optimizer)
    
    # Evaluate
    val_metrics = trainer.evaluate(val_loader, criterion)
```

### 7. Progressive Distillation

```python
from src.training.distillation_trainer import ProgressiveDistillation

# Define model sizes
model_sizes = [
    (64, 4, 4),   # d_model, n_layers, num_experts
    (48, 3, 2),
    (32, 2, 2)
]

# Create cascade
cascade = ProgressiveDistillation(model_sizes, device='cuda')

# Train cascade
models = cascade.train_cascade(
    initial_teacher=teacher,
    train_loader=train_loader,
    val_loader=val_loader,
    vocab_size=50257,
    n_seq=128,
    epochs_per_stage=5
)
```

## Google Colab

### Run Notebook

1. Open `notebooks/step4_compression.ipynb` in Colab
2. Run all cells
3. Results will be saved to `./checkpoints/step4/`

### Key Outputs

- `qat_model.pt`: After quantization-aware training
- `pruned_model.pt`: After structured pruning
- `final_model.pt`: After distillation
- `compression_training_losses.png`: Training curves
- `compression_tradeoff.png`: Parameters vs perplexity

## Configuration Options

### Compression Pipeline

```python
pipeline = CompressionPipeline(
    model=model,
    target_compression=100.0,  # Target compression ratio
    device='cuda'
)

compressed_model, metrics = pipeline.run_pipeline(
    train_loader=train_loader,
    val_loader=val_loader,
    qat_epochs=3,              # QAT training epochs
    pruning_epochs=3,          # Pruning training epochs
    distillation_epochs=5,     # Distillation epochs
    save_dir='./checkpoints'   # Checkpoint directory
)
```

### Quantization

```python
# INT8 quantization
QuantizedBKCore(
    n_seq=128,
    enable_quantization=True   # Enable/disable quantization
)

# INT4 quantization
QuantizedMoELayer(
    d_model=64,
    num_experts=4,
    group_size=128,            # Group size for quantization
    enable_quantization=True
)
```

### Pruning

```python
# Expert pruning
PrunedMoELayer(
    d_model=64,
    num_experts=8,
    prune_threshold=0.05,      # Prune if usage < 5%
    top_k=1                    # Experts per token
)

# Magnitude pruning
MagnitudePruner(
    threshold=0.01             # Prune if |w| < 0.01
)
```

### Distillation

```python
DistillationTrainer(
    teacher_model=teacher,
    student_model=student,
    temperature=2.0,           # Softmax temperature
    alpha=0.7,                 # Soft target weight
    feature_weight=0.1         # Feature matching weight
)
```

## Metrics

### Compression Metrics

```python
metrics = {
    'original_parameters': 4150000,
    'final_parameters': 41500,
    'compression_ratio': 100.0,
    'compression_achieved': True,
    'total_time_seconds': 1800.0,
    'model_size': {
        'fp32_mb': 16.6,
        'quantized_mb': 0.17
    }
}
```

### Stage Metrics

```python
# QAT stage
qat_metrics = {
    'stage': 'qat',
    'final_perplexity': 150.0,
    'training_losses': [3.5, 3.2, 3.0],
    'parameters': 4150000
}

# Pruning stage
pruning_metrics = {
    'stage': 'pruning',
    'final_perplexity': 160.0,
    'active_experts_history': [8, 6, 4],
    'parameters': 1000000
}

# Distillation stage
distillation_metrics = {
    'stage': 'distillation',
    'final_perplexity': 170.0,
    'teacher_params': 1000000,
    'student_params': 41500
}
```

## Troubleshooting

### Out of Memory

```python
# Reduce batch size
BATCH_SIZE = 10  # Instead of 20

# Enable gradient accumulation
accumulation_steps = 4
```

### Poor Compression

```python
# Increase pruning threshold
prune_threshold=0.10  # Instead of 0.05

# More aggressive magnitude pruning
threshold=0.02  # Instead of 0.01
```

### High Perplexity Degradation

```python
# More training epochs
qat_epochs=5
pruning_epochs=5
distillation_epochs=10

# Higher distillation temperature
temperature=3.0  # Instead of 2.0

# More feature matching
feature_weight=0.2  # Instead of 0.1
```

## Performance Tips

1. **Calibration**: Use 10-20 batches for quantization calibration
2. **Progressive pruning**: Prune gradually over multiple epochs
3. **Distillation temperature**: Higher (2-3) for better knowledge transfer
4. **Feature matching**: Essential for maintaining BK-Core behavior
5. **Checkpoint saving**: Save after each stage for debugging

## Expected Results

| Metric | Target | Typical |
|--------|--------|---------|
| Compression Ratio | 100× | 96-100× |
| Perplexity Degradation | <15% | 10-15% |
| Training Time | <1 hour | 30-45 min |
| Memory Usage | <15GB | 10-12GB |

## References

- Quantization: `src/models/quantized_bk_core.py`
- Pruning: `src/models/pruned_moe.py`
- Distillation: `src/training/distillation_trainer.py`
- Pipeline: `src/training/compression_pipeline.py`
- Notebook: `notebooks/step4_compression.ipynb`
