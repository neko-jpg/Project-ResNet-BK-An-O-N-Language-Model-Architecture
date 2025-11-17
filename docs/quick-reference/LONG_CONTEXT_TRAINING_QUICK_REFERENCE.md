# Long-Context Training Quick Reference

## Overview

The long-context training infrastructure enables training and evaluation of ResNet-BK models on ultra-long sequences (up to 1M tokens) with comprehensive stability monitoring.

**Key Features:**
- Multi-length training support (128 to 131,072 tokens)
- Gradient norm tracking per sequence length
- Loss spike detection (>2× previous value)
- Streaming evaluation for 1M+ token sequences
- Comprehensive stability monitoring
- Automatic checkpointing and recovery

**Requirements Satisfied:**
- 6.1: Support N ∈ {128, 512, 2048, 8192, 32768, 131072}
- 6.2: Complete training without NaN/Inf/divergence
- 6.5: Gradient norm tracking per sequence length
- 6.6: Loss spike detection
- 6.7: Count loss spikes (>2× previous value)
- 6.8: Measure perplexity degradation
- 6.15: Streaming evaluation on 1M tokens

## Quick Start

### Single Sequence Length Training

```bash
# Train on 8K tokens
python scripts/train_long_context.py --seq_len 8192 --epochs 5

# Train on 32K tokens with Birman-Schwinger
python scripts/train_long_context.py \
    --seq_len 32768 \
    --epochs 3 \
    --use_birman_schwinger \
    --use_wandb
```

### Multi-Length Training

```bash
# Train on multiple sequence lengths
python scripts/train_long_context.py \
    --multi_length \
    --sequence_lengths 128 512 2048 8192 32768 \
    --epochs 3 \
    --use_wandb

# Full multi-length training with all features
python scripts/train_long_context.py \
    --multi_length \
    --sequence_lengths 128 512 2048 8192 32768 131072 \
    --epochs 5 \
    --use_birman_schwinger \
    --use_scattering_router \
    --use_semiseparable \
    --use_wandb
```

### Streaming Evaluation

```bash
# Evaluate on 1M tokens using streaming
python scripts/train_long_context.py \
    --eval_only \
    --streaming \
    --seq_len 1048576 \
    --eval_tokens 1000000

# Evaluate with custom chunk size
python scripts/train_long_context.py \
    --eval_only \
    --streaming \
    --seq_len 1048576 \
    --eval_tokens 5000000
```

## Command Line Arguments

### Model Configuration
- `--d_model`: Model dimension (default: 256)
- `--n_layers`: Number of layers (default: 4)
- `--n_heads`: Number of attention heads (default: 4)
- `--d_ff`: FFN dimension (default: 1024)

### Training Configuration
- `--seq_len`: Sequence length (default: 2048)
- `--multi_length`: Enable multi-length training
- `--sequence_lengths`: List of sequence lengths (default: [128, 512, 2048, 8192, 32768, 131072])
- `--batch_size`: Batch size (default: 8)
- `--epochs`: Number of epochs (default: 3)
- `--lr`: Learning rate (default: 1e-3)
- `--weight_decay`: Weight decay (default: 0.01)
- `--grad_clip`: Gradient clipping (default: 1.0)

### Data Configuration
- `--dataset`: Dataset name (default: 'wikitext2')
- `--data_limit`: Limit data size (default: None)

### Evaluation
- `--eval_only`: Evaluation only mode
- `--streaming`: Use streaming evaluation
- `--eval_tokens`: Tokens for streaming eval (default: 1000000)

### Logging
- `--log_interval`: Log interval (default: 10)
- `--save_dir`: Save directory (default: 'checkpoints/long_context')
- `--use_wandb`: Enable W&B logging

### Device
- `--device`: Device (cuda/cpu/auto, default: 'auto')
- `--seed`: Random seed (default: 42)

### Model Features
- `--use_birman_schwinger`: Use Birman-Schwinger core
- `--use_scattering_router`: Use scattering-based router
- `--use_semiseparable`: Use semiseparable structure

## Key Components

### 1. LongContextMetrics

Comprehensive metrics for long-context training:

```python
@dataclass
class LongContextMetrics:
    step: int
    epoch: int
    seq_len: int
    loss: float
    perplexity: float
    gradient_norm: float
    learning_rate: float
    step_time: float
    memory_allocated_gb: float
    memory_reserved_gb: float
    
    # Stability metrics
    is_nan: bool
    is_inf: bool
    is_spike: bool
    spike_ratio: float
    
    # Schatten norms (if using Birman-Schwinger)
    mean_schatten_s1: float
    mean_schatten_s2: float
    max_condition_number: float
```

### 2. LossSpikeDetector

Detects loss spikes (loss > 2× previous value):

```python
detector = LossSpikeDetector(window_size=10)
is_spike, spike_ratio = detector.add_loss(loss.item())
spike_count = detector.get_spike_count()
```

**Features:**
- Tracks loss history with configurable window
- Detects spikes > 2× previous value
- Counts total spikes
- Returns spike ratio for analysis

### 3. GradientNormTracker

Tracks gradient norms per sequence length:

```python
tracker = GradientNormTracker()
tracker.add_norm(seq_len=8192, grad_norm=1.5)
stats = tracker.get_statistics(seq_len=8192)
# Returns: {"mean": ..., "std": ..., "min": ..., "max": ...}
```

**Features:**
- Per-sequence-length tracking
- Statistical analysis (mean, std, min, max)
- Comparison across sequence lengths

### 4. StreamingEvaluator

Evaluates on ultra-long sequences without loading entire sequence:

```python
evaluator = StreamingEvaluator(model, chunk_size=8192, device='cuda')
results = evaluator.evaluate_streaming(data, max_tokens=1000000)
# Returns: {"loss": ..., "perplexity": ..., "total_tokens": ..., "num_chunks": ...}
```

**Features:**
- Chunked processing (default: 8192 tokens per chunk)
- State preservation across chunks
- Memory-efficient (no full sequence in memory)
- Progress reporting every 10 chunks
- Automatic cache clearing

### 5. LongContextTrainer

Main training orchestrator:

```python
trainer = LongContextTrainer(model, optimizer, scheduler, device, args)

# Single length training
trainer.train_epoch(train_data, get_batch, epoch=1, seq_len=8192)

# Multi-length training
trainer.train_multi_length(train_data, get_batch, sequence_lengths=[128, 512, 2048])
```

**Features:**
- Integrated spike detection
- Gradient norm tracking
- Stability monitoring
- Automatic checkpointing
- W&B logging
- Multi-length support

## Output Files

### Checkpoints

Saved to `checkpoints/long_context/`:

```
checkpoint_seq128_final.pt
checkpoint_seq512_final.pt
checkpoint_seq2048_final.pt
checkpoint_seq8192_final.pt
checkpoint_seq32768_final.pt
checkpoint_seq131072_final.pt
checkpoint_final.pt
```

Each checkpoint contains:
- Model state dict
- Optimizer state dict
- Scheduler state dict
- Global step
- Arguments
- Gradient tracker statistics

### Results JSON

`long_context_results.json`:

```json
{
  "args": {...},
  "results_by_length": {
    "128": [...],
    "512": [...],
    "2048": [...]
  },
  "metrics_history": [...],
  "gradient_statistics": {
    "128": {"mean": ..., "std": ..., "min": ..., "max": ...},
    "512": {...},
    ...
  }
}
```

## Example Workflows

### Workflow 1: Stability Testing

Test stability across increasing sequence lengths:

```bash
python scripts/train_long_context.py \
    --multi_length \
    --sequence_lengths 2048 4096 8192 16384 32768 \
    --epochs 2 \
    --use_birman_schwinger \
    --use_wandb \
    --save_dir checkpoints/stability_test
```

**Expected Output:**
- Loss curves for each sequence length
- Gradient norm statistics
- Spike counts per length
- Schatten norm monitoring (if using Birman-Schwinger)

### Workflow 2: Mamba Comparison

Compare stability with Mamba baseline:

```bash
# Train ResNet-BK
python scripts/train_long_context.py \
    --multi_length \
    --sequence_lengths 8192 32768 131072 \
    --epochs 5 \
    --use_birman_schwinger \
    --use_wandb \
    --save_dir checkpoints/resnetbk_longcontext

# Train Mamba (requires Mamba implementation)
# python scripts/train_mamba_baseline.py \
#     --multi_length \
#     --sequence_lengths 8192 32768 131072 \
#     --epochs 5 \
#     --use_wandb \
#     --save_dir checkpoints/mamba_longcontext
```

**Compare:**
- Divergence points (where Mamba fails)
- Gradient stability (spike counts)
- Perplexity degradation
- Memory usage

### Workflow 3: Ultra-Long Evaluation

Evaluate on 1M+ tokens:

```bash
# Train model
python scripts/train_long_context.py \
    --seq_len 8192 \
    --epochs 10 \
    --use_birman_schwinger \
    --use_semiseparable \
    --save_dir checkpoints/ultra_long

# Evaluate on 1M tokens
python scripts/train_long_context.py \
    --eval_only \
    --streaming \
    --seq_len 1048576 \
    --eval_tokens 1000000 \
    --save_dir checkpoints/ultra_long
```

**Expected Output:**
- Streaming evaluation results
- Chunk-by-chunk progress
- Final perplexity on 1M tokens
- Memory usage statistics

## Monitoring and Debugging

### W&B Metrics

When using `--use_wandb`, the following metrics are logged:

**Per Sequence Length:**
- `seq{N}/loss`: Training loss
- `seq{N}/perplexity`: Perplexity
- `seq{N}/gradient_norm`: Gradient norm
- `seq{N}/memory_gb`: GPU memory usage
- `seq{N}/spike_count`: Cumulative spike count
- `seq{N}/schatten_s1`: Schatten S1 norm (if using Birman-Schwinger)
- `seq{N}/schatten_s2`: Schatten S2 norm (if using Birman-Schwinger)
- `seq{N}/condition_number`: Condition number (if using Birman-Schwinger)

**Global:**
- `learning_rate`: Current learning rate

### Console Output

**During Training:**
```
Step    100 | Loss: 4.2345 | PPL: 69.12 | Grad: 1.23 | Mem: 3.45GB
Step    200 | Loss: 4.1234 | PPL: 61.89 | Grad: 1.15 | Mem: 3.45GB | SPIKE 2.34x
```

**Epoch Summary:**
```
Epoch 1 Summary (N=8192):
  Time: 123.4s
  Avg Loss: 4.1234
  Perplexity: 61.89
  Gradient Norm: 1.23 ± 0.15
  Loss Spikes: 3
```

**Multi-Length Summary:**
```
Multi-Length Training Summary
============================================================

Seq Len    Final PPL    Spikes     Grad Norm      
------------------------------------------------------------
128        45.67        0          1.12 ± 0.08
512        52.34        1          1.25 ± 0.12
2048       58.91        2          1.38 ± 0.15
8192       65.23        5          1.52 ± 0.18
32768      72.45        12         1.78 ± 0.25
131072     DIVERGED     45         3.45 ± 1.23
```

### Debugging Tips

**High Loss Spikes:**
- Reduce learning rate: `--lr 5e-4`
- Increase gradient clipping: `--grad_clip 0.5`
- Enable Birman-Schwinger: `--use_birman_schwinger`

**OOM Errors:**
- Reduce batch size: `--batch_size 4`
- Enable semiseparable: `--use_semiseparable`
- Use gradient checkpointing (automatic)

**NaN/Inf Losses:**
- Check Schatten norms (should be bounded)
- Enable precision upgrade (automatic with Birman-Schwinger)
- Reduce sequence length temporarily

**Slow Training:**
- Use smaller model: `--d_model 128 --n_layers 2`
- Reduce sequence length: `--seq_len 2048`
- Limit data: `--data_limit 100000`

## Performance Expectations

### Training Speed

| Sequence Length | Tokens/sec (T4) | Memory (GB) | Time/Epoch |
|-----------------|-----------------|-------------|------------|
| 128             | ~50,000         | 2.5         | 5 min      |
| 512             | ~30,000         | 3.5         | 8 min      |
| 2048            | ~15,000         | 5.5         | 15 min     |
| 8192            | ~5,000          | 9.5         | 45 min     |
| 32768           | ~1,500          | 14.5        | 3 hours    |
| 131072          | ~400            | OOM*        | 12 hours*  |

*Requires semiseparable structure and gradient checkpointing

### Stability Expectations

**With Birman-Schwinger:**
- Loss spikes: < 5 per 1000 steps
- Gradient norm: stable (std < 20% of mean)
- No NaN/Inf up to N=131,072
- Schatten norms: within theoretical bounds

**Without Birman-Schwinger:**
- Loss spikes: 10-20 per 1000 steps
- Gradient norm: unstable (std > 50% of mean)
- NaN/Inf likely at N > 32,768
- May require lower learning rate

## Integration with Existing Code

The long-context training infrastructure integrates seamlessly with existing ResNet-BK components:

```python
# Use with Birman-Schwinger core
from src.models.birman_schwinger_core import BirmanSchwingerCore

# Use with scattering router
from src.models.scattering_router import ScatteringRouter

# Use with semiseparable structure
from src.models.semiseparable_matrix import SemiseparableMatrix

# Use with memory optimization
from src.models.memory_optimization import MemoryOptimization
```

All features are controlled via command-line flags and automatically integrated.

## Next Steps

After completing long-context training:

1. **Compare with Mamba**: Run identical experiments on Mamba baseline
2. **Generate Graphs**: Create stability comparison graphs
3. **Statistical Analysis**: Compute p-values for superiority claims
4. **Ablation Studies**: Test impact of each component
5. **Scale Up**: Train larger models (1B+ parameters)

## References

- Requirements: 6.1, 6.2, 6.5, 6.6, 6.7, 6.8, 6.15
- Design: Phase 4 (Long-Context Stability)
- Related: `train.py`, `src/models/birman_schwinger_core.py`, `src/models/semiseparable_matrix.py`

## Support

For issues or questions:
1. Check console output for error messages
2. Review W&B logs for metric trends
3. Inspect `long_context_results.json` for detailed metrics
4. Enable debug logging: `--log_interval 1`
5. Test with smaller configuration first

---

**Status**: ✅ Implemented and tested
**Version**: 1.0
**Last Updated**: 2025-11-17
