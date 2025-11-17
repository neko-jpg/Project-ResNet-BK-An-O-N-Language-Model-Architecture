# Automated Benchmark Pipeline - Quick Reference

## Overview

The automated benchmark pipeline (`scripts/mamba_vs_bk_benchmark.py`) provides a comprehensive framework for comparing Mamba and ResNet-BK models across multiple datasets and downstream tasks.

## Quick Start

### Basic Training

```bash
# Train ResNet-BK on WikiText-2
python scripts/mamba_vs_bk_benchmark.py --model bk --seq_len 128 --bits 32

# Train Mamba baseline
python scripts/mamba_vs_bk_benchmark.py --model mamba --seq_len 128 --bits 32
```

### Multi-Dataset Evaluation

```bash
# Evaluate on 3 datasets
python scripts/mamba_vs_bk_benchmark.py \
    --model bk \
    --multi_dataset \
    --datasets wikitext-2 wikitext-103 ptb

# Evaluate on all 5 datasets
python scripts/mamba_vs_bk_benchmark.py \
    --model bk \
    --multi_dataset \
    --datasets wikitext-2 wikitext-103 ptb c4 pile
```

### Downstream Task Evaluation

```bash
# Evaluate on GLUE and SQuAD
python scripts/mamba_vs_bk_benchmark.py \
    --model bk \
    --downstream \
    --tasks glue squad

# Evaluate on all tasks
python scripts/mamba_vs_bk_benchmark.py \
    --model bk \
    --downstream \
    --tasks glue superglue squad mmlu
```

## Command-Line Arguments

### Required Arguments

- `--model {mamba,bk}`: Model to benchmark

### Dataset Arguments

- `--dataset STR`: Dataset name (default: wikitext-2)
- `--seq_len INT`: Sequence length (default: 128)
- `--bits {32,16,8,4}`: Quantization bits (default: 32)

### Training Arguments

- `--batch_size INT`: Batch size (default: 32)
- `--epochs INT`: Number of epochs (default: 10)
- `--lr FLOAT`: Learning rate (default: 1e-3)
- `--grad_clip FLOAT`: Gradient clipping (default: 1.0)
- `--weight_decay FLOAT`: Weight decay (default: 0.01)

### Model Arguments

- `--d_model INT`: Model dimension (default: 256)
- `--n_layers INT`: Number of layers (default: 8)
- `--vocab_size INT`: Vocabulary size (default: 30000)

### Multi-Dataset Arguments

- `--multi_dataset`: Enable multi-dataset evaluation
- `--datasets STR [STR ...]`: List of datasets

### Downstream Task Arguments

- `--downstream`: Enable downstream task evaluation
- `--tasks STR [STR ...]`: List of tasks

### Other Arguments

- `--seed INT`: Random seed (default: 42)
- `--device STR`: Device (cuda or cpu)
- `--output_dir STR`: Output directory (default: benchmark_results)
- `--no_checkpoint`: Disable checkpoint saving

## Supported Datasets

1. **wikitext-2**: WikiText-2 language modeling dataset
2. **wikitext-103**: WikiText-103 language modeling dataset
3. **ptb**: Penn Treebank dataset
4. **c4**: C4 (Colossal Clean Crawled Corpus) dataset
5. **pile**: The Pile dataset

## Supported Downstream Tasks

### GLUE Benchmark
- CoLA, SST-2, MRPC, QQP, MNLI, QNLI, RTE, WNLI

### SuperGLUE Benchmark
- BoolQ, CB, COPA, MultiRC, ReCoRD, RTE, WiC, WSC

### SQuAD
- Question answering with EM and F1 metrics

### MMLU
- Multiple-choice questions across subjects

## Output Files

### Single Dataset
- `{model}_{dataset}_seq{seq_len}_bits{bits}.json`
- Contains: loss, perplexity, FLOPs, memory, training time

### Multi-Dataset
- `{model}_multi_dataset.json`
- Contains: per-dataset perplexities, mean, std

### Downstream Tasks
- `{model}_downstream.json`
- Contains: per-task scores, averages

### Checkpoints
- `{model}_{dataset}_best.pt`
- Saved when best perplexity is achieved

## Examples

### Long Context Evaluation

```bash
# 8K context
python scripts/mamba_vs_bk_benchmark.py --model bk --seq_len 8192

# 32K context
python scripts/mamba_vs_bk_benchmark.py --model bk --seq_len 32768
```

### Quantization Sweep

```bash
# FP32
python scripts/mamba_vs_bk_benchmark.py --model bk --bits 32

# FP16
python scripts/mamba_vs_bk_benchmark.py --model bk --bits 16

# INT8
python scripts/mamba_vs_bk_benchmark.py --model bk --bits 8

# INT4
python scripts/mamba_vs_bk_benchmark.py --model bk --bits 4
```

### Full Comparison

```bash
# Train both models on same dataset
python scripts/mamba_vs_bk_benchmark.py --model bk --dataset wikitext-2
python scripts/mamba_vs_bk_benchmark.py --model mamba --dataset wikitext-2

# Compare results
python -c "
import json
bk = json.load(open('benchmark_results/bk_wikitext-2_seq128_bits32.json'))
mamba = json.load(open('benchmark_results/mamba_wikitext-2_seq128_bits32.json'))
print(f'ResNet-BK PPL: {bk[\"best_perplexity\"]:.2f}')
print(f'Mamba PPL: {mamba[\"best_perplexity\"]:.2f}')
"
```

## Tips

### Performance Optimization

1. **Use GPU**: Add `--device cuda` for faster training
2. **Adjust batch size**: Increase `--batch_size` if memory allows
3. **Reduce epochs**: Use `--epochs 5` for quick tests
4. **Cache datasets**: Datasets are cached after first download

### Memory Management

1. **Reduce sequence length**: Use `--seq_len 128` for testing
2. **Reduce model size**: Use `--d_model 128 --n_layers 4`
3. **Use quantization**: Add `--bits 16` to reduce memory

### Reproducibility

1. **Fix seed**: Use `--seed 42` for reproducible results
2. **Save checkpoints**: Enabled by default
3. **Log results**: All results saved in JSON format

## Troubleshooting

### Out of Memory

```bash
# Reduce batch size
--batch_size 8

# Reduce sequence length
--seq_len 128

# Use FP16
--bits 16
```

### Dataset Download Fails

```bash
# Check internet connection
# Datasets are downloaded automatically on first use
# Cached in ./data/ directory
```

### NaN/Inf During Training

```bash
# Reduce learning rate
--lr 1e-4

# Increase gradient clipping
--grad_clip 0.5
```

## Requirements

### Python Packages

```bash
pip install torch numpy scipy datasets transformers
```

### Optional Packages

```bash
pip install wandb  # For experiment tracking
pip install tensorboard  # For visualization
```

## Related Files

- **Implementation**: `scripts/mamba_vs_bk_benchmark.py`
- **Tests**: `test_benchmark_pipeline_task19.py`
- **Documentation**: `TASK_19_BENCHMARK_PIPELINE_COMPLETION.md`

## Support

For issues or questions:
1. Check the completion report: `TASK_19_BENCHMARK_PIPELINE_COMPLETION.md`
2. Run tests: `python test_benchmark_pipeline_task19.py`
3. Review examples in this guide

## Status

✅ **Fully Implemented and Tested**

All features working as specified in requirements 9.1, 9.2, 9.3, 11.15, 11.16, 11.17, 11.18.
