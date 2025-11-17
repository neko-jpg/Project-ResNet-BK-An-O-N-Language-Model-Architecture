---
name: Performance Issue
about: Report performance problems or unexpected slowdowns
title: '[PERFORMANCE] '
labels: performance
assignees: ''
---

## Performance Issue Description

A clear description of the performance problem.

## Benchmark Results

**Current Performance:**
- Metric: [e.g., tokens/second, PPL, memory usage]
- Value: [e.g., 500 tokens/s]
- Expected: [e.g., 1000 tokens/s]

**Benchmark Code:**
```python
# Code used to measure performance
import time
from src.models.resnet_bk import ResNetBK

model = ResNetBK(...)

start = time.time()
# ... benchmark code ...
end = time.time()

print(f"Time: {end - start:.2f}s")
```

## Environment

**System Information:**
- OS: [e.g., Ubuntu 22.04]
- Python version: [e.g., 3.10.8]
- PyTorch version: [e.g., 2.1.0]
- CUDA version: [e.g., 12.1]
- ResNet-BK version: [e.g., 0.9.0]

**Hardware:**
- GPU: [e.g., NVIDIA RTX 4090]
- CPU: [e.g., Intel i9-13900K]
- RAM: [e.g., 32GB]
- VRAM: [e.g., 24GB]

**Configuration:**
```yaml
# Paste relevant config
d_model: 256
n_layers: 8
n_seq: 2048
batch_size: 8
```

## Profiling Results

If you've profiled the code, include results:

**PyTorch Profiler:**
```
# Top 10 operations by time
Operation                    | Time (ms) | % Total
-----------------------------|-----------|--------
aten::matmul                 | 150.2     | 45%
aten::add                    | 50.1      | 15%
...
```

**Memory Profile:**
```
# Memory usage breakdown
Component        | Memory (GB) | % Total
-----------------|-------------|--------
Model weights    | 2.5         | 30%
Activations      | 4.0         | 48%
...
```

## Comparison

**Comparison with Other Models:**
| Model | Metric | Value | Ratio |
|-------|--------|-------|-------|
| ResNet-BK | tokens/s | 500 | 1.0× |
| Mamba | tokens/s | 800 | 1.6× |
| Transformer | tokens/s | 300 | 0.6× |

## Reproduction

Steps to reproduce the performance issue:

1. Install ResNet-BK version: `X.Y.Z`
2. Run benchmark: `python scripts/benchmark.py`
3. Observe slow performance

## Expected Performance

Based on documentation/paper, expected performance should be:
- [e.g., 1000 tokens/s on RTX 4090]
- [e.g., O(N) complexity, not O(N²)]

## Additional Context

- [ ] This is a regression (performance was better in version: `X.Y.Z`)
- [ ] This affects production use
- [ ] This is blocking my work

**Related Issues:**
Link to related performance issues if any.

## Checklist

- [ ] I have profiled the code
- [ ] I have compared with other models
- [ ] I have checked GPU utilization
- [ ] I have tried different batch sizes
- [ ] I have checked the [performance guide](../../docs/PERFORMANCE.md)
