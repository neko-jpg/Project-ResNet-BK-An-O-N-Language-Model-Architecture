---
name: Bug Report
about: Report a bug to help us improve ResNet-BK
title: '[BUG] '
labels: bug
assignees: ''
---

## Bug Description

A clear and concise description of what the bug is.

## To Reproduce

Steps to reproduce the behavior:

1. Install ResNet-BK version: `X.Y.Z`
2. Run command: `python ...`
3. See error

**Minimal Code Example:**
```python
# Paste minimal code that reproduces the issue
from src.models.resnet_bk import ResNetBK

model = ResNetBK(...)
# ...
```

## Expected Behavior

A clear and concise description of what you expected to happen.

## Actual Behavior

What actually happened. Include error messages and stack traces.

```
Paste error message here
```

## Environment

**System Information:**
- OS: [e.g., Ubuntu 22.04, macOS 13.0, Windows 11]
- Python version: [e.g., 3.10.8]
- PyTorch version: [e.g., 2.1.0]
- CUDA version: [e.g., 12.1]
- ResNet-BK version: [e.g., 0.9.0]

**Hardware:**
- GPU: [e.g., NVIDIA RTX 4090, T4, None]
- RAM: [e.g., 32GB]
- VRAM: [e.g., 24GB]

**Installation Method:**
- [ ] pip install
- [ ] From source
- [ ] Docker
- [ ] Google Colab

## Additional Context

Add any other context about the problem here.

**Screenshots:**
If applicable, add screenshots to help explain your problem.

**Related Issues:**
Link to related issues if any.

## Checklist

- [ ] I have searched existing issues to avoid duplicates
- [ ] I have provided a minimal reproducible example
- [ ] I have included my environment information
- [ ] I have checked the [FAQ](../../FAQ.md)
- [ ] I have checked the [documentation](https://resnet-bk.readthedocs.io)
