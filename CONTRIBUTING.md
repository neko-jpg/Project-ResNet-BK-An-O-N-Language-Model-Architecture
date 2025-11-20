# Contributing to ResNet-BK

Thank you for your interest in contributing to ResNet-BK! This document provides guidelines for contributing to the project.

## 🌟 Ways to Contribute

We welcome contributions of all kinds:

- **Bug Reports**: Found a bug? Please open an issue with detailed reproduction steps
- **Feature Requests**: Have an idea? Share it in the issues
- **Code Contributions**: Submit pull requests for bug fixes or new features
- **Documentation**: Help improve our docs, add examples, or fix typos
- **Testing**: Run experiments and share results
- **Research**: Validate theoretical claims or propose improvements

## 🚀 Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/Project-ResNet-BK-An-O-N-Language-Model-Architecture.git
cd Project-ResNet-BK-An-O-N-Language-Model-Architecture
```

### 2. Set Up Development Environment

#### Option A: Using Docker (Recommended)

```bash
# Build the Docker image
docker-compose build

# Start the container
docker-compose up -d

# Enter the container
docker exec -it mamba-killer-dev bash

# Run tests
pytest tests/
```

#### Option B: Local Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .

# Run tests
pytest tests/
```

### 3. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

## 📝 Development Guidelines

### Code Style

We use **Black** for code formatting and **flake8** for linting:

```bash
# Format code
black src/ tests/ examples/

# Check linting
flake8 src/ tests/ examples/
```

### Type Hints

Please add type hints to all function signatures:

```python
def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Forward pass with type hints.
    
    Args:
        x: Input tensor (B, N, D)
        state: Optional state tensor
    
    Returns:
        output: Output tensor (B, N, D)
        new_state: Updated state tensor
    """
    pass
```

### Documentation

- Add docstrings to all public functions and classes (Google style)
- Include mathematical formulas where relevant
- Provide usage examples in docstrings

Example:

```python
def compute_bk_core(
    h_diag: torch.Tensor,
    h_super: torch.Tensor,
    h_sub: torch.Tensor,
    z: complex
) -> torch.Tensor:
    """
    Compute BK-Core using Birman-Schwinger operator.
    
    Mathematical Background:
        G_ii = diag((H - zI)^{-1})
        where H is a tridiagonal matrix with potential V.
    
    Args:
        h_diag: Diagonal elements (N,)
        h_super: Super-diagonal elements (N-1,)
        h_sub: Sub-diagonal elements (N-1,)
        z: Complex energy parameter
    
    Returns:
        Green's function diagonal (N,) complex tensor
    
    Example:
        >>> h_diag = torch.randn(512)
        >>> h_super = torch.randn(511)
        >>> h_sub = torch.randn(511)
        >>> z = 0.1 + 0.1j
        >>> g = compute_bk_core(h_diag, h_super, h_sub, z)
        >>> g.shape
        torch.Size([512])
    """
    pass
```

### Testing

- Write tests for all new features
- Ensure all tests pass before submitting PR
- Aim for >80% code coverage

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_bk_core.py

# Run with coverage
pytest --cov=src tests/
```

### Commit Messages

Follow conventional commits format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Examples:

```
feat(phase2): add dissipative Hebbian layer

Implement fast weights with Lyapunov stability monitoring.
Includes unit tests and documentation.

Closes #123
```

```
fix(bk-core): resolve CUDA memory access error

Use torch.bmm instead of einsum for better CUDA compatibility.
Fixes illegal memory access in complex number operations.

Fixes #456
```

## 🧪 Running Experiments

### Quick Test

```bash
# Run Phase 1 demo
python examples/phase1_integration_demo.py

# Run Phase 2 demo
python examples/phase2_basic_usage.py
```

### Benchmarks

```bash
# Memory benchmark
python scripts/validate_phase1_memory.py

# Throughput benchmark
python scripts/benchmark_phase1_throughput.py

# BK-Core Triton benchmark
python scripts/benchmark_bk_triton.py
```

### Training

```bash
# Phase 1 training
python scripts/train_phase1.py --config configs/phase1_small.yaml

# Phase 2 training
python scripts/train_phase2.py --config configs/phase2_small.yaml
```

## 📊 Submitting Results

If you run experiments and get interesting results:

1. Save results in JSON format to `results/benchmarks/`
2. Include hardware specifications (GPU model, VRAM, etc.)
3. Document hyperparameters used
4. Share in an issue or PR

Example result file:

```json
{
  "experiment": "phase2_stability_test",
  "date": "2025-11-20",
  "hardware": {
    "gpu": "NVIDIA RTX 3080",
    "vram_gb": 10,
    "cuda_version": "11.8"
  },
  "config": {
    "model": "phase2_small",
    "batch_size": 4,
    "seq_length": 2048
  },
  "results": {
    "peak_vram_mb": 6890,
    "throughput_tokens_per_sec": 824.7,
    "perplexity": 50.5
  }
}
```

## 🔬 Research Contributions

We especially welcome:

- **Theoretical Analysis**: Proofs, bounds, convergence analysis
- **Ablation Studies**: Which components matter most?
- **Comparison Studies**: How does it compare to other models?
- **Scaling Studies**: Behavior at different model sizes
- **Application Studies**: Performance on specific tasks

## 📋 Pull Request Process

1. **Update Documentation**: Ensure README, docstrings, and relevant docs are updated
2. **Add Tests**: Include tests for new functionality
3. **Run Tests**: Ensure all tests pass
4. **Format Code**: Run `black` and `flake8`
5. **Update CHANGELOG**: Add entry describing your changes
6. **Submit PR**: Provide clear description of changes and motivation

### PR Template

```markdown
## Description
Brief description of changes

## Motivation
Why is this change needed?

## Changes
- Change 1
- Change 2

## Testing
How was this tested?

## Checklist
- [ ] Tests pass
- [ ] Code formatted with Black
- [ ] Documentation updated
- [ ] CHANGELOG updated
```

## 🐛 Reporting Bugs

When reporting bugs, please include:

1. **Environment**: OS, Python version, PyTorch version, GPU model
2. **Reproduction Steps**: Minimal code to reproduce the issue
3. **Expected Behavior**: What should happen
4. **Actual Behavior**: What actually happens
5. **Error Messages**: Full error traceback
6. **Additional Context**: Any other relevant information

## 💬 Communication

- **GitHub Issues**: For bugs, features, and discussions
- **Pull Requests**: For code contributions
- **Email**: arat252539@gmail.com for private inquiries

## 📜 Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Assume good intentions

## 🙏 Recognition

All contributors will be acknowledged in:
- CONTRIBUTORS.md file
- Paper acknowledgments (for significant contributions)
- Release notes

Thank you for contributing to ResNet-BK! 🚀
