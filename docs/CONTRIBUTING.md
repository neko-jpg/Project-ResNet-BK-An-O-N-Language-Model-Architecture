# Contributing to ResNet-BK

Thank you for your interest in contributing to ResNet-BK! This document provides guidelines and instructions for contributing.

---

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [Contribution Workflow](#contribution-workflow)
5. [Coding Standards](#coding-standards)
6. [Testing Guidelines](#testing-guidelines)
7. [Documentation](#documentation)
8. [Pull Request Process](#pull-request-process)
9. [Community](#community)

---

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors, regardless of:
- Experience level
- Background
- Identity
- Nationality

### Expected Behavior

- Be respectful and constructive
- Welcome newcomers and help them get started
- Focus on what is best for the community
- Show empathy towards other community members

### Unacceptable Behavior

- Harassment, discrimination, or offensive comments
- Trolling, insulting, or derogatory remarks
- Publishing others' private information
- Other conduct which could reasonably be considered inappropriate

### Enforcement

Violations of the code of conduct may result in:
1. Warning
2. Temporary ban
3. Permanent ban

Report violations to [arat252539@gmail.com].

---

## Getting Started

### Ways to Contribute

1. **Report Bugs**: Found a bug? Create an issue!
2. **Suggest Features**: Have an idea? We'd love to hear it!
3. **Improve Documentation**: Help others understand the code
4. **Write Tests**: Increase code coverage
5. **Fix Bugs**: Tackle open issues
6. **Add Features**: Implement new functionality
7. **Optimize Performance**: Make it faster!
8. **Share Results**: Train models and share findings

### Good First Issues

Look for issues labeled `good-first-issue` - these are great for newcomers!

---

## Development Setup

### Prerequisites

- Python 3.8+
- Git
- CUDA 11.8+ (for GPU development)

### Setup Instructions

1. **Fork the repository**

Click "Fork" on GitHub to create your own copy.

2. **Clone your fork**

```bash
git clone https://github.com/YOUR_USERNAME/Project-ResNet-BK-An-O-N-Language-Model-Architecture.git
cd Project-ResNet-BK-An-O-N-Language-Model-Architecture
```

3. **Add upstream remote**

```bash
git remote add upstream https://github.com/neko-jpg/Project-ResNet-BK-An-O-N-Language-Model-Architecture.git
```

4. **Create virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

5. **Install dependencies**

```bash
# Install package in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

6. **Verify installation**

```bash
# Run tests
pytest tests/

# Check code style
flake8 src/
black --check src/
```

---

## Contribution Workflow

### 1. Create a Branch

```bash
# Update main branch
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/my-awesome-feature
```

**Branch naming conventions:**
- `feature/` - New features
- `bugfix/` - Bug fixes
- `docs/` - Documentation changes
- `test/` - Test additions/fixes
- `refactor/` - Code refactoring

### 2. Make Changes

- Write clean, readable code
- Follow coding standards (see below)
- Add tests for new functionality
- Update documentation as needed

### 3. Test Your Changes

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_birman_schwinger_core.py

# Run with coverage
pytest --cov=src tests/

# Check code style
flake8 src/
black --check src/
mypy src/
```

### 4. Commit Changes

```bash
# Stage changes
git add .

# Commit with descriptive message
git commit -m "Add feature: scattering phase visualization"
```

**Commit message format:**
```
<type>: <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Test additions/fixes
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `style`: Code style changes
- `chore`: Maintenance tasks

**Example:**
```
feat: Add scattering phase visualization

Implement visualization of scattering phase δ_ε(λ) for each token.
Includes correlation with linguistic difficulty (perplexity).

Closes #123
```

### 5. Push Changes

```bash
git push origin feature/my-awesome-feature
```

### 6. Create Pull Request

1. Go to your fork on GitHub
2. Click "New Pull Request"
3. Select your branch
4. Fill out the PR template
5. Submit!

---

## Coding Standards

### Python Style Guide

We follow [PEP 8](https://pep8.org/) with some modifications:

- **Line length**: 100 characters (not 79)
- **Indentation**: 4 spaces (no tabs)
- **Quotes**: Double quotes for strings
- **Imports**: Organized by standard library, third-party, local

### Code Formatting

We use [Black](https://black.readthedocs.io/) for automatic formatting:

```bash
# Format all files
black src/ tests/

# Check without modifying
black --check src/
```

### Type Hints

Use type hints for all function signatures:

```python
def compute_scattering_phase(
    G_ii: torch.Tensor,
    epsilon: float = 1.0
) -> torch.Tensor:
    """
    Compute scattering phase δ_ε(λ).
    
    Args:
        G_ii: Complex resolvent diagonal [batch, n_seq, 2]
        epsilon: Regularization parameter
        
    Returns:
        phase: Scattering phase [batch, n_seq]
    """
    ...
```

### Docstrings

Use Google-style docstrings:

```python
def my_function(arg1: int, arg2: str) -> bool:
    """
    Short description of function.
    
    Longer description with more details about what the function does,
    including any important notes or caveats.
    
    Args:
        arg1: Description of arg1
        arg2: Description of arg2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When arg1 is negative
        
    Examples:
        >>> result = my_function(42, "hello")
        >>> print(result)
        True
    """
    ...
```

### Naming Conventions

- **Classes**: `PascalCase` (e.g., `BirmanSchwingerCore`)
- **Functions**: `snake_case` (e.g., `compute_scattering_phase`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_SEQUENCE_LENGTH`)
- **Private**: Prefix with `_` (e.g., `_internal_helper`)

### Code Organization

```python
# 1. Standard library imports
import os
import sys
from typing import Dict, List, Optional, Tuple

# 2. Third-party imports
import torch
import torch.nn as nn
import numpy as np

# 3. Local imports
from src.models import BirmanSchwingerCore
from src.utils import compute_schatten_norm

# 4. Constants
MAX_SEQUENCE_LENGTH = 1048576
DEFAULT_EPSILON = 1.0

# 5. Classes and functions
class MyClass:
    ...

def my_function():
    ...
```

---

## Testing Guidelines

### Test Structure

```
tests/
├── test_birman_schwinger_core.py
├── test_prime_bump_potential.py
├── test_scattering_router.py
├── test_semiseparable_matrix.py
└── test_integration.py
```

### Writing Tests

Use `pytest` for all tests:

```python
import pytest
import torch
from src.models import BirmanSchwingerCore

class TestBirmanSchwingerCore:
    """Tests for BirmanSchwingerCore."""
    
    def test_forward_shape(self):
        """Test that forward pass returns correct shape."""
        bk_core = BirmanSchwingerCore(n_seq=512, epsilon=1.0)
        v = torch.randn(8, 512)
        G_ii = bk_core(v, z=1.0j)
        
        assert G_ii.shape == (8, 512, 2)
    
    def test_schatten_bounds(self):
        """Test that Schatten norms satisfy bounds."""
        bk_core = BirmanSchwingerCore(n_seq=512, epsilon=1.0)
        v = torch.randn(8, 512)
        G_ii = bk_core(v, z=1.0j)
        
        s1, s2 = bk_core.compute_schatten_norms()
        
        # Verify Hilbert-Schmidt bound
        assert s2 <= (1/2) * (1.0)**(-0.5) * torch.norm(v, p=2)
    
    def test_mourre_estimate(self):
        """Test that Mourre estimate holds."""
        bk_core = BirmanSchwingerCore(n_seq=512, epsilon=1.0)
        
        assert bk_core.verify_mourre_estimate()
    
    @pytest.mark.parametrize("epsilon", [0.5, 0.75, 1.0])
    def test_epsilon_values(self, epsilon):
        """Test different epsilon values."""
        bk_core = BirmanSchwingerCore(n_seq=512, epsilon=epsilon)
        v = torch.randn(8, 512)
        G_ii = bk_core(v, z=1.0j)
        
        assert torch.isfinite(G_ii).all()
```

### Test Coverage

Aim for >80% code coverage:

```bash
# Run with coverage
pytest --cov=src --cov-report=html tests/

# View coverage report
open htmlcov/index.html
```

### Integration Tests

Test end-to-end workflows:

```python
def test_full_training_pipeline():
    """Test complete training pipeline."""
    # 1. Create model
    model = LanguageModel(vocab_size=1000, d_model=128, n_layers=2)
    
    # 2. Create data
    train_loader = create_dummy_dataloader(batch_size=4, seq_len=64)
    
    # 3. Train
    trainer = Trainer(model, train_loader, val_loader=None)
    trainer.train(num_epochs=1)
    
    # 4. Verify
    assert model.training_steps > 0
    assert not torch.isnan(model.get_loss())
```

---

## Documentation

### Code Documentation

- Add docstrings to all public functions and classes
- Include examples in docstrings
- Document complex algorithms with inline comments

### User Documentation

When adding features, update:
- `README.md` - If it affects quick start or overview
- `TUTORIAL.md` - If it requires new tutorial section
- `API_REFERENCE.md` - If it adds new public API
- `FAQ.md` - If it addresses common questions

### Mathematical Documentation

For mathematical components:
- Reference theorem/proposition from paper
- Include formula in docstring
- Explain physical/mathematical meaning

Example:
```python
def compute_scattering_phase(self, G_ii: torch.Tensor) -> torch.Tensor:
    """
    Compute scattering phase δ_ε(λ) = arg(det_2(I + K_ε(λ + i0))).
    
    Based on Proposition BK-formula from riemann_hypothesis_main.tex:
    d/dλ log D_ε(λ) = -Tr((H_ε - λ)^{-1} - (H_0 - λ)^{-1})
    
    The scattering phase represents the phase shift induced by the
    potential V_ε, and correlates with linguistic difficulty.
    
    Args:
        G_ii: Complex resolvent diagonal [batch, n_seq, 2]
        
    Returns:
        phase: Scattering phase [batch, n_seq]
        
    References:
        - Proposition BK-formula (riemann_hypothesis_main.tex)
        - Birman-Krein formula for spectral shift
    """
    ...
```

---

## Pull Request Process

### Before Submitting

- [ ] Tests pass: `pytest tests/`
- [ ] Code is formatted: `black src/`
- [ ] No linting errors: `flake8 src/`
- [ ] Type hints are correct: `mypy src/`
- [ ] Documentation is updated
- [ ] Commit messages are clear

### PR Template

When creating a PR, fill out this template:

```markdown
## Description

Brief description of changes.

## Type of Change

- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing

Describe tests you've added/run.

## Checklist

- [ ] Tests pass
- [ ] Code is formatted
- [ ] Documentation updated
- [ ] No breaking changes (or documented)

## Related Issues

Closes #123
```

### Review Process

1. **Automated Checks**: CI runs tests and linting
2. **Code Review**: Maintainer reviews code
3. **Feedback**: Address review comments
4. **Approval**: Maintainer approves PR
5. **Merge**: PR is merged to main

### After Merge

- Delete your feature branch
- Update your fork:
```bash
git checkout main
git pull upstream main
git push origin main
```

---

## Community

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and discussions
- **Pull Requests**: Code contributions

### Getting Help

- Check [FAQ.md](FAQ.md) first
- Search existing issues
- Ask in GitHub Discussions
- Create new issue if needed

### Recognition

Contributors are recognized in:
- `CONTRIBUTORS.md` file
- Release notes
- Paper acknowledgments (for significant contributions)

---

## Development Tips

### Debugging

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Use pdb for debugging
import pdb; pdb.set_trace()

# Check tensor values
print(f"Shape: {tensor.shape}, Min: {tensor.min()}, Max: {tensor.max()}")
```

### Profiling

```python
# Profile code
import cProfile
cProfile.run('train_one_epoch()', 'profile.stats')

# Analyze profile
import pstats
p = pstats.Stats('profile.stats')
p.sort_stats('cumulative').print_stats(20)
```

### Memory Debugging

```python
# Track memory usage
import torch
torch.cuda.reset_peak_memory_stats()
train_one_batch()
peak_memory = torch.cuda.max_memory_allocated() / 1e9
print(f"Peak memory: {peak_memory:.2f} GB")
```

---

## Questions?

If you have questions about contributing:

1. Check this guide
2. Search [GitHub Issues](https://github.com/neko-jpg/Project-ResNet-BK-An-O-N-Language-Model-Architecture/issues)
3. Ask in [GitHub Discussions](https://github.com/neko-jpg/Project-ResNet-BK-An-O-N-Language-Model-Architecture/discussions)
4. Contact maintainers

---

**Thank you for contributing to ResNet-BK! 🚀**
