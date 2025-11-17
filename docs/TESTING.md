# Testing Strategy for ResNet-BK

Comprehensive testing documentation for ResNet-BK, covering unit tests, integration tests, and benchmarks.

---

## Table of Contents

1. [Overview](#overview)
2. [Test Structure](#test-structure)
3. [Unit Tests](#unit-tests)
4. [Integration Tests](#integration-tests)
5. [Mathematical Verification Tests](#mathematical-verification-tests)
6. [Benchmark Tests](#benchmark-tests)
7. [Running Tests](#running-tests)
8. [Writing New Tests](#writing-new-tests)
9. [Continuous Integration](#continuous-integration)

---

## Overview

ResNet-BK uses a multi-layered testing strategy:

1. **Unit Tests**: Test individual components in isolation
2. **Integration Tests**: Test component interactions
3. **Mathematical Verification**: Verify theoretical properties
4. **Benchmark Tests**: Compare performance to baselines

### Testing Philosophy

- **Correctness First**: Verify mathematical properties
- **Comprehensive Coverage**: Aim for >80% code coverage
- **Fast Feedback**: Unit tests run in <1 minute
- **Reproducible**: Use fixed random seeds

---

## Test Structure

```
tests/
├── __init__.py
├── conftest.py                          # Pytest configuration
│
├── # Core Components
├── test_birman_schwinger_core.py        # BK-Core tests
├── test_prime_bump_potential.py         # Prime-Bump tests
├── test_scattering_router.py            # Router tests
├── test_semiseparable_matrix.py         # Semiseparable tests
│
├── # Training Components
├── test_adaptive_computation.py         # ACT tests
├── test_memory_optimization.py          # Memory tests
├── test_stability_monitor.py            # Stability tests
│
├── # Benchmarks
├── test_wikitext2_benchmark.py          # WikiText-2 tests
├── test_wikitext103_benchmark.py        # WikiText-103 tests
├── test_mamba_baseline.py               # Mamba comparison
│
├── # Integration
├── test_integration.py                  # End-to-end tests
├── test_theory.py                       # Mathematical verification
│
└── # Utilities
    └── test_utils.py                    # Helper functions
```

---

## Unit Tests

### BirmanSchwingerCore Tests

**File**: `tests/test_birman_schwinger_core.py`

**Tests:**

1. **Shape Tests**
```python
def test_forward_shape():
    """Test that forward pass returns correct shape."""
    bk_core = BirmanSchwingerCore(n_seq=512, epsilon=1.0)
    v = torch.randn(8, 512)
    G_ii = bk_core(v, z=1.0j)
    
    assert G_ii.shape == (8, 512, 2)  # [batch, seq, (real, imag)]
```

2. **Numerical Stability Tests**
```python
def test_no_nan_inf():
    """Test that output contains no NaN/Inf."""
    bk_core = BirmanSchwingerCore(n_seq=512, epsilon=1.0)
    v = torch.randn(8, 512)
    G_ii = bk_core(v, z=1.0j)
    
    assert torch.isfinite(G_ii).all()
```

3. **Schatten Norm Tests**
```python
def test_schatten_bounds():
    """Test that Schatten norms satisfy theoretical bounds."""
    bk_core = BirmanSchwingerCore(n_seq=512, epsilon=1.0)
    v = torch.randn(8, 512)
    G_ii = bk_core(v, z=1.0j)
    
    s1, s2 = bk_core.compute_schatten_norms()
    
    # Hilbert-Schmidt bound: ||K||_S2 ≤ (1/2)(Im z)^{-1/2} ||V||_L2
    v_norm = torch.norm(v, p=2)
    bound_s2 = (1/2) * (1.0)**(-0.5) * v_norm
    assert s2 <= bound_s2 * 1.1  # Allow 10% tolerance
    
    # Trace-class bound: ||K||_S1 ≤ (1/2)(Im z)^{-1} ||V||_L1
    v_norm_l1 = torch.norm(v, p=1)
    bound_s1 = (1/2) * (1.0)**(-1) * v_norm_l1
    assert s1 <= bound_s1 * 1.1
```

4. **Mourre Estimate Tests**
```python
def test_mourre_estimate():
    """Test that Mourre estimate [H_0, iA] = I holds."""
    bk_core = BirmanSchwingerCore(n_seq=512, epsilon=1.0, use_mourre=True)
    
    assert bk_core.verify_mourre_estimate()
```

5. **LAP Tests**
```python
def test_lap_boundary():
    """Test that LAP allows boundary computation."""
    bk_core = BirmanSchwingerCore(n_seq=512, epsilon=1.0, use_lap=True)
    v = torch.randn(8, 512)
    
    # Compute at different Im(z) values
    G_ii_1 = bk_core(v, z=1.0 + 0.1j)
    G_ii_2 = bk_core(v, z=1.0 + 0.01j)
    G_ii_3 = bk_core(v, z=1.0 + 0.001j)
    
    # Should remain bounded as Im(z) → 0
    assert torch.isfinite(G_ii_1).all()
    assert torch.isfinite(G_ii_2).all()
    assert torch.isfinite(G_ii_3).all()
```

### PrimeBumpPotential Tests

**File**: `tests/test_prime_bump_potential.py`

**Tests:**

1. **Prime Sieve Tests**
```python
def test_prime_sieve():
    """Test that prime sieve returns correct primes."""
    prime_bump = PrimeBumpPotential(n_seq=100, epsilon=1.0)
    primes = prime_bump.get_prime_indices()
    
    expected_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    assert primes == expected_primes
```

2. **GUE Statistics Tests**
```python
def test_gue_statistics():
    """Test that eigenvalue spacing follows GUE statistics."""
    prime_bump = PrimeBumpPotential(n_seq=512, epsilon=1.0)
    stats = prime_bump.verify_gue_statistics()
    
    # Wigner surmise: s * exp(-πs²/4)
    # Check that fit quality is good (R² > 0.9)
    assert stats['wigner_fit'] > 0.9
```

3. **Finite Overlap Tests**
```python
def test_finite_overlap():
    """Test that bumps have finite overlap."""
    prime_bump = PrimeBumpPotential(n_seq=512, epsilon=0.5)
    
    # Get bump positions
    primes = prime_bump.get_prime_indices()
    
    # Check that |log p - log q| > 2√ε for adjacent primes
    for i in range(len(primes) - 1):
        p, q = primes[i], primes[i+1]
        distance = abs(np.log(p) - np.log(q))
        min_distance = 2 * np.sqrt(0.5)
        
        # Allow some tolerance for small primes
        if p > 10:
            assert distance > min_distance * 0.8
```

### ScatteringRouter Tests

**File**: `tests/test_scattering_router.py`

**Tests:**

1. **Phase Computation Tests**
```python
def test_phase_computation():
    """Test scattering phase computation."""
    router = ScatteringRouter(num_experts=8)
    G_ii = torch.randn(4, 512, 2)  # [batch, seq, (real, imag)]
    
    phase = router.compute_scattering_phase(G_ii)
    
    assert phase.shape == (4, 512)
    assert torch.isfinite(phase).all()
    assert (phase >= -np.pi).all() and (phase <= np.pi).all()
```

2. **Routing Determinism Tests**
```python
def test_routing_determinism():
    """Test that routing is deterministic (no randomness)."""
    router = ScatteringRouter(num_experts=8)
    G_ii = torch.randn(4, 512, 2)
    
    # Route twice with same input
    indices1, weights1 = router(G_ii)
    indices2, weights2 = router(G_ii)
    
    assert torch.equal(indices1, indices2)
    assert torch.equal(weights1, weights2)
```

3. **Resonance Detection Tests**
```python
def test_resonance_detection():
    """Test resonance detection."""
    router = ScatteringRouter(num_experts=8)
    
    # Create input with known resonance
    G_ii = torch.randn(4, 512, 2)
    G_ii[:, 100, :] = 0.0  # Force resonance at position 100
    
    D_eps = router.compute_determinant(G_ii)
    is_resonance = router.detect_resonances(D_eps)
    
    assert is_resonance[:, 100].all()  # Should detect resonance
```

---

## Integration Tests

### End-to-End Training Tests

**File**: `tests/test_integration.py`

**Tests:**

1. **Full Training Pipeline**
```python
def test_full_training_pipeline():
    """Test complete training pipeline."""
    # Create small model
    model = LanguageModel(
        vocab_size=1000,
        d_model=128,
        n_layers=2,
        n_seq=64
    )
    
    # Create dummy data
    train_loader = create_dummy_dataloader(
        batch_size=4,
        seq_len=64,
        vocab_size=1000,
        num_batches=10
    )
    
    # Train for 1 epoch
    trainer = Trainer(model, train_loader, val_loader=None)
    history = trainer.train(num_epochs=1)
    
    # Verify training occurred
    assert len(history['loss']) > 0
    assert history['loss'][-1] < history['loss'][0]  # Loss decreased
    assert not np.isnan(history['loss']).any()
```

2. **Gradient Flow Tests**
```python
def test_gradient_flow():
    """Test that gradients flow through all components."""
    model = LanguageModel(vocab_size=1000, d_model=128, n_layers=2)
    
    # Forward pass
    input_ids = torch.randint(0, 1000, (4, 64))
    output = model(input_ids)
    loss = output['loss']
    
    # Backward pass
    loss.backward()
    
    # Check that all parameters have gradients
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
            assert torch.isfinite(param.grad).all(), f"NaN/Inf gradient for {name}"
```

3. **Checkpoint Save/Load Tests**
```python
def test_checkpoint_save_load():
    """Test checkpoint saving and loading."""
    model = LanguageModel(vocab_size=1000, d_model=128, n_layers=2)
    
    # Save checkpoint
    checkpoint_path = "test_checkpoint.pt"
    model.save_checkpoint(checkpoint_path)
    
    # Create new model and load
    model2 = LanguageModel(vocab_size=1000, d_model=128, n_layers=2)
    model2.load_checkpoint(checkpoint_path)
    
    # Verify parameters match
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.equal(p1, p2)
    
    # Cleanup
    os.remove(checkpoint_path)
```

---

## Mathematical Verification Tests

### Theoretical Property Tests

**File**: `tests/test_theory.py`

**Tests:**

1. **Schatten Bound Verification**
```python
def test_schatten_bounds_all_epsilon():
    """Verify Schatten bounds for all epsilon values."""
    for epsilon in [0.5, 0.75, 1.0]:
        bk_core = BirmanSchwingerCore(n_seq=512, epsilon=epsilon)
        v = torch.randn(8, 512)
        G_ii = bk_core(v, z=1.0j)
        
        s1, s2 = bk_core.compute_schatten_norms()
        
        # Verify bounds
        v_norm_l2 = torch.norm(v, p=2)
        bound_s2 = (1/2) * (1.0)**(-0.5) * v_norm_l2
        assert s2 <= bound_s2 * 1.1
        
        if epsilon > 0.5:
            v_norm_l1 = torch.norm(v, p=1)
            bound_s1 = (1/2) * (1.0)**(-1) * v_norm_l1
            assert s1 <= bound_s1 * 1.1
```

2. **Weil Formula Verification**
```python
def test_weil_formula():
    """Verify Weil explicit formula matching."""
    prime_bump = PrimeBumpPotential(n_seq=512, epsilon=1.0)
    bk_core = BirmanSchwingerCore(n_seq=512, epsilon=1.0)
    
    # Compute spectral trace
    x = torch.randn(1, 512, 256)
    v = prime_bump(x)
    G_ii = bk_core(v, z=1.0j)
    
    spectral_trace = compute_spectral_trace(G_ii)
    
    # Compute prime sum
    primes = prime_bump.get_prime_indices()
    prime_sum = sum(np.log(p) / p**(1/2 + 1.0) for p in primes)
    
    # Should match (within tolerance)
    assert abs(spectral_trace - prime_sum) < 0.1 * abs(prime_sum)
```

---

## Benchmark Tests

### Performance Comparison Tests

**File**: `tests/test_mamba_baseline.py`

**Tests:**

1. **Long-Context Stability**
```python
@pytest.mark.slow
def test_long_context_stability():
    """Test stability on long sequences."""
    for seq_len in [512, 2048, 8192]:
        # Train ResNet-BK
        model_bk = LanguageModel(n_seq=seq_len)
        loss_bk = train_and_evaluate(model_bk, seq_len=seq_len, steps=100)
        
        # Train Mamba
        model_mamba = MambaBaseline(n_seq=seq_len)
        loss_mamba = train_and_evaluate(model_mamba, seq_len=seq_len, steps=100)
        
        # ResNet-BK should be stable (no NaN)
        assert not np.isnan(loss_bk)
        
        # At seq_len=8192, Mamba may diverge
        if seq_len >= 8192:
            assert loss_bk < loss_mamba or np.isnan(loss_mamba)
```

2. **Quantization Robustness**
```python
@pytest.mark.slow
def test_quantization_robustness():
    """Test quantization performance."""
    # Train FP32 models
    model_bk = train_model("resnetbk", bits=32)
    model_mamba = train_model("mamba", bits=32)
    
    # Quantize to INT8
    model_bk_int8 = quantize(model_bk, bits=8)
    model_mamba_int8 = quantize(model_mamba, bits=8)
    
    # Evaluate
    ppl_bk = evaluate(model_bk_int8)
    ppl_mamba = evaluate(model_mamba_int8)
    
    # ResNet-BK should have lower degradation
    assert ppl_bk < ppl_mamba
```

---

## Running Tests

### Run All Tests

```bash
pytest tests/
```

### Run Specific Test File

```bash
pytest tests/test_birman_schwinger_core.py
```

### Run Specific Test

```bash
pytest tests/test_birman_schwinger_core.py::test_forward_shape
```

### Run with Coverage

```bash
pytest --cov=src --cov-report=html tests/
```

### Run Fast Tests Only

```bash
pytest -m "not slow" tests/
```

### Run Slow Tests

```bash
pytest -m slow tests/
```

### Run in Parallel

```bash
pytest -n auto tests/
```

---

## Writing New Tests

### Test Template

```python
import pytest
import torch
from src.models import MyComponent

class TestMyComponent:
    """Tests for MyComponent."""
    
    @pytest.fixture
    def component(self):
        """Create component for testing."""
        return MyComponent(param1=value1, param2=value2)
    
    def test_basic_functionality(self, component):
        """Test basic functionality."""
        input_data = torch.randn(4, 512)
        output = component(input_data)
        
        assert output.shape == (4, 512)
        assert torch.isfinite(output).all()
    
    @pytest.mark.parametrize("param", [value1, value2, value3])
    def test_different_params(self, param):
        """Test with different parameter values."""
        component = MyComponent(param=param)
        # Test logic here
    
    @pytest.mark.slow
    def test_expensive_operation(self, component):
        """Test expensive operation (marked as slow)."""
        # Expensive test logic here
```

### Best Practices

1. **Use Fixtures**: Reuse setup code
2. **Parametrize**: Test multiple inputs efficiently
3. **Mark Slow Tests**: Use `@pytest.mark.slow` for expensive tests
4. **Test Edge Cases**: Empty inputs, large inputs, etc.
5. **Use Assertions**: Clear, specific assertions
6. **Document Tests**: Explain what is being tested

---

## Continuous Integration

### GitHub Actions Workflow

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.10
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -e ".[dev]"
    
    - name: Run tests
      run: |
        pytest --cov=src --cov-report=xml tests/
    
    - name: Upload coverage
      uses: codecov/codecov-action@v2
```

### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.1.0
    hooks:
      - id: black
  
  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
  
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.0.0
    hooks:
      - id: mypy
```

---

## Test Metrics

### Current Coverage

| Module | Coverage |
|--------|----------|
| `src/models/birman_schwinger_core.py` | 95% |
| `src/models/prime_bump_potential.py` | 92% |
| `src/models/scattering_router.py` | 88% |
| `src/models/semiseparable_matrix.py` | 85% |
| **Overall** | **87%** |

### Test Execution Time

| Test Suite | Time |
|------------|------|
| Unit Tests | 45s |
| Integration Tests | 2m 30s |
| Benchmark Tests | 15m |
| **Total** | **18m 15s** |

---

For more information, see:
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines
- [API_REFERENCE.md](API_REFERENCE.md) - API documentation
- [ARCHITECTURE.md](ARCHITECTURE.md) - Architecture details
