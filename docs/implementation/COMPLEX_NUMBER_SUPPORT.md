# Complex Number Support Infrastructure

## Overview

Phase 2では非エルミート演算子による「忘却機構」を実装するため、複素数テンソルのサポートが必要になります。このドキュメントでは、Phase 1で準備された複素数サポートインフラストラクチャについて説明します。

## Physical Intuition (物理的直観)

- **Phase 2の非エルミート演算子**: 複素固有値を持つ
- **複素数の実部**: エネルギー（記憶の強度）
- **複素数の虚部**: 減衰率（忘却の速度）
- **時間発展**: exp(-iHt)により、自然な忘却機構を実現

## Mathematical Foundation

### Complex Tensor Representation

```
実数テンソル: R^n
複素数テンソル: C^n = R^n + iR^n

PyTorch dtypes:
- torch.complex64: 32-bit real + 32-bit imaginary
- torch.complex128: 64-bit real + 64-bit imaginary
```

### Conversion Between Real and Complex

```python
# Real to Complex
x_real = torch.randn(10, 20)
x_complex = real_to_complex(x_real)  # Imaginary part = 0

# Complex to Real (multiple modes)
x_complex = torch.complex(torch.randn(10, 20), torch.randn(10, 20))

# Mode 1: Concatenate [real, imag]
x_concat = complex_to_real(x_complex, mode='concat')  # Shape: (10, 20, 2)

# Mode 2: Separate (real, imag)
real, imag = complex_to_real(x_complex, mode='separate')  # Each: (10, 20)

# Mode 3: Magnitude only
magnitude = complex_to_real(x_complex, mode='magnitude')  # Shape: (10, 20)

# Mode 4: Phase only
phase = complex_to_real(x_complex, mode='phase')  # Shape: (10, 20)
```

## Component Support Status

### AR-SSM Layer

**Current Status (Phase 1)**: Real-valued only

**Complex Support**: Partial
- Can accept complex inputs but converts to real internally
- Uses real part only, discards imaginary part
- Issues warning when complex input detected

**Phase 2 Roadmap**:
- Complex-valued gates for non-Hermitian dynamics
- Complex projections (U_proj, V_proj)
- Complex cumulative sum operations

**Example**:
```python
from src.models.phase1 import AdaptiveRankSemiseparableLayer

layer = AdaptiveRankSemiseparableLayer(d_model=128, max_rank=32)

# Phase 1: Real input (recommended)
x_real = torch.randn(4, 100, 128)
y, diagnostics = layer(x_real)

# Phase 1: Complex input (converts to real with warning)
x_complex = torch.complex(torch.randn(4, 100, 128), torch.randn(4, 100, 128))
y, diagnostics = layer(x_complex)  # Warning issued
assert diagnostics['input_was_complex'] == True
```

### HTT Embedding

**Current Status (Phase 1)**: Real-valued phase rotation (cos(θ))

**Complex Support**: Prepared for Phase 2
- Uses cos(θ) amplitude modulation in Phase 1
- `forward_complex()` method implements full exp(iθ) rotation
- Phase parameters designed to be extensible to complex

**Phase 2 Roadmap**:
- Full complex phase rotation: exp(iθ) = cos(θ) + i·sin(θ)
- Complex Tensor Train cores
- Complex einsum contraction

**Example**:
```python
from src.models.phase1 import HolographicTTEmbedding

embedding = HolographicTTEmbedding(
    vocab_size=50000,
    d_model=1024,
    rank=16,
    phase_encoding=True
)

# Phase 1: Real-valued output
input_ids = torch.randint(0, 50000, (4, 128))
output_real = embedding(input_ids)  # Shape: (4, 128, 1024), dtype: float32

# Phase 2 (experimental): Complex-valued output
output_complex = embedding.forward_complex(input_ids)  # dtype: complex64
```

### LNS Kernel

**Current Status**: Real-valued only

**Complex Support**: Not planned
- LNS operates in logarithmic domain (inherently real)
- Complex logarithm introduces branch cuts and ambiguity
- Will be disabled when complex operations are needed

**Recommendation**: Use standard matmul for complex operations

### BK-Core (BirmanSchwingerCore)

**Current Status**: Full complex support

**Complex Support**: Yes (complex64/complex128)
- Already supports complex-valued operations
- Uses complex128 internally for numerical stability
- Outputs complex64 for memory efficiency
- Resolvent G_ii is complex-valued

**Example**:
```python
# BK-Core already outputs complex tensors
G_ii = bk_core(...)  # dtype: torch.complex64
```

### Stability Monitor

**Current Status**: Complex-aware

**Complex Support**: Yes
- Can handle complex-valued resolvent G_ii
- Determinant computation works with complex tensors
- Eigenvalue tracking supports complex eigenvalues

### Fused Scan Kernel

**Current Status (Phase 1)**: Real-valued

**Complex Support**: Possible in Phase 2
- Triton kernel currently operates on real tensors
- Can be extended by treating complex as 2-channel real
- Fallback torch.cumsum already supports complex

## Utility Functions

### Type Checking

```python
from src.models.phase1 import is_complex_tensor

x = torch.randn(10, 20)
is_complex_tensor(x)  # False

x_complex = torch.complex(x, torch.zeros_like(x))
is_complex_tensor(x_complex)  # True
```

### Conversion Functions

```python
from src.models.phase1 import (
    real_to_complex,
    complex_to_real,
    ensure_complex,
    ensure_real,
)

# Ensure complex (convert if needed)
x = torch.randn(10, 20)
x_complex = ensure_complex(x)  # dtype: complex64

# Ensure real (extract real part if needed)
x_complex = torch.complex(torch.randn(10, 20), torch.randn(10, 20))
x_real = ensure_real(x_complex)  # dtype: float32
```

### Complex Phase Rotation

```python
from src.models.phase1 import complex_phase_rotation

x = torch.randn(10, 20, 128)
phase = torch.randn(128)

# Phase 1: Real approximation (cos(θ))
x_rotated_real = complex_phase_rotation(x, phase, use_full_complex=False)

# Phase 2: Full complex rotation (exp(iθ))
x_rotated_complex = complex_phase_rotation(x, phase, use_full_complex=True)
```

### Safe Mixed Operations

```python
from src.models.phase1 import safe_complex_operation

x_real = torch.randn(10, 20)
y_complex = torch.complex(torch.randn(10, 20), torch.randn(10, 20))

# Automatic conversion to complex
result = safe_complex_operation(x_real, y_complex, operation='add')
# result.dtype: complex64
```

### Dtype Compatibility Checking

```python
from src.models.phase1 import check_dtype_compatibility

x = torch.randn(10, 20)
y = torch.complex(torch.randn(10, 20), torch.randn(10, 20))

try:
    check_dtype_compatibility(x, y, operation="addition")
except TypeError as e:
    print(e)  # "Dtype mismatch in addition: ..."
```

## Complex Linear Layer

Phase 2で使用する複素数重みを持つ線形層:

```python
from src.models.phase1 import ComplexLinear

layer = ComplexLinear(in_features=128, out_features=256)

# Complex input
x = torch.complex(torch.randn(10, 128), torch.randn(10, 128))
y = layer(x)  # Shape: (10, 256), dtype: complex64

# Real input (automatically converted)
x_real = torch.randn(10, 128)
y = layer(x_real)  # Shape: (10, 256), dtype: complex64
```

## Documentation Functions

### Component Support Status

```python
from src.models.phase1 import document_complex_support

support_status = document_complex_support()
for component, info in support_status.items():
    print(f"{component}:")
    print(f"  Status: {info['status']}")
    print(f"  Phase 2 Ready: {info['phase2_ready']}")
    print(f"  Notes: {info['notes']}")
```

### Migration Guide

```python
from src.models.phase1 import get_complex_conversion_guide

guide = get_complex_conversion_guide()
print(guide)
```

## Phase 2 Migration Strategy

### 1. AR-SSM Layer

**Current (Phase 1)**:
```python
# Real-valued gates and projections
gates = self.complexity_gate(x)  # Real output
u = self.U_proj(x)  # Real projection
v = self.V_proj(x)  # Real projection
```

**Phase 2**:
```python
# Complex-valued gates for non-Hermitian dynamics
gates = complex_phase_rotation(
    self.complexity_gate(x),
    self.gate_phase,
    use_full_complex=True
)
u = self.U_proj_complex(x)  # ComplexLinear
v = self.V_proj_complex(x)  # ComplexLinear
```

### 2. HTT Embedding

**Current (Phase 1)**:
```python
# Real approximation: cos(θ)
phase_mod = torch.cos(self.phase_shift)
c1 = c1 * phase_mod
```

**Phase 2**:
```python
# Full complex rotation: exp(iθ)
c1 = complex_phase_rotation(
    c1,
    self.phase_shift,
    use_full_complex=True
)
```

### 3. Integration with BK-Core

**BK-Core → AR-SSM**:
```python
# BK-Core outputs complex
G_ii = bk_core(...)  # complex64

# Phase 1: Convert to real
features_real = ensure_real(G_ii)
output = ar_ssm(features_real)

# Phase 2: Keep complex
output = ar_ssm_complex(G_ii)  # Full complex support
```

## Testing Strategy

### Unit Tests

```python
def test_complex_conversion():
    """Test real ↔ complex conversion"""
    x_real = torch.randn(10, 20)
    x_complex = real_to_complex(x_real)
    assert is_complex_tensor(x_complex)
    
    x_back = ensure_real(x_complex)
    assert torch.allclose(x_real, x_back)

def test_complex_phase_rotation():
    """Test phase rotation"""
    x = torch.randn(10, 20, 128)
    phase = torch.randn(128)
    
    # Real approximation
    y_real = complex_phase_rotation(x, phase, use_full_complex=False)
    assert not is_complex_tensor(y_real)
    
    # Full complex
    y_complex = complex_phase_rotation(x, phase, use_full_complex=True)
    assert is_complex_tensor(y_complex)

def test_ar_ssm_complex_input():
    """Test AR-SSM with complex input"""
    layer = AdaptiveRankSemiseparableLayer(d_model=128, max_rank=32)
    
    x_complex = torch.complex(
        torch.randn(4, 100, 128),
        torch.randn(4, 100, 128)
    )
    
    with pytest.warns(UserWarning):
        y, diagnostics = layer(x_complex)
    
    assert diagnostics['input_was_complex']
    assert not is_complex_tensor(y)  # Phase 1: converts to real

def test_htt_complex_forward():
    """Test HTT complex forward pass"""
    embedding = HolographicTTEmbedding(
        vocab_size=1000,
        d_model=128,
        rank=8,
        phase_encoding=True
    )
    
    input_ids = torch.randint(0, 1000, (4, 50))
    
    # Phase 1: Real output
    output_real = embedding(input_ids)
    assert not is_complex_tensor(output_real)
    
    # Phase 2: Complex output
    output_complex = embedding.forward_complex(input_ids)
    assert is_complex_tensor(output_complex)
```

### Integration Tests

```python
def test_bk_core_ar_ssm_integration():
    """Test BK-Core → AR-SSM integration"""
    # Mock BK-Core output (complex)
    G_ii = torch.complex(
        torch.randn(4, 100, 128),
        torch.randn(4, 100, 128)
    )
    
    # Phase 1: Convert to real
    ar_ssm = AdaptiveRankSemiseparableLayer(d_model=128, max_rank=32)
    features_real = ensure_real(G_ii)
    output, _ = ar_ssm(features_real)
    
    assert not is_complex_tensor(output)

def test_mixed_real_complex_operations():
    """Test mixed real/complex operations"""
    x_real = torch.randn(10, 20)
    y_complex = torch.complex(torch.randn(10, 20), torch.randn(10, 20))
    
    # Safe operation with auto-conversion
    result = safe_complex_operation(x_real, y_complex, operation='add')
    assert is_complex_tensor(result)
    
    # Unsafe operation (should raise)
    with pytest.raises(TypeError):
        check_dtype_compatibility(x_real, y_complex, "addition")
```

## Performance Considerations

### Memory Usage

```python
# Real tensor: 4 bytes per element (float32)
x_real = torch.randn(1000, 1000)  # 4 MB

# Complex tensor: 8 bytes per element (complex64 = 2×float32)
x_complex = torch.complex(x_real, torch.zeros_like(x_real))  # 8 MB
```

### Computational Cost

- Complex addition: ~Same as real
- Complex multiplication: ~4× real (4 real multiplications + 2 additions)
- Complex matmul: ~4× real matmul

### Optimization Tips

1. **Use complex64 instead of complex128** when possible
2. **Avoid unnecessary conversions** between real and complex
3. **Batch complex operations** to amortize overhead
4. **Use in-place operations** when safe

## References

- Requirements: 11.1, 11.2, 11.3, 11.4, 11.5
- Design: Section "Phase 2 Preparation and Complex Number Support"
- PyTorch Complex Numbers: https://pytorch.org/docs/stable/complex_numbers.html

## Future Work (Phase 2)

1. **Non-Hermitian Forgetting Mechanism**
   - Complex eigenvalues for memory decay
   - Imaginary part controls forgetting rate
   
2. **Complex-Valued AR-SSM**
   - Complex gates and projections
   - Complex cumulative sum operations
   
3. **Complex HTT Embedding**
   - Full exp(iθ) phase rotation
   - Complex Tensor Train cores
   
4. **Gradient Flow Through Complex Operations**
   - Wirtinger derivatives
   - Complex-aware gradient clipping

## Contact

For questions or issues related to complex number support, please refer to:
- Design document: `.kiro/specs/phase1-efficiency-engine/design.md`
- Requirements: `.kiro/specs/phase1-efficiency-engine/requirements.md`
- Implementation: `src/models/phase1/complex_utils.py`
