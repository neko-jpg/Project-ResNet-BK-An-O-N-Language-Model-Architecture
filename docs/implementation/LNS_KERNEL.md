# LNS (Logarithmic Number System) Kernel Implementation

## Overview

LNS (Logarithmic Number System) Kernelは、乗算器(FMA)を加算器(ADD)に変換することで、推論時の計算コストと消費電力を大幅に削減する実験的なTritonカーネルです。

**Status**: ⚠️ **EXPERIMENTAL** - 推論専用、精度低下あり

## Mathematical Foundation

### Standard Matrix Multiplication

```
C = A @ B = Σₖ A[i,k] × B[k,j]
```

各要素の計算には K 回の乗算と K-1 回の加算が必要です。

### LNS Transformation

対数領域では、乗算が加算に変換されます：

```
a × b = exp(log(a) + log(b))
```

行列積の場合：

```
C[i,j] = Σₖ A[i,k] × B[k,j]
log(C[i,j]) = log(Σₖ exp(log(A[i,k]) + log(B[k,j])))
```

### Max-Log Approximation

完全なLNS実装では、対数領域での加算に高価な指数関数が必要です。
Max-Log近似を使用することで、これを回避します：

```
log(x + y) ≈ max(log(x), log(y))  when x >> y or y >> x
```

行列積への適用：

```
log(C[i,j]) ≈ maxₖ(log(A[i,k]) + log(B[k,j]))
```

## Physical Intuition (物理的直観)

### 計算コスト削減

- **FMA (Fused Multiply-Add)**: 乗算と加算を1命令で実行
  - 消費電力: 高い（ADDの3-5倍）
  - スループット: 低い（ADDの約半分）

- **ADD (Addition)**: 加算のみ
  - 消費電力: 低い
  - スループット: 高い

LNSカーネルは、すべてのFMA命令をADD命令に置き換えることで、
推論時の消費電力を削減し、スループットを向上させます。

### Max-Log近似の意味

Max-Log近似は、支配的な項のみを保持します：

```
log(100 + 1) ≈ log(100) = 4.605
正確な値: log(101) = 4.615
誤差: 0.01 (0.2%)
```

ニューラルネットワークでは、活性化がスパースな場合、
この近似は非常に正確です。

## Implementation Details

### Triton Kernel

```python
@triton.jit
def lns_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Block-level parallelism
    pid = tl.program_id(axis=0)
    
    # Initialize accumulator with -inf (log domain zero)
    accumulator = tl.full((BLOCK_SIZE_M, BLOCK_SIZE_N), float('-inf'), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load blocks
        a = tl.load(a_ptrs, mask=a_mask, other=float('-inf'))
        b = tl.load(b_ptrs, mask=b_mask, other=float('-inf'))
        
        # Log-domain multiplication: log(a*b) = log(a) + log(b)
        log_prod = a + b
        
        # Max-log accumulation: log(sum) ≈ max(log terms)
        accumulator = tl.maximum(accumulator, log_prod)
    
    # Store result
    tl.store(c_ptrs, accumulator, mask=c_mask)
```

### Python Wrapper

```python
def lns_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    LNS matrix multiplication.
    
    Args:
        a: (M, K) matrix in log domain
        b: (K, N) matrix in log domain
    
    Returns:
        c: (M, N) matrix in log domain
    """
    # Validate inputs
    assert a.is_cuda and b.is_cuda
    assert a.shape[1] == b.shape[0]
    
    # Launch kernel
    lns_matmul_kernel[grid](...)
    
    return c
```

### LNSLinear Layer

```python
class LNSLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, use_lns=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features)) if bias else None
        self.use_lns = use_lns
        
        # Pre-computed log weights for inference
        self.register_buffer('log_weight', None)
        self.register_buffer('weight_sign', None)
    
    def prepare_lns_weights(self):
        """Convert weights to log domain (call once before inference)."""
        with torch.no_grad():
            self.log_weight = torch.log(torch.abs(self.weight) + 1e-8)
            self.weight_sign = torch.sign(self.weight)
    
    def forward(self, x):
        if self.training or not self.use_lns:
            # Training: standard computation
            return F.linear(x, self.weight, self.bias)
        
        # Inference: LNS computation
        if self.log_weight is None:
            self.prepare_lns_weights()
        
        # Convert input to log domain
        log_x = torch.log(torch.abs(x) + 1e-8)
        x_sign = torch.sign(x)
        
        # LNS matmul
        log_out = lns_matmul(log_x, self.log_weight.T)
        
        # Compute sign
        out_sign = torch.sign(torch.matmul(x_sign, self.weight_sign.T))
        
        # Convert back to linear domain
        out = out_sign * torch.exp(log_out)
        
        # Add bias
        if self.bias is not None:
            out = out + self.bias
        
        return out
```

## Usage

### Basic Usage

```python
from src.kernels.lns_kernel import lns_matmul
from src.models.phase1 import LNSLinear

# Direct kernel usage
a = torch.log(torch.abs(torch.randn(128, 256, device='cuda')) + 0.1)
b = torch.log(torch.abs(torch.randn(256, 512, device='cuda')) + 0.1)
c = lns_matmul(a, b)  # Result in log domain

# Using LNSLinear layer
layer = LNSLinear(512, 256, use_lns=True)
layer.eval()  # Switch to inference mode
layer.prepare_lns_weights()  # Pre-compute log weights

x = torch.randn(32, 512, device='cuda')
y = layer(x)  # Uses LNS kernel internally
```

### Converting Existing Models

```python
from src.models.phase1 import convert_linear_to_lns

# Load trained model
model = MyModel()
model.load_state_dict(torch.load('checkpoint.pt'))

# Convert all Linear layers to LNSLinear
model = convert_linear_to_lns(model)
model.eval()

# Prepare all LNS weights
for module in model.modules():
    if isinstance(module, LNSLinear):
        module.prepare_lns_weights()

# Now inference uses LNS kernel
output = model(input)
```

## Performance Characteristics

### Speedup

理論的には、LNSカーネルは標準matmulより高速になるはずですが、
実際のスループットはGPUアーキテクチャに依存します。

**Expected Performance** (based on design targets):
- Small matrices (< 512x512): 0.8-1.2x (overhead dominant)
- Medium matrices (512-1024): 1.0-1.5x
- Large matrices (> 1024): 1.2-2.0x

**Actual Performance** (measured):
Run `python scripts/benchmark_lns_kernel.py` to measure on your hardware.

### Accuracy

Max-Log近似により、精度が低下します：

**Expected Accuracy Loss**:
- Sparse activations: < 1% relative error
- Dense activations: 5-15% relative error
- Worst case: 20-30% relative error

**Mitigation Strategies**:
1. Use only for inference (not training)
2. Apply to specific layers (not all layers)
3. Fine-tune model with LNS enabled
4. Use higher precision (FP32 instead of FP16)

### Memory Usage

LNSカーネルは、標準matmulと同等のメモリを使用します。
ただし、log変換のための追加バッファが必要です。

## Limitations and Caveats

### ⚠️ Experimental Status

このカーネルは実験的であり、本番環境での使用は推奨されません。

### Accuracy Loss

Max-Log近似により、精度が低下します。特に：
- 密な活性化パターン
- 小さな値が多い場合
- 符号が頻繁に変わる場合

### Training Not Supported

LNSカーネルは推論専用です。学習時は標準matmulを使用します。

理由：
- Max-Log近似は微分不可能
- Straight-through estimatorを使用しても勾配が不正確
- 学習の収束が困難

### Sign Handling

符号の処理が近似的です：
```python
sign(out) = sign(x) @ sign(weight)
```

これは、すべての項が同じ符号の場合のみ正確です。

### Hardware Requirements

- CUDA-capable GPU required
- Triton must be installed
- CPU fallback not available (raises error)

## Benchmarking

### Running Benchmarks

```bash
# Full benchmark suite
python scripts/benchmark_lns_kernel.py

# Custom parameters
python scripts/benchmark_lns_kernel.py \
    --num-warmup 20 \
    --num-iterations 200 \
    --num-trials 20 \
    --output results/my_benchmark.json
```

### Interpreting Results

ベンチマーク結果は3つのセクションに分かれています：

1. **Speedup**: LNSカーネルの速度向上
   - 1.0x未満: LNSが遅い（オーバーヘッド支配的）
   - 1.0-1.5x: 中程度の改善
   - 1.5x以上: 大幅な改善

2. **Accuracy**: 精度低下の測定
   - < 5%: 許容範囲
   - 5-15%: 注意が必要
   - > 15%: 使用を再検討

3. **Memory**: メモリ使用量
   - 通常、標準matmulと同等

## Future Work

### Phase 2 Integration

Phase 2では、複素数対応のLNSカーネルを検討します：
- 複素数の対数: log(z) = log(|z|) + i·arg(z)
- 位相の処理: arg(z₁·z₂) = arg(z₁) + arg(z₂)

### Improved Approximations

Max-Log近似の改善：
- Log-Sum-Exp with correction term
- Learned approximation functions
- Adaptive precision based on activation statistics

### Training Support

学習時のLNS使用：
- Straight-through estimator with gradient correction
- Mixed-precision training (LNS for forward, FP32 for backward)
- Curriculum learning (gradually introduce LNS)

## References

### Requirements
- 3.1: Log-domain matrix multiplication
- 3.2: Max-log accumulation
- 3.3: Inference-only usage
- 3.4: Pre-computed log weights
- 3.5: Training fallback
- 3.6: Accuracy measurement
- 11.6: Phase 2 preparation
- 12.1: Documentation

### Related Work
- Arnold, M. G., et al. (2011). "Logarithmic Number Systems: Theory and Applications"
- Coleman, J. N., et al. (2008). "The European Logarithmic Microprocessor"
- Miyashita, D., et al. (2016). "Convolutional Neural Networks using Logarithmic Data Representation"

### Design Document
See `.kiro/specs/phase1-efficiency-engine/design.md` Section 3: "Logarithmic Number System (LNS) Kernel"

## Contact

For questions or issues related to LNS kernel:
- Open an issue on GitHub
- Refer to AGENTS.md for development guidelines
- Check TROUBLESHOOTING.md for common problems
