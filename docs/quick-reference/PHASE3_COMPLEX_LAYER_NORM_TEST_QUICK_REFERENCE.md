# ComplexLayerNorm Test Quick Reference

## テスト実行

```bash
# ComplexLayerNormのテストのみ実行
pytest tests/test_complex_ops.py::TestComplexLayerNorm -v

# すべての複素演算テストを実行
pytest tests/test_complex_ops.py -v
```

## テストケース概要

### 1. test_normalization_accuracy
正規化後の平均が0、分散が1に近いことを確認

**期待値**:
- 平均（実部）: ~0.0 (±1e-5)
- 平均（虚部）: ~0.0 (±1e-5)
- 分散: ~1.0 (±1e-1)

### 2. test_gradient_computation
勾配計算が正常に動作することを確認

**検証項目**:
- gamma.grad が計算される
- beta.grad が計算される
- NaN/Inf が含まれない

### 3. test_affine_transformation
アフィン変換が正しく動作することを確認

**検証式**:
```
y = gamma * y_norm + beta
```

### 4. test_complex_tensor_input
ComplexTensor入力でも正常に動作することを確認

**入力形式**:
- ComplexTensor (float16)
- 3次元: (batch, seq, features)

## 使用例

```python
from src.models.phase3.complex_ops import ComplexLayerNorm
from src.models.phase3.complex_tensor import ComplexTensor
import torch

# ComplexLayerNormの作成
norm = ComplexLayerNorm(64)

# ComplexTensor入力
x = ComplexTensor(
    torch.randn(4, 10, 64, dtype=torch.float16),
    torch.randn(4, 10, 64, dtype=torch.float16)
)

# 正規化
y = norm(x)

# 平均と分散の確認
mean = y.mean(dim=(0, 1))
var = y.abs_squared().mean(dim=(0, 1))

print(f"Mean magnitude: {mean.abs().mean():.6f}")  # ~0.0
print(f"Variance: {var.mean():.6f}")  # ~1.0
```

## 数式

### 複素正規化
```
z' = γ · (z - μ) / √(σ² + ε) + β

where:
    μ = E[z]  (複素平均)
    σ² = E[|z - μ|²]  (複素分散)
    γ, β: 学習可能なアフィン変換パラメータ（実数）
```

## トラブルシューティング

### 平均が0にならない
- アフィン変換が有効になっていないか確認
- `elementwise_affine=False` で正規化のみをテスト

### 分散が1にならない
- 許容誤差を確認（1e-1は妥当）
- 入力データの分布を確認

### 勾配がNaN
- 入力にNaN/Infが含まれていないか確認
- イプシロン（eps）の値を確認

## 関連ファイル

- 実装: `src/models/phase3/complex_ops.py`
- テスト: `tests/test_complex_ops.py`
- ComplexTensor: `src/models/phase3/complex_tensor.py`

## 要件

- Requirement 1.11: 複素平均と複素分散の計算
- Requirement 1.12: アフィン変換（実数パラメータ）

## ステータス

✅ 実装完了  
✅ テスト合格（4/4）  
✅ 要件満足
