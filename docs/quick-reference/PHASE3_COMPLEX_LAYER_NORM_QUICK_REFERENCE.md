# Phase 3: ComplexLayerNorm クイックリファレンス

## 概要

ComplexLayerNormは、複素平面上で正規化を行う層です。振幅と位相の両方を正規化することで、学習の安定性を向上させます。

## 基本的な使用方法

```python
from src.models.phase3.complex_ops import ComplexLayerNorm
from src.models.phase3.complex_tensor import ComplexTensor
import torch

# ComplexLayerNormの作成
norm = ComplexLayerNorm(normalized_shape=64, eps=1e-5)

# ComplexTensor入力
x = ComplexTensor(
    torch.randn(4, 10, 64, dtype=torch.float16),
    torch.randn(4, 10, 64, dtype=torch.float16)
)
y = norm(x)

# complex64入力
x_complex64 = torch.randn(4, 10, 64, dtype=torch.complex64)
y_complex64 = norm(x_complex64)
```

## 数学的定式化

```
z' = γ · (z - μ) / √(σ² + ε) + β

where:
    μ = E[z]  (複素平均)
    σ² = E[|z - μ|²]  (複素分散)
    γ, β: 学習可能なアフィン変換パラメータ（実数）
```

## パラメータ

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `normalized_shape` | int or tuple | - | 正規化する次元の形状 |
| `eps` | float | 1e-5 | 数値安定性のためのイプシロン |
| `elementwise_affine` | bool | True | アフィン変換を使用するか |

## 学習可能なパラメータ

- **gamma** (torch.Tensor): スケールパラメータ（実数）
  - 形状: `normalized_shape`
  - 初期値: 1.0

- **beta** (torch.Tensor): シフトパラメータ（実数）
  - 形状: `normalized_shape`
  - 初期値: 0.0

## 入力・出力

### 入力
- **ComplexTensor**: (B, N, D) または任意の形状
- **complex64**: (B, N, D) または任意の形状

### 出力
- 入力と同じ型・形状

## 物理的直観

複素正規化は、振幅と位相の両方を正規化します：

1. **振幅の正規化**: 情報の「強さ」を標準化
2. **位相の正規化**: 情報の「方向性」を標準化
3. **学習の安定化**: 勾配消失/爆発を防止

## 数値安定性

### ゼロ除算対策
```python
# 分散にイプシロンを加算
std = torch.sqrt(var + self.eps)  # eps = 1e-5
```

### オーバーフロー対策
```python
# float32で計算してからfloat16に戻す
var = centered.abs_squared().mean(dim=dims, keepdim=True)
```

## 使用例

### 基本的な使用
```python
# 64次元の特徴量を正規化
norm = ComplexLayerNorm(64)

# 入力: (batch=4, seq=10, features=64)
x = ComplexTensor(
    torch.randn(4, 10, 64, dtype=torch.float16),
    torch.randn(4, 10, 64, dtype=torch.float16)
)

# 正規化
y = norm(x)

# 正規化後の統計
print(f"Mean (real): {y.real.mean():.6f}")  # ~0.0
print(f"Mean (imag): {y.imag.mean():.6f}")  # ~0.0
print(f"Variance: {y.abs_squared().mean():.6f}")  # ~1.0
```

### アフィン変換なし
```python
# アフィン変換を無効化
norm = ComplexLayerNorm(64, elementwise_affine=False)

# gamma, betaは存在しない
assert norm.gamma is None
assert norm.beta is None
```

### 多次元正規化
```python
# 最後の2次元を正規化
norm = ComplexLayerNorm((10, 64))

# 入力: (batch=4, seq=10, features=64)
x = ComplexTensor(
    torch.randn(4, 10, 64, dtype=torch.float16),
    torch.randn(4, 10, 64, dtype=torch.float16)
)

# 正規化（最後の2次元で正規化）
y = norm(x)
```

### 学習での使用
```python
import torch.nn as nn
import torch.optim as optim

# モデルの定義
class ComplexModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = ComplexLayerNorm(64)
    
    def forward(self, x):
        return self.norm(x)

# モデルとオプティマイザー
model = ComplexModel()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 学習ループ
for epoch in range(10):
    # Forward pass
    x = torch.randn(4, 64, dtype=torch.complex64, requires_grad=True)
    y = model(x)
    
    # 損失計算
    loss = y.abs().sum()
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

## テスト

### 正規化の検証
```python
import torch

# ComplexLayerNorm
norm = ComplexLayerNorm(64, elementwise_affine=False)

# ランダム入力
x = torch.randn(4, 10, 64, dtype=torch.complex64)

# 正規化
y = norm(x)

# 平均が0に近いことを確認
mean_real = y.real.mean(dim=-1)
mean_imag = y.imag.mean(dim=-1)
assert torch.allclose(mean_real, torch.zeros_like(mean_real), atol=1e-5)
assert torch.allclose(mean_imag, torch.zeros_like(mean_imag), atol=1e-5)

# 分散が1に近いことを確認
var = (y.real ** 2 + y.imag ** 2).mean(dim=-1)
assert torch.allclose(var, torch.ones_like(var), atol=1e-1)
```

### 勾配の検証
```python
# ComplexLayerNorm
norm = ComplexLayerNorm(64)

# 入力（勾配計算を有効化）
x = torch.randn(4, 64, dtype=torch.complex64, requires_grad=True)

# Forward pass
y = norm(x)

# 損失計算
loss = y.abs().sum()

# Backward pass
loss.backward()

# 勾配が計算されていることを確認
assert norm.gamma.grad is not None
assert norm.beta.grad is not None
assert not torch.isnan(norm.gamma.grad).any()
assert not torch.isnan(norm.beta.grad).any()
```

## パフォーマンス

### メモリ使用量
- **ComplexTensor (float16)**: 4 bytes/element
- **complex64**: 8 bytes/element
- **メモリ削減**: 50%

### 計算量
- **時間複雑度**: O(N·D)
  - N: バッチサイズ × シーケンス長
  - D: 特徴量の次元数
- **空間複雑度**: O(D)（パラメータ）

## トラブルシューティング

### 問題: 正規化後の分散が1にならない

**原因**: 入力の分散が極端に小さいまたは大きい

**解決策**:
```python
# イプシロンを調整
norm = ComplexLayerNorm(64, eps=1e-4)  # デフォルト: 1e-5
```

### 問題: 勾配がNaNになる

**原因**: 入力に極端な値が含まれている

**解決策**:
```python
# 入力をクリップ
x = torch.clamp(x, min=-10, max=10)
y = norm(x)
```

### 問題: メモリ不足

**原因**: complex64を使用している

**解決策**:
```python
# ComplexTensorを使用（50%メモリ削減）
from src.models.phase3.complex_tensor import ComplexTensor

x = ComplexTensor(
    torch.randn(4, 10, 64, dtype=torch.float16),
    torch.randn(4, 10, 64, dtype=torch.float16)
)
y = norm(x)
```

## 関連モジュール

- **ComplexTensor**: 複素数テンソルの基本実装
- **ComplexLinear**: 複素線形層
- **ModReLU**: 位相保存活性化関数

## 参考文献

1. Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). Layer normalization. arXiv preprint arXiv:1607.06450.
2. Trabelsi, C., et al. (2017). Deep complex networks. arXiv preprint arXiv:1705.09792.

## 更新履歴

- 2025-11-21: 初版作成
- Task 4完了: ComplexLayerNormの実装とテスト
