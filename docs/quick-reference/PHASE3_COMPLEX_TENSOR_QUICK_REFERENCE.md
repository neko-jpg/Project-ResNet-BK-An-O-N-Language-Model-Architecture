# Phase 3: ComplexTensor クイックリファレンス

## 概要

ComplexTensorは、メモリ使用量を50%削減する半精度複素数（complex32）データ構造です。

## 基本的な使い方

### インポート
```python
from src.models.phase3.complex_tensor import ComplexTensor
import torch
```

### 作成方法

#### 1. 実部と虚部から作成
```python
real = torch.randn(4, 10, 64, dtype=torch.float16)
imag = torch.randn(4, 10, 64, dtype=torch.float16)
z = ComplexTensor(real, imag)
```

#### 2. 実数から作成
```python
# 虚部なし（ゼロ）
real = torch.randn(4, 10, 64)
z = ComplexTensor.from_real(real)

# 虚部あり
imag = torch.randn(4, 10, 64)
z = ComplexTensor.from_real(real, imag)
```

#### 3. complex64から変換
```python
z_complex64 = torch.randn(4, 10, 64, dtype=torch.complex64)
z = ComplexTensor.from_complex64(z_complex64)
```

## 複素数演算

### 基本演算
```python
z1 = ComplexTensor(torch.ones(2, 3, dtype=torch.float16), 
                   torch.zeros(2, 3, dtype=torch.float16))
z2 = ComplexTensor(torch.zeros(2, 3, dtype=torch.float16), 
                   torch.ones(2, 3, dtype=torch.float16))

# 加算: (1 + 0i) + (0 + 1i) = (1 + 1i)
z3 = z1 + z2

# 減算: (1 + 0i) - (0 + 1i) = (1 - 1i)
z4 = z1 - z2

# 乗算: (1 + 0i) * (0 + 1i) = (0 + 1i)
z5 = z1 * z2

# 除算: (1 + 0i) / (0 + 1i) = (0 - 1i)
z6 = z1 / z2
```

### スカラー演算
```python
z = ComplexTensor(torch.ones(2, 3, dtype=torch.float16), 
                  torch.ones(2, 3, dtype=torch.float16))

# スカラー加算: (1 + 1i) + 2 = (3 + 1i)
z_plus_2 = z + 2

# スカラー乗算: (1 + 1i) * 2 = (2 + 2i)
z_times_2 = z * 2
```

### 複素数関数
```python
# 共役: conj(a + bi) = a - bi
z_conj = z.conj()

# 絶対値: |a + bi| = √(a² + b²)
abs_z = z.abs()

# 偏角: arg(a + bi) = arctan2(b, a)
angle_z = z.angle()

# 絶対値の2乗（高速）: |a + bi|² = a² + b²
abs_squared_z = z.abs_squared()
```

## デバイス管理

```python
# CPU → CUDA
z_cuda = z.cuda()

# CUDA → CPU
z_cpu = z_cuda.cpu()

# 任意のデバイスへ
z_device = z.to('cuda:0')
```

## 形状操作

```python
z = ComplexTensor(torch.randn(4, 10, 64, dtype=torch.float16),
                  torch.randn(4, 10, 64, dtype=torch.float16))

# 形状変更
z_view = z.view(4, 640)
z_reshape = z.reshape(40, 64)

# 次元入れ替え
z_permute = z.permute(2, 0, 1)  # (64, 4, 10)
z_transpose = z.transpose(0, 1)  # (10, 4, 64)

# 次元追加/削除
z_unsqueeze = z.unsqueeze(0)  # (1, 4, 10, 64)
z_squeeze = z_unsqueeze.squeeze(0)  # (4, 10, 64)
```

## 統計関数

```python
# 平均
z_mean = z.mean(dim=1)  # 次元1で平均
z_mean_all = z.mean()   # 全体の平均

# 合計
z_sum = z.sum(dim=1)    # 次元1で合計
z_sum_all = z.sum()     # 全体の合計

# ノルム
norm_l2 = z.norm(p=2)   # L2ノルム
norm_l1 = z.norm(p=1)   # L1ノルム
```

## 勾配計算

```python
# 勾配計算を有効化
z = z.requires_grad_(True)

# 勾配計算グラフから切り離し
z_detach = z.detach()

# 複製（勾配計算グラフを保持）
z_clone = z.clone()
```

## PyTorch互換性

```python
# ComplexTensor → complex64
z_complex64 = z.to_complex64()

# complex64 → ComplexTensor
z_back = ComplexTensor.from_complex64(z_complex64)

# 既存のPyTorchコードとの統合
def my_function(z):
    if isinstance(z, ComplexTensor):
        z = z.to_complex64()
    # 既存のPyTorchコードで処理
    result = torch.fft.fft(z)
    return ComplexTensor.from_complex64(result)
```

## メモリ効率

### メモリ使用量の比較
```python
# complex64: 8 bytes/element
z_complex64 = torch.randn(1000, 1000, dtype=torch.complex64)

# ComplexTensor (complex32): 4 bytes/element (50%削減)
real = torch.randn(1000, 1000, dtype=torch.float16)
imag = torch.randn(1000, 1000, dtype=torch.float16)
z_complex32 = ComplexTensor(real, imag)
```

### メモリレイアウト
```python
# Planar形式: 実部と虚部を分離
# [RRR...] [III...]
# - CUDAのcoalesced accessに最適
# - 実部と虚部を独立して処理可能
```

## 数値安定性

### ゼロ除算対策
```python
# 自動的に小さな値（1e-8）を加算
z1 = ComplexTensor(torch.tensor([[1.0]], dtype=torch.float16),
                   torch.tensor([[1.0]], dtype=torch.float16))
z2 = ComplexTensor(torch.tensor([[0.0]], dtype=torch.float16),
                   torch.tensor([[0.0]], dtype=torch.float16))
z3 = z1 / z2  # NaN/Infにならない
```

### オーバーフロー対策
```python
# 中間計算でfloat32を使用
z1 = ComplexTensor(torch.tensor([[1000.0]], dtype=torch.float16),
                   torch.tensor([[1000.0]], dtype=torch.float16))
z2 = z1 * z1  # オーバーフローを回避
```

## 実装例

### 複素数ニューラルネットワーク層
```python
import torch.nn as nn

class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight_real = nn.Parameter(torch.randn(out_features, in_features, dtype=torch.float16))
        self.weight_imag = nn.Parameter(torch.randn(out_features, in_features, dtype=torch.float16))
    
    def forward(self, z: ComplexTensor) -> ComplexTensor:
        # 複素行列積: (A + iB)(x + iy) = (Ax - By) + i(Bx + Ay)
        real_out = torch.matmul(z.real, self.weight_real.t()) - \
                   torch.matmul(z.imag, self.weight_imag.t())
        imag_out = torch.matmul(z.real, self.weight_imag.t()) + \
                   torch.matmul(z.imag, self.weight_real.t())
        return ComplexTensor(real_out, imag_out)
```

### 複素数活性化関数（ModReLU）
```python
class ModReLU(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(features, dtype=torch.float16))
    
    def forward(self, z: ComplexTensor) -> ComplexTensor:
        # z' = ReLU(|z| + b) · z / |z|
        mag = z.abs()
        phase = z / (mag + 1e-6)  # 位相保存
        new_mag = torch.relu(mag + self.bias)
        return phase * new_mag
```

## トラブルシューティング

### Q: NaN/Infが発生する
```python
# A: 数値安定性対策を確認
# 1. ゼロ除算対策が有効か確認
# 2. 中間計算でfloat32を使用しているか確認
# 3. 入力データの範囲を確認
```

### Q: メモリ使用量が期待より多い
```python
# A: データ型を確認
# 1. real, imagがfloat16であることを確認
assert z.real.dtype == torch.float16
assert z.imag.dtype == torch.float16

# 2. 不要な中間変数を削除
del intermediate_result
torch.cuda.empty_cache()
```

### Q: 勾配が正しく計算されない
```python
# A: requires_gradを確認
z = z.requires_grad_(True)

# 勾配計算後に確認
assert z.real.grad is not None
assert z.imag.grad is not None
```

## パフォーマンスTips

### 1. Planar形式を活用
```python
# 実部と虚部を独立して処理
real_processed = process_real(z.real)
imag_processed = process_imag(z.imag)
z_result = ComplexTensor(real_processed, imag_processed)
```

### 2. abs_squared()を使用
```python
# abs()よりも高速（sqrtを回避）
abs_squared = z.abs_squared()  # a² + b²
```

### 3. バッチ処理
```python
# 小さなバッチを避ける
# 推奨: batch_size >= 4
z = ComplexTensor(torch.randn(16, 512, 512, dtype=torch.float16),
                  torch.randn(16, 512, 512, dtype=torch.float16))
```

## 参考資料

- 設計書: `.kiro/specs/phase3-physics-transcendence/design.md`
- 要件定義: `.kiro/specs/phase3-physics-transcendence/requirements.md`
- テストコード: `tests/test_complex_tensor.py`
- 完了報告: `results/benchmarks/PHASE3_TASK1_COMPLETION_SUMMARY.md`

---

**最終更新**: 2025-11-21  
**バージョン**: 1.0.0
