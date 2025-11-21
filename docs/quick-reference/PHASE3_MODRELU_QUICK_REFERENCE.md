# Phase 3 ModReLU - クイックリファレンス

## 概要

ModReLU（Modulus ReLU）は、複素数の振幅をフィルタリングしながら位相を保存する活性化関数です。

## 基本的な使い方

```python
from src.models.phase3.complex_ops import ModReLU
from src.models.phase3.complex_tensor import ComplexTensor
import torch

# ModReLU層の作成
modrelu = ModReLU(features=64, use_half=True)

# ComplexTensor入力
x = ComplexTensor(
    torch.randn(4, 10, 64, dtype=torch.float16),
    torch.randn(4, 10, 64, dtype=torch.float16)
)
y = modrelu(x)

# complex64入力
x_complex = torch.randn(4, 10, 64, dtype=torch.complex64)
y_complex = modrelu(x_complex)
```

## 数式

```
z' = ReLU(|z| + b) · z / |z|

where:
    |z| = √(real² + imag²)  (振幅)
    z / |z|  (位相を保存)
    b  (学習可能なバイアス)
```

## パラメータ

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| features | int | - | 特徴量の次元数 |
| use_half | bool | True | float16を使用するか |

## 物理的直観

### 振幅フィルタリング
- 情報の「強さ」を表す振幅をReLUでフィルタリング
- 弱い信号を抑制し、強い信号を通過させる

### 位相保存
- 情報の「方向性」を表す位相を保存
- 文脈や意味の方向性を維持

## 数値安定性

### ゼロ除算対策
```python
mag_safe = mag + 1e-6  # イプシロンを加算
phase = z / mag_safe   # 安全な除算
```

### 勾配の健全性
- バイアスパラメータの勾配が正常に計算される
- NaN/Infが発生しない

## メモリ効率

| 設定 | メモリ使用量 |
|------|-------------|
| use_half=True | 2 bytes/parameter |
| use_half=False | 4 bytes/parameter |

## テスト

```bash
# ModReLUのテストを実行
python -m pytest tests/test_complex_ops.py::TestModReLU -v
```

## 例: 位相保存の確認

```python
import torch
from src.models.phase3.complex_ops import ModReLU

modrelu = ModReLU(64, use_half=False)
x = torch.randn(4, 64, dtype=torch.complex64)

# 位相を計算（活性化前）
phase_before = torch.angle(x)

# Forward pass
y = modrelu(x)

# 位相を計算（活性化後）
phase_after = torch.angle(y)

# 位相が保存されていることを確認
mask = torch.abs(x) > 0.1  # 振幅が十分大きい要素のみ
print(torch.allclose(phase_before[mask], phase_after[mask], atol=1e-3))
# True
```

## 例: 振幅フィルタリングの確認

```python
import torch
from src.models.phase3.complex_ops import ModReLU

modrelu = ModReLU(64, use_half=False)
modrelu.bias.data.fill_(-1.0)  # バイアスを-1に設定

# 小さい振幅の入力
x = torch.full((4, 64), 0.5+0.5j, dtype=torch.complex64)
# |x| ≈ 0.707

# Forward pass
y = modrelu(x)

# ReLU(0.707 - 1.0) = ReLU(-0.293) = 0
# 振幅がゼロになることを確認
print(torch.abs(y).max())  # ~0.0
```

## トラブルシューティング

### Q: 位相が保存されない
A: 振幅が非常に小さい（< 1e-6）場合、数値誤差により位相が不安定になります。入力の振幅を確認してください。

### Q: 勾配がNaNになる
A: バイアスの初期値が極端に大きい/小さい可能性があります。デフォルトのゼロ初期化を使用してください。

### Q: メモリ不足
A: `use_half=True` を設定してfloat16を使用してください。

## 関連ドキュメント

- [ComplexTensor Quick Reference](PHASE3_COMPLEX_TENSOR_QUICK_REFERENCE.md)
- [ComplexLinear Quick Reference](PHASE3_COMPLEX_OPS_QUICK_REFERENCE.md)
- [Phase 3 Implementation Guide](../PHASE3_IMPLEMENTATION_GUIDE.md)

## Requirements

- Requirement 1.9: ModReLU数式の実装
- Requirement 1.10: ModReLU単体テストの実装

---

**最終更新**: 2025-11-21  
**ステータス**: ✅ 実装完了
